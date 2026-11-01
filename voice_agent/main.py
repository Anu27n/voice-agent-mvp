"""HTTP and WebSocket endpoints.

POST /sessions {timezone?}      -> {session_id, input_sample_rate, frame_ms, output_sample_rate}
GET  /sessions                  -> recent calls, newest first
GET  /sessions/{id}             -> {session, transcript, metrics, errors}
GET  /sessions/{id}/transcript.md
GET  /metrics/summary?last=20   -> median / p90 per stage
GET  /metrics                   -> Prometheus text format
GET  /healthz                   -> providers, demo mode, active calls
WS   /ws/voice?session_id=...   -> binary audio + JSON events (see docs/DESIGN.md)
GET  /                          -> the web client (/history.html: call history)

When ACCESS_CODE is set, everything except /healthz and the static client needs
it, as an ``X-Access-Code`` header or a ``code`` query parameter.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from .agent import Agent
from .config import Settings
from .metrics import summarize
from .pipeline import VoicePipeline
from .providers import LLMProvider, STTProvider, TTSProvider, build_llm, build_stt, build_tts
from .report import prometheus_text, transcript_markdown
from .session import Session, SessionManager
from .store import Store
from .tools import ToolExecutor, default_tools, is_valid_timezone

log = logging.getLogger(__name__)

DEFAULT_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class SessionRequest(BaseModel):
    timezone: str | None = None  # the caller's IANA time zone, e.g. "Asia/Kolkata"


class WebSocketTransport:
    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws

    async def send_text(self, data: str) -> None:
        await self._ws.send_text(data)

    async def send_bytes(self, data: bytes) -> None:
        await self._ws.send_bytes(data)

    async def close(self, code: int = 1000) -> None:
        if self._ws.application_state == WebSocketState.CONNECTED:
            await self._ws.close(code)


def create_app(
    settings: Settings | None = None,
    *,
    stt: STTProvider | None = None,
    llm: LLMProvider | None = None,
    tts: TTSProvider | None = None,
    store: Store | None = None,
) -> FastAPI:
    settings = (settings or Settings.from_env()).resolve_auto()
    stt = stt or build_stt(settings)
    llm = llm or build_llm(settings)
    tts = tts or build_tts(settings)
    store = store or Store(settings.db_path)
    manager = SessionManager(store, settings)
    weather_http = httpx.AsyncClient(timeout=4.0)

    def make_agent(session: Session) -> Agent:
        # Per call: the time tool defaults to the caller's zone, and end_call
        # hangs up this call only.
        def hang_up() -> None:
            session.hangup_requested = True

        tools = default_tools(
            session.timezone or settings.default_timezone,
            weather=settings.enable_weather,
            on_end_call=hang_up if settings.allow_agent_hangup else None,
            http_client=weather_http,
        )
        return Agent(
            llm,
            ToolExecutor(tools, timeout_s=settings.tool_timeout_s),
            settings.system_prompt,
            timeout_s=settings.llm_timeout_s,
            max_tool_rounds=settings.max_tool_rounds,
        )

    def code_ok(presented: str | None) -> bool:
        expected = settings.access_code
        return not expected or (presented is not None and hmac.compare_digest(presented, expected))

    def require_access(request: Request) -> None:
        presented = request.headers.get("x-access-code") or request.query_params.get("code")
        if not code_ok(presented):
            raise HTTPException(status_code=401, detail="access code required")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        log.info(
            "providers: stt=%s llm=%s tts=%s%s",
            stt.name,
            llm.name,
            tts.name,
            " (demo mode)" if settings.demo_mode else "",
        )
        yield
        for provider in (stt, llm, tts):
            await provider.aclose()
        await weather_http.aclose()
        store.close()

    app = FastAPI(title="Voice Agent MVP", version="0.2.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.sessions = manager
    app.state.providers = {"stt": stt, "llm": llm, "tts": tts}
    protected = [Depends(require_access)]

    @app.post("/sessions", status_code=201, dependencies=protected)
    async def create_session(request: SessionRequest | None = None) -> dict:
        if manager.active >= settings.max_active_calls:
            raise HTTPException(status_code=429, detail="Too many calls in progress. Please try again shortly.")
        timezone = request.timezone if request else None
        if timezone and not is_valid_timezone(timezone):
            log.warning("ignoring unknown caller time zone %r", timezone)
            timezone = None
        session = await manager.create(timezone=timezone)
        return {
            "session_id": session.id,
            "input_sample_rate": settings.input_sample_rate,
            "frame_ms": settings.frame_ms,
            "output_sample_rate": tts.sample_rate,
        }

    @app.get("/sessions", dependencies=protected)
    async def list_sessions(limit: int = Query(20, ge=1, le=200)) -> dict:
        sessions = await store.list_sessions(limit)
        for item in sessions:
            live = manager.get(item["id"])
            item["state"] = live.state.value if live else ("ended" if item["ended_at"] else "idle")
        return {"sessions": sessions}

    async def load_session(session_id: str) -> dict:
        data = await store.get_session(session_id)
        if data is None:
            raise HTTPException(status_code=404, detail="session not found")
        live = manager.get(session_id)
        data["session"]["state"] = live.state.value if live else ("ended" if data["session"]["ended_at"] else "idle")
        return data

    @app.get("/sessions/{session_id}", dependencies=protected)
    async def get_session(session_id: str) -> dict:
        return await load_session(session_id)

    @app.get("/sessions/{session_id}/transcript.md", dependencies=protected, response_class=PlainTextResponse)
    async def get_transcript(session_id: str) -> PlainTextResponse:
        markdown = transcript_markdown(await load_session(session_id))
        headers = {"Content-Disposition": f'inline; filename="call-{session_id[:8]}.md"'}
        return PlainTextResponse(markdown, media_type="text/markdown; charset=utf-8", headers=headers)

    @app.get("/metrics/summary", dependencies=protected)
    async def metrics_summary(last: int = Query(20, ge=1, le=1000)) -> dict:
        rows = await asyncio.to_thread(store.recent_metrics, last)
        return {"turns": len(rows), "stages": summarize(rows)}

    @app.get("/metrics", dependencies=protected, response_class=PlainTextResponse)
    async def metrics() -> PlainTextResponse:
        counts = await asyncio.to_thread(store.counts)
        recent = await asyncio.to_thread(store.recent_metrics, 100)
        return PlainTextResponse(
            prometheus_text(counts, recent, manager.active), media_type="text/plain; version=0.0.4"
        )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "providers": {"stt": stt.name, "llm": llm.name, "tts": tts.name},
            "demo_mode": settings.demo_mode,
            "access_code_required": bool(settings.access_code),
            "active_sessions": manager.active,
            "max_active_calls": settings.max_active_calls,
        }

    @app.websocket("/ws/voice")
    async def voice(ws: WebSocket, session_id: str = Query(""), code: str | None = Query(None)) -> None:
        await ws.accept()
        if not code_ok(code):
            await ws.send_text(
                json.dumps({"type": "error", "code": "unauthorized", "message": "Access code required."})
            )
            await ws.close(code=4401)
            return
        session = manager.claim(session_id)
        if session is None:
            await ws.send_text(
                json.dumps({"type": "error", "code": "invalid_session", "message": "Unknown, ended or busy session."})
            )
            await ws.close(code=4404)
            return

        pipeline = VoicePipeline(
            session,
            settings,
            stt=stt,
            agent=make_agent(session),
            tts=tts,
            store=store,
            transport=WebSocketTransport(ws),
            on_ended=lambda: manager.release(session.id),
        )
        await pipeline.start()
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await pipeline.on_audio(message["bytes"])
                elif message.get("text") is not None:
                    await pipeline.on_message(message["text"])
        except (WebSocketDisconnect, RuntimeError):
            pass  # RuntimeError: receive() after the server closed the socket
        finally:
            await pipeline.on_disconnect()
            manager.release(session.id)

    web_dir = Path(settings.web_dir) if settings.web_dir else DEFAULT_WEB_DIR
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    else:
        log.warning("web client not found at %s; serving the API only", web_dir)
    return app
