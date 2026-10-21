"""HTTP and WebSocket endpoints.

POST /sessions {timezone?} -> {session_id, input_sample_rate, frame_ms, output_sample_rate}
GET  /sessions/{id}       -> {session, transcript, metrics, errors}
WS   /ws/voice?session_id -> binary audio + JSON events (see docs/DESIGN.md)
GET  /                    -> the web client
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from .agent import Agent
from .config import Settings
from .pipeline import VoicePipeline
from .providers import LLMProvider, STTProvider, TTSProvider, build_llm, build_stt, build_tts
from .session import SessionManager
from .store import Store
from .tools import ToolExecutor, default_tools, is_valid_timezone

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


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
    settings = settings or Settings.from_env()
    stt = stt or build_stt(settings)
    llm = llm or build_llm(settings)
    tts = tts or build_tts(settings)
    store = store or Store(settings.db_path)
    manager = SessionManager(store, settings)

    def make_agent(caller_timezone: str | None) -> Agent:
        # Per call, so "what time is it?" answers in the caller's time zone.
        tools = ToolExecutor(
            default_tools(caller_timezone or settings.default_timezone), timeout_s=settings.tool_timeout_s
        )
        return Agent(
            llm,
            tools,
            settings.system_prompt,
            timeout_s=settings.llm_timeout_s,
            max_tool_rounds=settings.max_tool_rounds,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        log.info("providers: stt=%s llm=%s tts=%s", stt.name, llm.name, tts.name)
        yield
        for provider in (stt, llm, tts):
            await provider.aclose()
        store.close()

    app = FastAPI(title="Voice Agent MVP", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.sessions = manager
    app.state.providers = {"stt": stt, "llm": llm, "tts": tts}

    @app.post("/sessions", status_code=201)
    async def create_session(request: SessionRequest | None = None) -> dict:
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

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> dict:
        data = await store.get_session(session_id)
        if data is None:
            raise HTTPException(status_code=404, detail="session not found")
        live = manager.get(session_id)
        data["session"]["state"] = live.state.value if live else ("ended" if data["session"]["ended_at"] else "idle")
        return data

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "providers": {"stt": stt.name, "llm": llm.name, "tts": tts.name},
            "active_sessions": manager.active,
        }

    @app.websocket("/ws/voice")
    async def voice(ws: WebSocket, session_id: str = Query("")) -> None:
        await ws.accept()
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
            agent=make_agent(session.timezone),
            tts=tts,
            store=store,
            transport=WebSocketTransport(ws),
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

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
