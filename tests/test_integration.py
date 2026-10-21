"""End to end: a recorded WAV file streamed over a real WebSocket to a real
server, through VAD -> STT -> LLM + tools -> TTS, then read back over REST.

Providers are the offline mocks (the STT script holds what the recording says),
so this runs anywhere; tests/test_live_providers.py repeats the speech stages
against the real vendors when API keys are available.
"""

from __future__ import annotations

import asyncio
import json
import struct
import time

import httpx
from websockets.asyncio.client import connect

from conftest import FIXTURES, SPOKEN
from voice_agent.audio import iter_frames, read_wav


async def test_recorded_call_through_the_full_pipeline(live_server):
    host, app = live_server
    pcm, rate = read_wav(FIXTURES / "two_turns.wav")

    async with httpx.AsyncClient(base_url=f"http://{host}") as http:
        created = await http.post("/sessions", json={"timezone": "Asia/Kolkata"})
        assert created.status_code == 201
        info = created.json()
        assert info["input_sample_rate"] == rate and info["frame_ms"] == 20

        events: list[dict] = []
        audio_turns: list[int] = []
        agent_busy = False

        async with connect(f"ws://{host}/ws/voice?session_id={info['session_id']}") as ws:

            async def receive() -> None:
                nonlocal agent_busy
                async for message in ws:
                    if isinstance(message, bytes):
                        audio_turns.append(struct.unpack("<I", message[:4])[0])
                        continue
                    event = json.loads(message)
                    events.append(event)
                    if event["type"] == "state":
                        agent_busy = event["state"] in ("thinking", "speaking")
                    if event["type"] == "call_ended":
                        return

            receiver = asyncio.create_task(receive())
            quiet = bytes(info["input_sample_rate"] * info["frame_ms"] // 1000 * 2)
            for frame in iter_frames(pcm, rate, info["frame_ms"]):
                await ws.send(frame)
                await asyncio.sleep(0.005)  # about 4x real time
                while agent_busy:  # the caller waits for the reply; the mic keeps streaming
                    await ws.send(quiet)
                    await asyncio.sleep(0.02)

            deadline = time.monotonic() + 10
            while len([e for e in events if e["type"] == "metrics"]) < 2 or agent_busy:
                assert time.monotonic() < deadline, f"no second reply; events: {events}"
                await ws.send(quiet)
                await asyncio.sleep(0.02)
            await ws.send(json.dumps({"type": "end_call"}))
            await asyncio.wait_for(receiver, 5)

        saved = (await http.get(f"/sessions/{info['session_id']}")).json()

    def of(kind: str) -> list[dict]:
        return [e for e in events if e["type"] == kind]

    # Live captions and committed user turns, in order.
    assert of("transcript_partial")
    assert [(e["text"], e["turn_id"]) for e in of("transcript_final")] == [(SPOKEN[0], 1), (SPOKEN[1], 2)]
    # The VAD found exactly the two questions; the click between them never opened an STT stream.
    assert app.state.providers["stt"].opened == 2

    # Tool calls, spoken replies and audio for both turns.
    assert [e["name"] for e in of("tool_call")] == ["get_current_time", "get_order_status"]
    assert of("tool_result")[0]["result"]["timezone"] == "Asia/Kolkata"  # the caller's zone, not the server's
    replies = {turn: " ".join(e["text"] for e in of("agent_text") if e["turn_id"] == turn) for turn in (1, 2)}
    assert replies[1].startswith("Let me check the time. It's ")
    assert "Order A1001 is shipped" in replies[2]
    assert set(audio_turns) == {1, 2}

    states = [e["state"] for e in of("state")]
    assert states[:4] == ["listening", "thinking", "speaking", "listening"]
    assert events[-1] == {"type": "call_ended", "reason": "user_ended"}

    metrics = of("metrics")
    assert [m["turn_id"] for m in metrics] == [1, 2]
    assert all(m["total_ms"] is not None and m["total_ms"] > 0 for m in metrics)

    # Everything is persisted and retrievable by session id.
    assert saved["session"]["end_reason"] == "user_ended"
    assert saved["session"]["state"] == "ended"
    assert saved["session"]["config"]["providers"] == {"stt": "mock", "llm": "mock", "tts": "mock"}
    assert saved["session"]["config"]["caller_timezone"] == "Asia/Kolkata"
    assert [(t["turn_id"], t["role"]) for t in saved["transcript"]] == [
        (1, "user"),
        (1, "agent"),
        (2, "user"),
        (2, "agent"),
    ]
    assert saved["transcript"][1]["tool_calls"][0]["name"] == "get_current_time"
    assert [m["turn_id"] for m in saved["metrics"]] == [1, 2]
    assert saved["errors"] == []


async def test_unknown_session_is_rejected(live_server):
    host, _ = live_server
    async with connect(f"ws://{host}/ws/voice?session_id=nope") as ws:
        error = json.loads(await ws.recv())
        assert error["code"] == "invalid_session"
        await ws.wait_closed()
    assert ws.close_code == 4404

    async with httpx.AsyncClient(base_url=f"http://{host}") as http:
        assert (await http.get("/sessions/nope")).status_code == 404
        health = (await http.get("/healthz")).json()
        assert health["status"] == "ok"
        assert (await http.get("/")).status_code == 200  # the web client is served


async def test_session_time_zone_is_optional_and_validated(live_server):
    host, _ = live_server
    async with httpx.AsyncClient(base_url=f"http://{host}") as http:
        for body, expected in (
            (None, None),
            ({"timezone": "Mars/Olympus"}, None),
            ({"timezone": "Europe/Paris"}, "Europe/Paris"),
        ):
            created = await http.post("/sessions", json=body)
            assert created.status_code == 201
            saved = (await http.get(f"/sessions/{created.json()['session_id']}")).json()
            assert saved["session"]["config"]["caller_timezone"] == expected
