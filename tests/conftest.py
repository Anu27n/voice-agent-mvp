from __future__ import annotations

import asyncio
import json
import socket
import struct
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_agent.agent import Agent  # noqa: E402
from voice_agent.audio import to_pcm16  # noqa: E402
from voice_agent.config import Settings  # noqa: E402
from voice_agent.main import create_app  # noqa: E402
from voice_agent.pipeline import VoicePipeline  # noqa: E402
from voice_agent.providers.mock import MockLLM, MockSTT, MockTTS  # noqa: E402
from voice_agent.session import Session  # noqa: E402
from voice_agent.store import Store, utcnow  # noqa: E402
from voice_agent.tools import ToolExecutor, default_tools  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_RATE = 16000
FRAME_MS = 20
# What tests/fixtures/two_turns.wav says; the mock STT "hears" these in order.
SPOKEN = ("What time is it right now?", "Can you check the status of order A1001?")


def fast_settings(**overrides: Any) -> Settings:
    """Settings tuned so tests run in well under a second per call."""
    base = Settings(
        db_path=":memory:",
        mock_stt_latency_ms=10,
        mock_llm_latency_ms=10,
        mock_tts_latency_ms=5,
        mock_tts_ms_per_char=4.0,
        stt_timeout_s=1.0,
        llm_timeout_s=1.0,
        tts_timeout_s=1.0,
    )
    return replace(base, **overrides)


def speech(ms: int, amplitude: float = 0.3) -> list[bytes]:
    """Voiced frames: a 180 Hz tone with harmonics at a speech-like level."""
    n = SAMPLE_RATE * ms // 1000
    t = np.arange(n) / SAMPLE_RATE
    wave = amplitude * (0.6 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 360 * t))
    return _frames(to_pcm16(wave))


def silence(ms: int) -> list[bytes]:
    rng = np.random.default_rng(ms)
    return _frames(to_pcm16(rng.normal(0.0, 0.001, SAMPLE_RATE * ms // 1000)))


def _frames(pcm: bytes) -> list[bytes]:
    size = SAMPLE_RATE * FRAME_MS // 1000 * 2
    return [pcm[i : i + size] for i in range(0, len(pcm) - size + 1, size)]


class FakeTransport:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.audio: list[tuple[int, bytes]] = []
        self.closed = False

    async def send_text(self, data: str) -> None:
        self.events.append(json.loads(data))

    async def send_bytes(self, data: bytes) -> None:
        (turn_id,) = struct.unpack("<I", data[:4])
        self.audio.append((turn_id, data[4:]))

    async def close(self, code: int = 1000) -> None:
        self.closed = True

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == kind]

    def states(self) -> list[str]:
        return [e["state"] for e in self.of("state")]


async def wait_for(predicate: Callable[[], bool], timeout: float = 3.0, message: str = "condition") -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {message}")
        await asyncio.sleep(0.005)


class Harness:
    """A VoicePipeline wired to mock providers, an in-memory DB and a fake socket."""

    def __init__(self, settings: Settings, *, stt=None, llm=None, tts=None) -> None:
        self.settings = settings
        self.stt = stt or MockSTT(settings)
        self.llm = llm or MockLLM(settings)
        self.tts = tts or MockTTS(settings)
        self.store = Store(":memory:")
        self.session = Session(id="test-session", started_at=utcnow(), created_monotonic=0.0)
        self.transport = FakeTransport()
        tools = ToolExecutor(default_tools(on_end_call=self._hang_up), timeout_s=settings.tool_timeout_s)
        self.agent = Agent(self.llm, tools, settings.system_prompt, timeout_s=settings.llm_timeout_s)
        self.pipeline = VoicePipeline(
            self.session,
            settings,
            stt=self.stt,
            agent=self.agent,
            tts=self.tts,
            store=self.store,
            transport=self.transport,
        )

    def _hang_up(self) -> None:
        self.session.hangup_requested = True

    async def start(self) -> "Harness":
        await self.store.create_session(self.session.id, self.session.started_at, self.settings.public_config())
        await self.pipeline.start()
        return self

    async def feed(self, frames: list[bytes]) -> None:
        for frame in frames:
            await self.pipeline.on_audio(frame)

    async def say(self, ms: int = 600) -> None:
        """One utterance: speech followed by enough silence to end it."""
        await self.feed(speech(ms) + silence(self.settings.vad_silence_ms + 100))

    async def wait_listening_after(self, turns: int, timeout: float = 3.0) -> None:
        await wait_for(
            lambda: len(self.transport.of("metrics")) >= turns and self.session.state.value == "listening",
            timeout,
            f"{turns} completed turn(s)",
        )

    async def flush(self) -> None:
        """Let the outbox deliver everything queued so far."""
        await asyncio.sleep(0.02)

    async def stop(self) -> None:
        await self.pipeline.on_disconnect()
        self.store.close()


@pytest.fixture
async def harness_factory():
    created: list[Harness] = []

    async def make(settings: Settings | None = None, **providers: Any) -> Harness:
        harness = Harness(settings or fast_settings(), **providers)
        created.append(harness)
        return await harness.start()

    yield make
    for harness in created:
        if not harness.pipeline.ended.is_set():
            await harness.stop()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def live_server(tmp_path):
    """The real app on a real port (mock providers), served from a background thread.
    Yields ("127.0.0.1:<port>", app)."""
    settings = fast_settings(db_path=str(tmp_path / "calls.db"), mock_stt_script=SPOKEN, mock_tts_ms_per_char=8.0)
    app = create_app(settings)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.02)
    yield f"127.0.0.1:{port}", app
    server.should_exit = True
    thread.join(timeout=10)
