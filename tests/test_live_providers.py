"""Smoke tests against the real vendor APIs. Skipped unless the keys are set:

DEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=... pytest -m live
"""

from __future__ import annotations

import os

import pytest

from conftest import FIXTURES
from voice_agent.audio import iter_frames, read_wav
from voice_agent.config import Settings
from voice_agent.providers.base import ChatMessage, LLMTurnEnd, TextDelta
from voice_agent.tools import default_tools

pytestmark = pytest.mark.live


def settings() -> Settings:
    return Settings.from_env()


@pytest.mark.skipif(not os.environ.get("DEEPGRAM_API_KEY"), reason="DEEPGRAM_API_KEY not set")
async def test_deepgram_transcribes_the_recording():
    from voice_agent.providers.deepgram import DeepgramSTT

    pcm, rate = read_wav(FIXTURES / "two_turns.wav")
    first_question = pcm[: int(rate * 3.0) * 2]  # the first utterance ends before 3 s
    stt = DeepgramSTT(settings())
    partials: list[str] = []

    async def on_partial(text: str) -> None:
        partials.append(text)

    stream = await stt.open_stream(sample_rate=rate, on_partial=on_partial)
    for frame in iter_frames(first_question, rate, 20):
        await stream.send(frame)
    transcript = await stream.finish()
    await stream.aclose()
    assert "time" in transcript.lower()


@pytest.mark.skipif(not os.environ.get("DEEPGRAM_API_KEY"), reason="DEEPGRAM_API_KEY not set")
async def test_deepgram_speaks():
    from voice_agent.providers.deepgram import DeepgramTTS

    tts = DeepgramTTS(settings())
    audio = b"".join([chunk async for chunk in tts.synthesize("Hello from the voice agent.")])
    await tts.aclose()
    assert len(audio) > tts.sample_rate  # more than half a second of PCM16


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
async def test_claude_calls_the_time_tool():
    from voice_agent.providers.anthropic_llm import AnthropicLLM

    llm = AnthropicLLM(settings())
    specs = [tool.spec for tool in default_tools()]
    events = [
        e
        async for e in llm.stream(
            system=settings().system_prompt,
            messages=[ChatMessage("user", "What time is it in Tokyo right now?")],
            tools=specs,
        )
    ]
    end = events[-1]
    assert isinstance(end, LLMTurnEnd)
    assert any(isinstance(e, TextDelta) for e in events) or end.message.tool_calls
    assert [c.name for c in end.message.tool_calls] == ["get_current_time"]
