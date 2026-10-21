"""Offline providers for local development and tests.

- MockSTT returns transcripts from a script (``MOCK_STT_SCRIPT``, ``|``-separated),
  one per finished utterance, with word-by-word partials as audio arrives.
  Special entries: ``!error`` fails, ``!timeout`` hangs, ``!empty`` returns "".
- MockLLM is a small rule-based agent that exercises the same paths as a real
  model: streamed text, tool calls (time and order questions), tool results,
  and memory of earlier turns. A user turn containing ``!llm_error`` fails.
- MockTTS renders a soft hum whose length follows the text length.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import AsyncIterator, Sequence

import numpy as np

from ..audio import pcm16_duration_ms, to_pcm16
from ..config import Settings
from .base import (
    ChatMessage,
    LLMError,
    LLMEvent,
    LLMProvider,
    LLMTurnEnd,
    PartialCallback,
    STTError,
    STTProvider,
    STTStream,
    TextDelta,
    ToolCall,
    ToolSpec,
    TTSProvider,
)
from .registry import register_llm, register_stt, register_tts

DEFAULT_STT_SCRIPT = (
    "Hi there, my name is Sam.",
    "What time is it right now?",
    "Can you check the status of order A1001?",
    "What about order B2002?",
    "Do you remember my name?",
    "Thanks, that's all for today.",
)
_SPECIAL = {"!error", "!timeout", "!empty"}


@register_stt("mock")
class MockSTT(STTProvider):
    name = "mock"

    def __init__(self, settings: Settings) -> None:
        self.script = settings.mock_stt_script or DEFAULT_STT_SCRIPT
        self.latency_s = settings.mock_stt_latency_ms / 1000.0
        self.opened = 0
        self.transcripts: list[str] = []
        self._index = 0

    def peek(self) -> str:
        return self.script[self._index % len(self.script)]

    def take(self) -> str:
        text = self.peek()
        self._index += 1
        return text

    async def open_stream(self, *, sample_rate: int, on_partial: PartialCallback) -> STTStream:
        self.opened += 1
        return _MockSTTStream(self, sample_rate, on_partial)


class _MockSTTStream(STTStream):
    def __init__(self, provider: MockSTT, sample_rate: int, on_partial: PartialCallback) -> None:
        self._provider = provider
        self._sample_rate = sample_rate
        self._on_partial = on_partial
        upcoming = provider.peek()
        self._words = [] if upcoming in _SPECIAL else upcoming.split()
        self._bytes = 0
        self._shown = 0

    async def send(self, pcm: bytes) -> None:
        self._bytes += len(pcm)
        words = min(len(self._words), int(pcm16_duration_ms(self._bytes, self._sample_rate) // 250))
        if words > self._shown:
            self._shown = words
            await self._on_partial(" ".join(self._words[:words]))

    async def finish(self) -> str:
        await asyncio.sleep(self._provider.latency_s)
        text = self._provider.take()
        if text == "!error":
            raise STTError("stt_failed", "mock STT failure")
        if text == "!timeout":
            await asyncio.sleep(3600)
        if text == "!empty":
            return ""
        self._provider.transcripts.append(text)
        return text

    async def aclose(self) -> None:
        return None


@register_llm("mock")
class MockLLM(LLMProvider):
    name = "mock"

    def __init__(self, settings: Settings) -> None:
        self.latency_s = settings.mock_llm_latency_ms / 1000.0
        self.requests: list[list[ChatMessage]] = []

    async def stream(
        self, *, system: str, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        self.requests.append(list(messages))
        await asyncio.sleep(self.latency_s)
        reply, calls = self._decide(messages, {t.name for t in tools})
        for piece in re.findall(r"\S+\s*", reply):
            yield TextDelta(piece)
            await asyncio.sleep(0.005)
        yield LLMTurnEnd(
            ChatMessage(role="assistant", text=reply, tool_calls=calls),
            "tool_use" if calls else "end_turn",
        )

    def _decide(self, messages: Sequence[ChatMessage], tools: set[str]) -> tuple[str, list[ToolCall]]:
        last = messages[-1]
        if last.role == "user" and last.tool_results:
            return " ".join(_describe_result(r.name, json.loads(r.content), r.is_error) for r in last.tool_results), []

        said = [m.text for m in messages if m.role == "user" and m.text]
        text = said[-1] if said else ""
        lower = text.lower()
        if "!llm_error" in lower:
            raise LLMError("llm_failed", "mock LLM failure")
        if "time" in lower and "get_current_time" in tools:
            return "Let me check the time.", [_call("get_current_time", {})]
        if "order" in lower and "get_order_status" in tools:
            match = re.search(r"\b([a-z])[\s-]?(\d{3,})\b", lower)
            if match:
                order_id = (match.group(1) + match.group(2)).upper()
                return "Let me look that up.", [_call("get_order_status", {"order_id": order_id})]
            return "Sure, what's the order number?", []
        if "name" in lower:
            for earlier in said[:-1]:
                found = re.search(r"my name is (\w+)", earlier, re.IGNORECASE)
                if found:
                    return f"Of course, your name is {found.group(1)}.", []
        turn = len(said)
        earlier = f" Earlier you said: {said[0]}" if turn > 1 else ""
        return f"You said: {text.rstrip('.!?')}. That's turn {turn} of our call.{earlier}", []


def _call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(f"mock_{uuid.uuid4().hex[:8]}", name, arguments)


def _describe_result(name: str, data: dict, is_error: bool) -> str:
    if is_error:
        return f"Sorry, I couldn't do that. {data.get('error', '')}".strip()
    if name == "get_current_time":
        return f"It's {data['spoken']}."
    if name == "get_order_status":
        details = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in data.items() if k not in {"order_id", "status"})
        return f"Order {data['order_id']} is {data['status']}" + (f", {details}." if details else ".")
    return f"The {name} tool says {json.dumps(data)}."


@register_tts("mock")
class MockTTS(TTSProvider):
    name = "mock"

    def __init__(self, settings: Settings) -> None:
        self.sample_rate = settings.tts_sample_rate
        self.latency_s = settings.mock_tts_latency_ms / 1000.0
        self.ms_per_char = settings.mock_tts_ms_per_char
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        await asyncio.sleep(self.latency_s)
        self.spoken.append(text)
        pcm = self.render(text)
        chunk = self.sample_rate // 10 * 2  # 100 ms
        for start in range(0, len(pcm), chunk):
            yield pcm[start : start + chunk]
            await asyncio.sleep(0)

    def render(self, text: str) -> bytes:
        duration_s = max(0.05, len(text) * self.ms_per_char / 1000.0)
        sr = self.sample_rate
        t = np.arange(int(duration_s * sr)) / sr
        pitch = 170.0 + 25.0 * np.sin(2 * np.pi * 0.7 * t)
        phase = 2 * np.pi * np.cumsum(pitch) / sr
        voice = 0.6 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
        syllables = 0.5 * (1 - np.cos(2 * np.pi * 4.0 * t))
        fade = np.minimum(1.0, np.minimum(t, duration_s - t) / 0.02)
        return to_pcm16(0.12 * voice * syllables * np.clip(fade, 0.0, 1.0))
