"""Provider interfaces. The pipeline only ever talks to these types.

Adding a vendor means writing one class that implements an interface below and
registering it under a name (see ``registry.py``); nothing in the pipeline
changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, Sequence


class ProviderError(Exception):
    """A recoverable failure in one pipeline stage."""

    stage = "provider"

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code


class STTError(ProviderError):
    stage = "stt"


class LLMError(ProviderError):
    stage = "llm"


class TTSError(ProviderError):
    stage = "tts"


# --------------------------------------------------------------------------- STT

PartialCallback = Callable[[str], Awaitable[None]]


class STTStream(ABC):
    """One utterance worth of streaming recognition."""

    @abstractmethod
    async def send(self, pcm: bytes) -> None:
        """Feed PCM16LE mono audio at the sample rate the stream was opened with."""

    @abstractmethod
    async def finish(self) -> str:
        """Signal end of audio and return the final transcript for the utterance."""

    @abstractmethod
    async def aclose(self) -> None:
        """Abort and release resources. Safe to call more than once."""


class STTProvider(ABC):
    name: str = "stt"

    @abstractmethod
    async def open_stream(self, *, sample_rate: int, on_partial: PartialCallback) -> STTStream:
        """Start recognising a new utterance. ``on_partial`` receives interim text."""

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------- LLM


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str  # JSON text
    is_error: bool = False


@dataclass
class ChatMessage:
    """Provider-neutral conversation message.

    ``native`` holds the provider's own representation of an assistant turn (for
    Anthropic: the response content blocks, including thinking blocks). Adapters
    replay it verbatim so the history stays byte-identical between requests.
    """

    role: Literal["user", "assistant"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    native: Any = None
    native_provider: str | None = None


@dataclass
class TextDelta:
    text: str


@dataclass
class LLMTurnEnd:
    """Last event of every ``LLMProvider.stream`` call."""

    message: ChatMessage
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | "refusal" | provider-specific


LLMEvent = TextDelta | LLMTurnEnd


class LLMProvider(ABC):
    name: str = "llm"

    @abstractmethod
    def stream(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> AsyncIterator[LLMEvent]:
        """One model call: yields TextDelta events, then exactly one LLMTurnEnd.

        Closing the iterator early must cancel the request.
        """

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------- TTS


class TTSProvider(ABC):
    name: str = "tts"
    sample_rate: int = 24000

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream PCM16LE mono audio at ``sample_rate`` for one sentence.

        Closing the iterator early must cancel synthesis.
        """

    async def aclose(self) -> None:
        return None
