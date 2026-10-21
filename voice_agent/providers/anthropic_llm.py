"""Claude via the Anthropic Python SDK (streaming, with tool use).

Notes on the request shape:
- ``output_config.effort`` is kept low by default: voice replies are short and
  time to first token matters more than deep reasoning.
- Assistant turns are replayed verbatim (``ChatMessage.native``), including
  thinking blocks, and the system prompt and tool list never change during a
  call. The history is therefore append-only, which keeps thinking-block
  signatures valid and lets the prompt cache hit on every turn.
- ``fallbacks="default"`` (beta) lets the API re-run a request that a safety
  classifier declines on Anthropic's recommended fallback model instead of
  returning a refusal. Haiku has no server-side fallback, so it is skipped there.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Sequence

from ..config import ConfigError, Settings
from .base import ChatMessage, LLMError, LLMEvent, LLMProvider, LLMTurnEnd, TextDelta, ToolCall, ToolSpec
from .registry import register_llm

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
_THINKING_TYPES = {"thinking", "redacted_thinking"}


@register_llm("anthropic")
class AnthropicLLM(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings, client: Any = None) -> None:
        import anthropic

        self._anthropic = anthropic
        if client is None:
            if not settings.anthropic_api_key:
                raise ConfigError("LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY")
            client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                max_retries=1,
                timeout=anthropic.Timeout(30.0, connect=5.0),
            )
        self._client = client
        self.model = settings.llm_model
        self.effort = settings.llm_effort
        self.max_tokens = settings.llm_max_tokens
        self.fallbacks = settings.llm_fallbacks if "haiku" not in settings.llm_model else "off"

    def request_kwargs(
        self, *, system: str, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": to_anthropic_messages(messages),
            "output_config": {"effort": self.effort},
            # Caches the stable prefix (system + tools + earlier turns) automatically.
            "cache_control": {"type": "ephemeral"},
        }
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                    # Tool inputs stream as generated; ToolExecutor validates them.
                    "eager_input_streaming": True,
                }
                for t in tools
            ]
        if self.fallbacks == "default":
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    async def stream(
        self, *, system: str, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        anthropic = self._anthropic
        for attempt in range(2):
            kwargs = self.request_kwargs(system=system, messages=messages, tools=tools)
            try:
                async with self._client.beta.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        if event.type == "text" and event.text:
                            yield TextDelta(event.text)
                    final = await stream.get_final_message()
                break
            except anthropic.BadRequestError as exc:
                # Recovery for a history whose thinking blocks no longer validate:
                # strip them once and retry. Rejected before any output, so nothing
                # has been yielded for this attempt.
                if attempt == 0 and "signature" in str(exc) and strip_thinking(messages):
                    log.warning("thinking blocks rejected; retrying without them")
                    continue
                raise LLMError("llm_bad_request", _describe(exc)) from exc
            except ValueError as exc:
                # Tool-input JSON the SDK could not parse at all (eager streaming).
                raise LLMError("llm_invalid_tool_input", str(exc)) from exc
            except anthropic.RateLimitError as exc:
                raise LLMError("llm_rate_limited", _describe(exc)) from exc
            except anthropic.APITimeoutError as exc:
                raise LLMError("llm_timeout", "request to Anthropic timed out") from exc
            except anthropic.APIConnectionError as exc:
                raise LLMError("llm_connection", "could not reach Anthropic") from exc
            except anthropic.APIStatusError as exc:
                raise LLMError(f"llm_http_{exc.status_code}", _describe(exc)) from exc

        yield to_turn_end(final)


def to_turn_end(final: Any) -> LLMTurnEnd:
    """Convert a final SDK message into the neutral LLMTurnEnd."""
    stop_reason = final.stop_reason or "end_turn"
    text = "".join(block.text for block in final.content if block.type == "text")
    if stop_reason == "refusal":
        # A refusal can cut a tool call off mid-input: never run or replay it.
        return LLMTurnEnd(ChatMessage(role="assistant", text=text), stop_reason)
    tool_calls = [
        ToolCall(block.id, block.name, block.input if isinstance(block.input, dict) else {})
        for block in final.content
        if block.type == "tool_use"
    ]
    if tool_calls and stop_reason == "max_tokens":
        raise LLMError("llm_truncated", "the model ran out of tokens in the middle of a tool call")
    native = [block.to_dict(mode="json") for block in final.content]
    return LLMTurnEnd(
        ChatMessage(role="assistant", text=text, tool_calls=tool_calls, native=native, native_provider="anthropic"),
        stop_reason,
    )


def to_anthropic_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "user":
            if msg.tool_results:
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r.call_id,
                                "content": r.content,
                                "is_error": r.is_error,
                            }
                            for r in msg.tool_results
                        ],
                    }
                )
            elif msg.text.strip():
                out.append({"role": "user", "content": msg.text})
            continue

        if msg.native is not None and msg.native_provider == "anthropic":
            out.append({"role": "assistant", "content": msg.native})
            continue
        content: list[dict[str, Any]] = []
        if msg.text.strip():
            content.append({"type": "text", "text": msg.text})
        for call in msg.tool_calls:
            content.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
        if content:
            out.append({"role": "assistant", "content": content})
    return out


def strip_thinking(messages: Sequence[ChatMessage]) -> bool:
    """Remove thinking blocks from stored assistant turns. Returns True if any were removed."""
    removed = False
    for msg in messages:
        if msg.native_provider == "anthropic" and isinstance(msg.native, list):
            kept = [b for b in msg.native if b.get("type") not in _THINKING_TYPES]
            if len(kept) != len(msg.native):
                msg.native = kept
                removed = True
    return removed


def _describe(exc: Exception) -> str:
    message = getattr(exc, "message", None) or str(exc)
    request_id = getattr(exc, "request_id", None)
    return f"{message} (request_id={request_id})" if request_id else message
