"""AnthropicLLM against a fake SDK client: request shape, event translation,
verbatim replay of earlier turns, and error handling."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from anthropic.types.beta import BetaTextBlock, BetaThinkingBlock, BetaToolUseBlock

from conftest import fast_settings
from voice_agent.providers.anthropic_llm import FALLBACK_BETA, AnthropicLLM, to_anthropic_messages
from voice_agent.providers.base import ChatMessage, LLMError, LLMTurnEnd, TextDelta, ToolResult, ToolSpec

TOOLS = [ToolSpec("get_current_time", "Get the time.", {"type": "object", "properties": {}})]


class FakeStream:
    def __init__(self, texts, final):
        self._events = [SimpleNamespace(type="text", text=t) for t in texts]
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._final


class FakeClient:
    """Mimics ``client.beta.messages.stream(**kwargs)``; each call pops a response."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def message(content, stop_reason="end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def api_error(cls, status, text):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(text, response=httpx2.Response(status, request=request), body=None)


def llm(client, **overrides):
    return AnthropicLLM(fast_settings(llm_provider="anthropic", **overrides), client=client)


async def run(provider, messages):
    return [e async for e in provider.stream(system="Be brief.", messages=messages, tools=TOOLS)]


async def test_request_shape():
    client = FakeClient(FakeStream(["Hi"], message([BetaTextBlock(type="text", text="Hi")])))
    await run(llm(client), [ChatMessage("user", "Hello")])
    kwargs = client.calls[0]
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["system"] == "Be brief."
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert kwargs["betas"] == [FALLBACK_BETA] and kwargs["fallbacks"] == "default"
    assert kwargs["tools"][0]["name"] == "get_current_time"
    assert kwargs["tools"][0]["eager_input_streaming"] is True
    assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
    assert "thinking" not in kwargs  # adaptive thinking is the model default


async def test_fallbacks_can_be_disabled_and_are_skipped_for_haiku():
    for overrides in ({"llm_fallbacks": "off"}, {"llm_model": "claude-haiku-5-5"}):
        client = FakeClient(FakeStream([], message([BetaTextBlock(type="text", text="ok")])))
        await run(llm(client, **overrides), [ChatMessage("user", "Hello")])
        assert "fallbacks" not in client.calls[0] and "betas" not in client.calls[0]


async def test_text_and_tool_use_are_translated():
    final = message(
        [
            BetaThinkingBlock(type="thinking", thinking="", signature="sig-1"),
            BetaTextBlock(type="text", text="Let me check."),
            BetaToolUseBlock(type="tool_use", id="toolu_1", name="get_current_time", input={}),
        ],
        stop_reason="tool_use",
    )
    events = await run(llm(FakeClient(FakeStream(["Let me ", "check."], final))), [ChatMessage("user", "Time?")])
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Let me ", "check."]
    end = events[-1]
    assert isinstance(end, LLMTurnEnd) and end.stop_reason == "tool_use"
    assert end.message.text == "Let me check."
    assert [(c.id, c.name, c.arguments) for c in end.message.tool_calls] == [("toolu_1", "get_current_time", {})]
    assert end.message.native[0] == {"type": "thinking", "thinking": "", "signature": "sig-1"}
    assert end.message.native_provider == "anthropic"


def test_history_is_replayed_verbatim():
    native = [
        {"type": "thinking", "thinking": "", "signature": "sig-1"},
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_current_time", "input": {}},
    ]
    messages = [
        ChatMessage("user", "Time?"),
        ChatMessage("assistant", "Let me check.", native=native, native_provider="anthropic"),
        ChatMessage("user", tool_results=[ToolResult("toolu_1", "get_current_time", '{"spoken": "3 PM"}')]),
        ChatMessage("assistant", "It's 3..."),  # interrupted turn: plain text
        ChatMessage("assistant", ""),  # nothing heard: skipped
        ChatMessage("user", "Thanks"),
    ]
    assert to_anthropic_messages(messages) == [
        {"role": "user", "content": "Time?"},
        {"role": "assistant", "content": native},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"spoken": "3 PM"}', "is_error": False}
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "It's 3..."}]},
        {"role": "user", "content": "Thanks"},
    ]


async def test_refusal_is_never_replayed_or_executed():
    final = message(
        [BetaToolUseBlock(type="tool_use", id="toolu_x", name="get_current_time", input={})], stop_reason="refusal"
    )
    events = await run(llm(FakeClient(FakeStream([], final))), [ChatMessage("user", "...")])
    end = events[-1]
    assert end.stop_reason == "refusal"
    assert end.message.tool_calls == [] and end.message.native is None


async def test_truncated_tool_call_is_an_error():
    final = message(
        [BetaToolUseBlock(type="tool_use", id="toolu_x", name="get_current_time", input={})], stop_reason="max_tokens"
    )
    with pytest.raises(LLMError) as info:
        await run(llm(FakeClient(FakeStream([], final))), [ChatMessage("user", "...")])
    assert info.value.code == "llm_truncated"


async def test_rejected_thinking_signatures_are_stripped_and_retried_once():
    history = [
        ChatMessage("user", "Hi"),
        ChatMessage(
            "assistant",
            "Hello!",
            native=[{"type": "thinking", "thinking": "", "signature": "stale"}, {"type": "text", "text": "Hello!"}],
            native_provider="anthropic",
        ),
        ChatMessage("user", "Again"),
    ]
    rejected = api_error(
        anthropic.BadRequestError, 400, "messages.1.content.0: Invalid `signature` in `thinking` block."
    )
    client = FakeClient(rejected, FakeStream(["Hi again"], message([BetaTextBlock(type="text", text="Hi again")])))
    events = await run(llm(client), history)
    assert events[-1].message.text == "Hi again"
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][1]["content"] == [{"type": "text", "text": "Hello!"}]


@pytest.mark.parametrize(
    "error, code",
    [
        (api_error(anthropic.RateLimitError, 429, "slow down"), "llm_rate_limited"),
        (api_error(anthropic.InternalServerError, 500, "oops"), "llm_http_500"),
        (api_error(anthropic.BadRequestError, 400, "bad"), "llm_bad_request"),
    ],
)
async def test_api_errors_become_llm_errors(error, code):
    with pytest.raises(LLMError) as info:
        await run(llm(FakeClient(error)), [ChatMessage("user", "Hi")])
    assert info.value.code == code


def test_missing_api_key_is_a_config_error():
    from voice_agent.config import ConfigError

    with pytest.raises(ConfigError):
        AnthropicLLM(fast_settings(llm_provider="anthropic", anthropic_api_key=None))
