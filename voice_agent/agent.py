"""The agent: one reply = one or more LLM rounds, with tool calls in between.

The conversation history is append-only. A round that ends in tool calls is
committed immediately (the next request needs it). The final round of a reply is
held back until the pipeline knows how much of it the caller actually heard:
after playback finishes it is committed as generated; after a barge-in only the
spoken part is committed, as plain text. Either way no message that was already
sent to the provider is ever edited, which keeps provider-side state such as
Claude's thinking-block signatures and the prompt cache valid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterator, Union

from .providers.base import (
    ChatMessage,
    LLMError,
    LLMProvider,
    LLMTurnEnd,
    ProviderError,
    TextDelta,
    ToolCall,
    ToolResult,
)
from .tools import ToolExecutor
from .util import aiter_timeout

REFUSAL_LINE = "Sorry, I can't help with that one. Is there anything else I can do?"
INTERRUPTED_TOOL_RESULT = json.dumps({"error": "Cancelled: the caller interrupted before this finished."})
TOOL_LIMIT_RESULT = json.dumps({"error": "Tool call limit reached for this reply. Answer the caller directly."})


class Conversation:
    """Message history for one call (provider-neutral, append-only)."""

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def add(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def add_user(self, text: str) -> None:
        self.add(ChatMessage(role="user", text=text))


@dataclass
class AgentText:
    text: str


@dataclass
class AgentRoundEnd:
    """The model finished a round that calls tools; flush any partial sentence."""


@dataclass
class AgentToolCall:
    call: ToolCall


@dataclass
class AgentToolResult:
    result: ToolResult


AgentEvent = Union[AgentText, AgentRoundEnd, AgentToolCall, AgentToolResult]


class Agent:
    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolExecutor,
        system_prompt: str,
        *,
        timeout_s: float = 10.0,
        max_tool_rounds: int = 3,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.timeout_s = timeout_s
        self.max_tool_rounds = max_tool_rounds

    def start(self, conversation: Conversation) -> "AgentRun":
        return AgentRun(self, conversation)


class AgentRun:
    """One reply in progress."""

    def __init__(self, agent: Agent, conversation: Conversation) -> None:
        self._agent = agent
        self._conversation = conversation
        self.round_index = 0
        self.round_committed = False  # has the current round's message been added to history?
        self.text = ""
        self.tool_log: list[dict] = []
        self._pending_calls: list[ToolCall] = []  # committed tool_use awaiting results
        self._final: ChatMessage | None = None

    async def events(self) -> AsyncIterator[AgentEvent]:
        agent = self._agent
        for round_index in range(agent.max_tool_rounds + 1):
            self.round_index = round_index
            self.round_committed = False
            end = None
            stream = agent.llm.stream(
                system=agent.system_prompt,
                messages=list(self._conversation.messages),
                tools=agent.tools.specs,
            )
            try:
                async for event in aiter_timeout(stream, agent.timeout_s, agent.timeout_s):
                    if isinstance(event, TextDelta):
                        if event.text:
                            self.text += event.text
                            yield AgentText(event.text)
                    elif isinstance(event, LLMTurnEnd):
                        end = event
            except TimeoutError as exc:
                raise LLMError("llm_timeout", "The language model took too long to respond.") from exc
            except ProviderError:
                raise
            except Exception as exc:
                raise LLMError("llm_failed", f"{type(exc).__name__}: {exc}") from exc
            if end is None:
                raise LLMError("llm_failed", "The model stream ended without a final message.")

            message = end.message
            if end.stop_reason == "refusal":
                # Never replay a refused turn's content (it may hold a cut-off tool call).
                if not message.text.strip():
                    self.text += REFUSAL_LINE
                    yield AgentText(REFUSAL_LINE)
                self._final = ChatMessage(role="assistant", text=message.text or REFUSAL_LINE)
                return
            if not message.tool_calls:
                self._final = message
                return

            self._conversation.add(message)
            self.round_committed = True
            self._pending_calls = list(message.tool_calls)
            yield AgentRoundEnd()
            for call in message.tool_calls:
                yield AgentToolCall(call)
            if round_index < agent.max_tool_rounds:
                results = await agent.tools.run_all(message.tool_calls)
            else:
                results = [ToolResult(c.id, c.name, TOOL_LIMIT_RESULT, is_error=True) for c in message.tool_calls]
            for call, result in zip(message.tool_calls, results, strict=True):
                self.tool_log.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "result": result.content,
                        "is_error": result.is_error,
                    }
                )
                yield AgentToolResult(result)
            self._conversation.add(ChatMessage(role="user", tool_results=results))
            self._pending_calls = []
        raise LLMError("llm_failed", "The model kept calling tools without answering.")

    def commit_completed(self) -> None:
        """The whole reply was played: keep the final round exactly as generated."""
        if self._final is not None and self._final.text.strip():
            self._conversation.add(self._final)
        self._final = None

    def commit_interrupted(self, spoken_text: str) -> None:
        """Barge-in: keep only what the caller heard of the unfinished round."""
        if self._pending_calls:
            # Every tool_use needs a matching result before the next request.
            self._conversation.add(
                ChatMessage(
                    role="user",
                    tool_results=[
                        ToolResult(c.id, c.name, INTERRUPTED_TOOL_RESULT, is_error=True) for c in self._pending_calls
                    ],
                )
            )
            self._pending_calls = []
        if not self.round_committed and spoken_text.strip():
            self._conversation.add(ChatMessage(role="assistant", text=spoken_text.strip()))
        self._final = None
