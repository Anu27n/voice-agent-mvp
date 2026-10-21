"""Agent tool loop and the append-only history rules, with a scripted LLM."""

import json

import pytest

from voice_agent.agent import (
    REFUSAL_LINE,
    Agent,
    AgentRoundEnd,
    AgentText,
    AgentToolCall,
    AgentToolResult,
    Conversation,
)
from voice_agent.providers.base import ChatMessage, LLMError, LLMProvider, LLMTurnEnd, TextDelta, ToolCall
from voice_agent.tools import ToolExecutor, default_tools


class ScriptedLLM(LLMProvider):
    """Each call pops the next round: (text, tool_calls, stop_reason)."""

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.requests = []

    async def stream(self, *, system, messages, tools):
        self.requests.append(list(messages))
        text, calls, stop = self.rounds.pop(0)
        for word in text.split():
            yield TextDelta(word + " ")
        native = [{"type": "thinking", "thinking": "", "signature": f"sig{len(self.requests)}"}]
        yield LLMTurnEnd(
            ChatMessage("assistant", text=text, tool_calls=calls, native=native, native_provider="scripted"), stop
        )


def make_agent(llm, max_tool_rounds=3):
    return Agent(llm, ToolExecutor(default_tools()), "system", max_tool_rounds=max_tool_rounds)


async def collect(run):
    return [event async for event in run.events()]


async def test_tool_round_then_answer():
    llm = ScriptedLLM(
        [
            ("Let me check.", [ToolCall("t1", "get_order_status", {"order_id": "A1002"})], "tool_use"),
            ("It ships tomorrow.", [], "end_turn"),
        ]
    )
    conversation = Conversation()
    conversation.add_user("Where is order A1002?")
    run = make_agent(llm).start(conversation)
    events = await collect(run)

    kinds = [type(e) for e in events]
    assert kinds.index(AgentRoundEnd) < kinds.index(AgentToolCall) < kinds.index(AgentToolResult)
    assert (
        "".join(e.text for e in events if isinstance(e, AgentText)).split()
        == "Let me check. It ships tomorrow.".split()
    )
    assert run.tool_log[0]["name"] == "get_order_status" and not run.tool_log[0]["is_error"]

    # The tool round is committed right away; the final answer only once played.
    roles = [(m.role, bool(m.tool_calls), bool(m.tool_results)) for m in conversation.messages]
    assert roles == [("user", False, False), ("assistant", True, False), ("user", False, True)]
    run.commit_completed()
    assert conversation.messages[-1].text == "It ships tomorrow."
    assert conversation.messages[-1].native[0]["signature"] == "sig2"  # replayed verbatim
    # The second request saw the first round's native content and the tool result.
    assert llm.requests[1][1].native[0]["signature"] == "sig1"
    assert json.loads(llm.requests[1][2].tool_results[0].content)["status"] == "processing"


async def test_interrupted_final_round_keeps_only_the_heard_text():
    llm = ScriptedLLM([("Your order shipped on Friday and should arrive Monday.", [], "end_turn")])
    conversation = Conversation()
    conversation.add_user("Where is my order?")
    run = make_agent(llm).start(conversation)
    await collect(run)
    run.commit_interrupted("Your order shipped...")
    assert [(m.role, m.text, m.native) for m in conversation.messages][-1] == (
        "assistant",
        "Your order shipped...",
        None,  # plain text: the generated (never sent back) native turn is dropped
    )


async def test_interrupt_during_tool_execution_closes_the_tool_call():
    llm = ScriptedLLM([("One moment.", [ToolCall("t9", "get_current_time", {})], "tool_use")])
    conversation = Conversation()
    conversation.add_user("What time is it?")
    run = make_agent(llm).start(conversation)
    events = run.events()
    async for event in events:
        if isinstance(event, AgentToolCall):
            break  # cancelled while the tool would run
    await events.aclose()
    run.commit_interrupted("One moment.")
    last = conversation.messages[-1]
    assert last.role == "user" and last.tool_results[0].call_id == "t9" and last.tool_results[0].is_error
    # The tool round was already committed, so its text isn't added a second time.
    assert sum(1 for m in conversation.messages if m.role == "assistant") == 1


async def test_refusal_is_spoken_politely_and_not_replayed():
    llm = ScriptedLLM([("", [ToolCall("t1", "get_current_time", {})], "refusal")])
    conversation = Conversation()
    conversation.add_user("...")
    run = make_agent(llm).start(conversation)
    events = await collect(run)
    assert [e.text for e in events if isinstance(e, AgentText)] == [REFUSAL_LINE]
    assert not any(isinstance(e, AgentToolCall) for e in events)
    run.commit_completed()
    assert conversation.messages[-1].native is None


async def test_runaway_tool_calls_are_capped():
    call = ToolCall("t", "get_current_time", {})
    llm = ScriptedLLM([("", [call], "tool_use")] * 3)
    conversation = Conversation()
    conversation.add_user("loop")
    with pytest.raises(LLMError):
        await collect(make_agent(llm, max_tool_rounds=1).start(conversation))
    # The capped round still got a matching (error) result.
    assert "limit" in conversation.messages[-1].tool_results[0].content


async def test_provider_crash_becomes_llm_error():
    class Crashy(LLMProvider):
        async def stream(self, *, system, messages, tools):
            raise KeyError("oops")
            yield  # pragma: no cover

    conversation = Conversation()
    conversation.add_user("hi")
    with pytest.raises(LLMError) as info:
        await collect(make_agent(Crashy()).start(conversation))
    assert info.value.code == "llm_failed"
