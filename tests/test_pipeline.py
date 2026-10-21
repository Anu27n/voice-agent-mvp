"""Pipeline behaviour with mock providers: the call flow, barge-in, every failure
case from the spec, and the call-ending rules."""

from __future__ import annotations

import asyncio
import json
import re

from conftest import fast_settings, silence, speech, wait_for
from voice_agent.pipeline import FALLBACK_LINE, MAX_DURATION_GOODBYE, SILENCE_GOODBYE
from voice_agent.providers.base import ChatMessage, LLMProvider, LLMTurnEnd, TextDelta, TTSError, TTSProvider


class MarkdownLLM(LLMProvider):
    """A model that ignores the 'plain spoken language' instruction."""

    TEXT = "**Sure!** Here are your options:\n- Express shipping 🚚\n- Standard shipping\n"

    async def stream(self, *, system, messages, tools):
        for piece in re.findall(r"\S+\s*", self.TEXT):
            yield TextDelta(piece)
        yield LLMTurnEnd(ChatMessage("assistant", text=self.TEXT), "end_turn")


async def test_markdown_is_cleaned_before_it_is_spoken(harness_factory):
    h = await harness_factory(llm=MarkdownLLM())
    await h.say()
    await h.wait_listening_after(1)
    await h.flush()
    expected = ["Sure!", "Here are your options:", "Express shipping", "Standard shipping"]
    assert [e["text"] for e in h.transport.of("agent_text")] == expected
    assert h.tts.spoken == expected


async def test_single_turn_happy_path(harness_factory):
    h = await harness_factory()
    await h.say()
    await h.wait_listening_after(1)
    await h.flush()

    assert h.transport.states()[:4] == ["listening", "thinking", "speaking", "listening"]
    assert h.transport.of("transcript_partial"), "live captions expected"
    final = h.transport.of("transcript_final")
    assert final == [{"type": "transcript_final", "text": "Hi there, my name is Sam.", "turn_id": 1}]
    agent_text = " ".join(e["text"] for e in h.transport.of("agent_text"))
    assert agent_text.startswith("You said: Hi there, my name is Sam")
    assert h.transport.audio and all(turn == 1 for turn, _ in h.transport.audio)

    metrics = h.transport.of("metrics")[0]
    stages = [metrics[k] for k in ("vad_end_ms", "stt_final_ms", "llm_first_token_ms", "tts_first_audio_ms")]
    assert all(v is not None and v >= 0 for v in stages)
    assert abs(sum(stages) - metrics["total_ms"]) < 1.0

    saved = await h.store.get_session(h.session.id)
    assert [(t["turn_id"], t["role"]) for t in saved["transcript"]] == [(1, "user"), (1, "agent")]
    assert saved["metrics"][0]["total_ms"] == metrics["total_ms"]


async def test_five_turn_conversation_remembers_and_uses_tools(harness_factory):
    h = await harness_factory()
    for turn in range(1, 6):
        await h.say()
        await h.wait_listening_after(turn)
    await h.flush()

    by_turn: dict[int, str] = {}
    for event in h.transport.of("agent_text"):
        by_turn[event["turn_id"]] = (by_turn.get(event["turn_id"], "") + " " + event["text"]).strip()
    assert by_turn[2].startswith("Let me check the time. It's ")
    assert "Order A1001 is shipped" in by_turn[3]
    assert "No order found" in by_turn[4]  # failed tool call, handled gracefully
    assert by_turn[5] == "Of course, your name is Sam."  # remembers turn 1

    tool_calls = [e["name"] for e in h.transport.of("tool_call")]
    assert tool_calls == ["get_current_time", "get_order_status", "get_order_status"]
    results = h.transport.of("tool_result")
    assert [r["is_error"] for r in results] == [False, False, True]

    # The last request carried the whole history, tool rounds included.
    last_request = h.llm.requests[-1]
    assert [m.role for m in last_request].count("user") >= 5
    assert any(m.tool_results for m in last_request)


async def test_noise_and_short_blips_never_reach_the_llm(harness_factory):
    h = await harness_factory()
    await h.feed(speech(60) + silence(800))  # click: too short to start an utterance
    await h.feed(speech(180) + silence(800))  # starts an utterance, but too short to keep
    await asyncio.sleep(0.1)

    assert h.stt.opened == 1
    assert h.stt.transcripts == []
    assert h.llm.requests == []
    assert not h.transport.of("transcript_final")
    assert h.session.state.value == "listening"


async def test_empty_transcript_is_ignored(harness_factory):
    h = await harness_factory(fast_settings(mock_stt_script=("!empty",)))
    await h.say()
    await asyncio.sleep(0.15)
    assert h.llm.requests == []
    assert not h.transport.of("transcript_final")
    assert h.session.state.value == "listening"


async def test_stt_error_says_fallback_and_call_continues(harness_factory):
    h = await harness_factory(fast_settings(mock_stt_script=("!error", "Hello again.")))
    await h.say()
    await h.wait_listening_after(1)
    await h.flush()
    assert [e["text"] for e in h.transport.of("agent_text")] == [FALLBACK_LINE]
    assert h.transport.of("error")[0]["code"] == "stt_failed"
    assert h.llm.requests == []

    await h.say()
    await h.wait_listening_after(2)
    assert h.transport.of("transcript_final")[0]["text"] == "Hello again."
    saved = await h.store.get_session(h.session.id)
    assert [(e["stage"], e["code"]) for e in saved["errors"]] == [("stt", "stt_failed")]


async def test_stt_timeout_says_fallback(harness_factory):
    h = await harness_factory(fast_settings(mock_stt_script=("!timeout",), stt_timeout_s=0.2))
    await h.say()
    await h.wait_listening_after(1)
    await h.flush()
    assert h.transport.of("error")[0]["code"] == "stt_timeout"
    assert [e["text"] for e in h.transport.of("agent_text")] == [FALLBACK_LINE]


async def test_llm_error_says_fallback_and_call_continues(harness_factory):
    h = await harness_factory(fast_settings(mock_stt_script=("Please !llm_error now.", "Hello.")))
    await h.say()
    await h.wait_listening_after(1)
    await h.flush()
    assert [e["text"] for e in h.transport.of("agent_text")] == [FALLBACK_LINE]
    assert h.transport.of("error")[0]["code"] == "llm_failed"

    await h.say()
    await h.wait_listening_after(2)
    assert h.transport.of("agent_text")[-1]["turn_id"] == 2


class HangingLLM(LLMProvider):
    async def stream(self, *, system, messages, tools):
        await asyncio.sleep(3600)
        yield  # pragma: no cover


async def test_llm_timeout_says_fallback(harness_factory):
    h = await harness_factory(fast_settings(llm_timeout_s=0.2), llm=HangingLLM())
    await h.say()
    await h.wait_listening_after(1)
    await h.flush()
    assert h.transport.of("error")[0]["code"] == "llm_timeout"
    assert [e["text"] for e in h.transport.of("agent_text")] == [FALLBACK_LINE]


class BrokenTTS(TTSProvider):
    sample_rate = 24000

    async def synthesize(self, text):
        raise TTSError("tts_http_500", "synthesis service unavailable")
        yield b""  # pragma: no cover


async def test_tts_error_is_reported_and_call_continues(harness_factory):
    h = await harness_factory(tts=BrokenTTS())
    await h.say()
    await h.wait_listening_after(1)
    await h.flush()
    codes = [e["code"] for e in h.transport.of("error")]
    assert codes[0] == "tts_http_500"
    assert h.transport.of("agent_text")[-1]["text"] == FALLBACK_LINE  # caption still shown
    assert h.transport.audio == []

    await h.say()
    await h.wait_listening_after(2)  # the call carries on


async def test_client_barge_in_stops_reply_and_keeps_only_heard_text(harness_factory):
    # Long agent audio so the reply is still playing when the user talks over it.
    h = await harness_factory(fast_settings(mock_tts_ms_per_char=60.0))
    await h.say()
    await wait_for(lambda: h.session.state.value == "speaking", message="agent speaking")
    await asyncio.sleep(0.1)  # let synthesis finish; playback still has seconds to go

    await h.feed(speech(200))  # the user starts talking over the agent...
    started = asyncio.get_running_loop().time()
    await h.pipeline.on_message(json.dumps({"type": "interrupt", "turn_id": 1, "played_ms": 500}))
    assert asyncio.get_running_loop().time() - started < 0.3  # not after the ~3.5 s of queued audio
    assert h.session.state.value == "listening"

    # ...and what they said becomes the next turn.
    await h.feed(speech(400) + silence(700))
    await wait_for(lambda: len(h.transport.of("transcript_final")) == 2, message="second transcript")
    assert h.transport.of("transcript_final")[1]["turn_id"] == 2

    saved = await h.store.get_session(h.session.id)
    agent_turn = next(t for t in saved["transcript"] if t["turn_id"] == 1 and t["role"] == "agent")
    assert agent_turn["interrupted"] is True
    # 500 ms into a ~2.1 s first sentence: only its first word was heard.
    assert agent_turn["text"] == "You..."

    # The LLM history holds what was heard, not the full generated reply.
    assistant = [m for m in h.session.conversation.messages if m.role == "assistant"]
    assert [m.text for m in assistant] == ["You..."]


async def test_user_talking_while_agent_thinks_cancels_the_reply(harness_factory):
    h = await harness_factory(fast_settings(mock_llm_latency_ms=600))
    await h.say()
    await wait_for(lambda: h.session.state.value == "thinking", message="thinking")
    await h.feed(speech(300))
    assert h.session.state.value == "listening"
    await h.feed(speech(300) + silence(700))
    await h.wait_listening_after(2)

    # Both user turns reached the model, and the cancelled reply left nothing behind.
    second_reply_request = h.llm.requests[1]
    assert [m.role for m in second_reply_request] == ["user", "user"]


async def test_echo_while_speaking_is_not_transcribed(harness_factory):
    h = await harness_factory(fast_settings(mock_tts_ms_per_char=40.0))
    await h.say()
    await wait_for(lambda: h.session.state.value == "speaking", message="agent speaking")
    opened = h.stt.opened
    await h.feed(speech(400) + silence(700))  # no interrupt from the client
    await h.wait_listening_after(1, timeout=5)
    assert h.stt.opened == opened
    assert len(h.transport.of("transcript_final")) == 1


async def test_silence_timeout_says_goodbye_and_ends(harness_factory):
    h = await harness_factory(fast_settings(silence_timeout_s=0.3))
    await asyncio.wait_for(h.pipeline.ended.wait(), 3)
    assert h.transport.of("agent_text")[-1]["text"] == SILENCE_GOODBYE
    assert h.transport.events[-1] == {"type": "call_ended", "reason": "silence_timeout"}
    assert h.transport.closed
    saved = await h.store.get_session(h.session.id)
    assert saved["session"]["end_reason"] == "silence_timeout"


async def test_max_duration_says_goodbye_and_ends(harness_factory):
    h = await harness_factory(fast_settings(max_call_duration_s=0.3))
    await asyncio.wait_for(h.pipeline.ended.wait(), 3)
    assert h.transport.of("agent_text")[-1]["text"] == MAX_DURATION_GOODBYE
    assert h.transport.events[-1] == {"type": "call_ended", "reason": "max_duration"}


async def test_end_call_from_client(harness_factory):
    h = await harness_factory()
    await h.pipeline.on_message(json.dumps({"type": "end_call"}))
    assert h.pipeline.ended.is_set()
    assert h.session.state.value == "ended"
    assert h.transport.events[-1] == {"type": "call_ended", "reason": "user_ended"}


async def test_disconnect_marks_session_ended_and_keeps_transcript(harness_factory):
    h = await harness_factory()
    await h.say()
    await h.wait_listening_after(1)
    await h.pipeline.on_disconnect()
    saved = await h.store.get_session(h.session.id)
    assert saved["session"]["end_reason"] == "disconnected"
    assert saved["session"]["ended_at"]
    assert len(saved["transcript"]) == 2


async def test_bad_client_messages_get_an_error(harness_factory):
    h = await harness_factory()
    await h.pipeline.on_message("not json")
    await h.pipeline.on_message(json.dumps({"type": "dance"}))
    await h.flush()
    assert [e["code"] for e in h.transport.of("error")] == ["bad_message", "unknown_event"]


async def test_interrupt_with_nothing_playing_is_acknowledged(harness_factory):
    h = await harness_factory()
    await h.pipeline.on_message(json.dumps({"type": "interrupt", "turn_id": 7}))
    await h.flush()
    assert h.transport.states() == ["listening", "listening"]
