import pytest

from voice_agent.metrics import TurnTimer, percentile, summarize
from voice_agent.store import Store
from voice_agent.timeline import SpeechTimeline


def timeline():
    t = SpeechTimeline()
    first = t.begin("One two three four.", round_index=0)
    t.add_audio(first, 1000)
    first.complete = True
    second = t.begin("Five six.", round_index=1)
    t.add_audio(second, 500)
    second.complete = True
    return t


def test_spoken_text_by_heard_time():
    t = timeline()
    assert t.spoken_text(0) == ""
    assert t.spoken_text(500) == "One two..."
    assert t.spoken_text(1000) == "One two three four."
    assert t.spoken_text(1250) == "One two three four. Five..."
    assert t.spoken_text(5000) == t.full_text() == "One two three four. Five six."


def test_spoken_text_filtered_by_round():
    assert timeline().spoken_text(1250, rounds={1}) == "Five..."


def test_sentence_still_synthesizing_uses_an_estimated_length():
    t = SpeechTimeline()
    segment = t.begin("A fairly long sentence that is still arriving.")
    t.add_audio(segment, 200)  # only 200 ms synthesized so far
    assert t.spoken_text(200) == ""  # nowhere near the end of the sentence


def test_turn_timer_stages_add_up():
    timer = TurnTimer(speech_end=10.0, vad_end=10.6, stt_final=10.8, llm_first_token=11.2, tts_first_audio=11.45)
    metrics = timer.as_metrics()
    assert metrics == {
        "vad_end_ms": 600.0,
        "stt_final_ms": 200.0,
        "llm_first_token_ms": 400.0,
        "tts_first_audio_ms": 250.0,
        "total_ms": 1450.0,
    }
    assert TurnTimer(speech_end=1.0).as_metrics()["total_ms"] is None


def test_percentile_and_summary():
    values = [float(v) for v in range(1, 11)]
    assert percentile(values, 50) == 5.5
    assert percentile(values, 90) == pytest.approx(9.1)
    rows = [{"total_ms": v, "vad_end_ms": None} for v in values]
    summary = summarize(rows)
    assert summary["total_ms"] == {"n": 10, "median": 5.5, "p90": 9.1}
    assert summary["vad_end_ms"] == {"n": 0, "median": None, "p90": None}


async def test_store_round_trip():
    store = Store(":memory:")
    await store.create_session("s1", "2026-10-21T10:00:00Z", {"providers": {"tts": "mock"}})
    await store.add_turn("s1", 1, "user", "hello")
    await store.add_turn("s1", 1, "agent", "hi...", interrupted=True, tool_calls=[{"name": "t"}])
    await store.add_metrics("s1", 1, {"vad_end_ms": 600.0, "total_ms": 1400.0})
    await store.add_error("s1", 1, "tts", "tts_timeout", "slow")
    await store.end_session("s1", "user_ended")
    await store.end_session("s1", "disconnected")  # the first end reason wins

    data = await store.get_session("s1")
    assert data["session"]["end_reason"] == "user_ended"
    assert data["session"]["config"] == {"providers": {"tts": "mock"}}
    assert [(t["role"], t["text"], t["interrupted"]) for t in data["transcript"]] == [
        ("user", "hello", False),
        ("agent", "hi...", True),
    ]
    assert data["transcript"][1]["tool_calls"] == [{"name": "t"}]
    assert data["metrics"][0]["total_ms"] == 1400.0 and data["metrics"][0]["stt_final_ms"] is None
    assert data["errors"][0]["code"] == "tts_timeout"
    assert store.recent_metrics(limit=5)[0]["turn_id"] == 1
    assert await store.get_session("missing") is None
    store.close()
