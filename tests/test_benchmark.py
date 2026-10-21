"""The latency benchmark drives a real server with the recorded WAV."""

from conftest import SPOKEN
from voice_agent.benchmark import run_benchmark


async def test_benchmark_completes_the_requested_turns_and_reports(live_server):
    host, _ = live_server
    # 4x real time keeps the test quick; real benchmarks run at 1x.
    result = await run_benchmark(f"http://{host}", turns=4, speed=4.0, timezone="UTC")

    assert len(result.metrics) == 4
    assert all(m["total_ms"] is not None for m in result.metrics)
    heard = [t["text"] for t in result.session["transcript"] if t["role"] == "user"]
    assert heard == [SPOKEN[0], SPOKEN[1], SPOKEN[0], SPOKEN[1]]  # the recording, replayed
    assert result.session["session"]["end_reason"] == "user_ended"
    assert result.session["errors"] == []

    report = result.report()
    assert "| Total (speech end -> first audio) | 4 |" in report
    assert "stt=mock, llm=mock, tts=mock" in report
    assert "4x real time" in report
