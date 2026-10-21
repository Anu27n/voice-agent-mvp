"""Latency benchmark: drive a running server with recorded speech.

Acts like a caller. It streams the recorded questions in
``tests/fixtures/two_turns.wav`` into a new call in real time, waits while the
agent thinks and speaks (the "mic" keeps streaming silence, as a real one
would), and replays the recording until the requested number of turns is done.
Then it ends the call and prints the median and p90 per stage, as measured and
stored by the server.

    python -m voice_agent                          # terminal 1: real or mock providers
    python -m voice_agent.benchmark --turns 20     # terminal 2
    python -m voice_agent.benchmark --turns 20 --out docs/latency-report.md

Keep ``--speed 1`` (real time) for numbers worth reporting: the VAD stage is
measured in wall-clock time, so faster playback understates it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
from websockets.asyncio.client import connect

from .audio import iter_frames, read_wav
from .report import build_report

DEFAULT_WAV = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "two_turns.wav"
SECONDS_PER_TURN_LIMIT = 45.0  # give up if a turn takes longer than this on average


@dataclass
class BenchmarkResult:
    session_id: str
    session: dict[str, Any]  # GET /sessions/{id}
    events: list[dict[str, Any]]
    speed: float

    @property
    def metrics(self) -> list[dict[str, Any]]:
        return self.session["metrics"]

    def report(self) -> str:
        providers = self.session["session"]["config"]["providers"]
        model = self.session["session"]["config"].get("llm_model")
        stack = ", ".join(f"{stage}={name}" for stage, name in providers.items())
        note = (
            f"Session `{self.session_id}` ({stack}{', model=' + model if model else ''}). "
            f"Recorded speech streamed at {self.speed:g}x real time; stages as measured by the server."
        )
        rows = [{**m, "session_id": self.session_id} for m in self.metrics]
        return build_report(rows, title="Latency report", note=note)


class _Pacer:
    """Sleeps so that ticks happen every ``interval`` seconds on average, without drift."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.next = time.perf_counter()

    async def tick(self) -> None:
        self.next += self.interval
        delay = self.next - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        elif delay < -1.0:
            self.next = time.perf_counter()  # fell far behind; don't burst to catch up


async def run_benchmark(
    base_url: str,
    *,
    turns: int = 20,
    wav: Path = DEFAULT_WAV,
    speed: float = 1.0,
    timezone: str | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> BenchmarkResult:
    pcm, rate = read_wav(wav)
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as http:
        created = await http.post("/sessions", json={"timezone": timezone} if timezone else None)
        created.raise_for_status()
        info = created.json()
        if info["input_sample_rate"] != rate:
            raise ValueError(f"{wav} is {rate} Hz but the server expects {info['input_sample_rate']} Hz")
        session_id = info["session_id"]
        frames = list(iter_frames(pcm, rate, info["frame_ms"]))
        quiet = bytes(len(frames[0]))
        ws_url = base_url.replace("http", "ws", 1).rstrip("/") + f"/ws/voice?session_id={session_id}"

        events: list[dict[str, Any]] = []
        busy = False
        completed = 0
        ended = False

        async with connect(ws_url) as ws:

            async def receive() -> None:
                nonlocal busy, completed, ended
                async for message in ws:
                    if isinstance(message, bytes):
                        continue
                    event = json.loads(message)
                    events.append(event)
                    if on_event is not None:
                        on_event(event)
                    if event["type"] == "state":
                        busy = event["state"] in ("thinking", "speaking")
                    elif event["type"] == "metrics":
                        completed += 1
                    elif event["type"] == "call_ended":
                        ended = True
                        return

            receiver = asyncio.create_task(receive())
            pacer = _Pacer(info["frame_ms"] / 1000.0 / speed)
            deadline = time.monotonic() + turns * SECONDS_PER_TURN_LIMIT
            try:
                while completed < turns and not ended:
                    for frame in frames:
                        while busy and not ended:  # the caller waits for the reply
                            await ws.send(quiet)
                            await pacer.tick()
                        if completed >= turns or ended:
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"only {completed} of {turns} turns completed")
                        await ws.send(frame)
                        await pacer.tick()
                while busy and not ended:  # let the last reply finish playing
                    await ws.send(quiet)
                    await pacer.tick()
                if not ended:
                    await ws.send(json.dumps({"type": "end_call"}))
                await asyncio.wait_for(receiver, 10.0)
            finally:
                receiver.cancel()

        saved = (await http.get(f"/sessions/{session_id}")).json()
    return BenchmarkResult(session_id, saved, events, speed)


def _progress(event: dict[str, Any]) -> None:
    if event["type"] == "transcript_final":
        print(f"turn {event['turn_id']:>2}  heard: {event['text']}")
    elif event["type"] == "error":
        print(f"         error: {event.get('code')}: {event.get('message')}")
    elif event["type"] == "metrics" and event.get("total_ms") is not None:
        stages = " ".join(
            f"{name} {event[key]:.0f}"
            for name, key in (
                ("vad", "vad_end_ms"),
                ("stt", "stt_final_ms"),
                ("llm", "llm_first_token_ms"),
                ("tts", "tts_first_audio_ms"),
            )
            if event.get(key) is not None
        )
        print(f"         total {event['total_ms']:.0f} ms ({stages})")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m voice_agent.benchmark",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="server base URL")
    parser.add_argument("--turns", type=int, default=20, help="turns to complete (default 20)")
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV, help="mono PCM16 WAV at the server's input rate")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed; 1 = real time")
    parser.add_argument("--timezone", default=None, help="caller time zone sent with the session")
    parser.add_argument("--out", type=Path, default=None, help="also write the report to this Markdown file")
    args = parser.parse_args()

    try:
        result = asyncio.run(
            run_benchmark(
                args.url,
                turns=args.turns,
                wav=args.wav,
                speed=args.speed,
                timezone=args.timezone,
                on_event=_progress,
            )
        )
    except (httpx.HTTPError, OSError) as exc:
        sys.exit(f"could not reach the server at {args.url}: {exc}")
    report = result.report()
    print()
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    if result.session["errors"]:
        print(f"note: {len(result.session['errors'])} error(s) were recorded during the run")


if __name__ == "__main__":
    main()
