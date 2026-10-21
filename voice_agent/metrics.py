"""Per-turn latency timing and summary statistics.

Stages are measured back to back from the moment the user stopped talking, so
the four stage values add up to ``total_ms``:

    speech_end --vad_end--> vad_end --stt_final--> stt_final
               --llm_first_token--> first LLM token --tts_first_audio--> first audio sent
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping

STAGES = ("vad_end_ms", "stt_final_ms", "llm_first_token_ms", "tts_first_audio_ms", "total_ms")


@dataclass
class TurnTimer:
    """Wall-clock marks (seconds, from the pipeline's monotonic clock) for one turn."""

    speech_end: float | None = None  # last voiced frame received
    vad_end: float | None = None  # VAD declared end of utterance
    stt_final: float | None = None  # final transcript returned
    llm_first_token: float | None = None  # first text token of the reply
    tts_first_audio: float | None = None  # first audio chunk queued to the client

    def as_metrics(self) -> dict[str, float | None]:
        return {
            "vad_end_ms": _delta(self.speech_end, self.vad_end),
            "stt_final_ms": _delta(self.vad_end, self.stt_final),
            "llm_first_token_ms": _delta(self.stt_final, self.llm_first_token),
            "tts_first_audio_ms": _delta(self.llm_first_token, self.tts_first_audio),
            "total_ms": _delta(self.speech_end, self.tts_first_audio),
        }


def _delta(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start) * 1000.0, 1)


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (same as numpy's default)."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize(rows: Iterable[Mapping[str, float | None]]) -> dict[str, dict[str, float | int | None]]:
    """Median and p90 per stage, ignoring turns where a stage didn't happen."""
    rows = list(rows)
    summary: dict[str, dict[str, float | int | None]] = {}
    for stage in STAGES:
        values = [float(row[stage]) for row in rows if row.get(stage) is not None]
        summary[stage] = {
            "n": len(values),
            "median": round(statistics.median(values), 1) if values else None,
            "p90": round(percentile(values, 90), 1) if values else None,
        }
    return summary
