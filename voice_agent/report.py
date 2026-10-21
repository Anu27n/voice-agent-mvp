"""Markdown latency report (median and p90 per stage)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping

from .metrics import STAGES, summarize

LABELS = {
    "vad_end_ms": "VAD end of speech",
    "stt_final_ms": "STT final transcript",
    "llm_first_token_ms": "LLM first token",
    "tts_first_audio_ms": "TTS first audio",
    "total_ms": "Total (speech end -> first audio)",
}
BUDGET_MS = 1500


def build_report(rows: Iterable[Mapping[str, Any]], title: str = "Latency report", note: str = "") -> str:
    rows = list(rows)
    summary = summarize(rows)
    sessions = len({row.get("session_id") for row in rows})
    lines = [
        f"# {title}",
        "",
        f"{len(rows)} turns from {sessions} call(s), generated {datetime.now():%Y-%m-%d %H:%M}.",
    ]
    if note:
        lines += ["", note]
    lines += [
        "",
        "| Stage | Turns | Median (ms) | p90 (ms) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for stage in STAGES:
        s = summary[stage]
        median = "-" if s["median"] is None else f"{s['median']:.0f}"
        p90 = "-" if s["p90"] is None else f"{s['p90']:.0f}"
        lines.append(f"| {LABELS[stage]} | {s['n']} | {median} | {p90} |")
    total = summary["total_ms"]["median"]
    if total is not None:
        verdict = "within" if total <= BUDGET_MS else "over"
        lines += ["", f"Median total is **{total:.0f} ms**, {verdict} the {BUDGET_MS} ms budget."]
    return "\n".join(lines) + "\n"
