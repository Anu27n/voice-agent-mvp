"""Reports: the latency report, call transcripts as Markdown, Prometheus metrics."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable, Mapping

from .metrics import STAGES, percentile, summarize

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


def transcript_markdown(data: Mapping[str, Any]) -> str:
    """One call (as returned by Store.get_session) as a readable Markdown document."""
    session = data["session"]
    providers = session["config"].get("providers") or {}
    lines = [
        f"# Call {session['id']}",
        "",
        f"- Started: {session['started_at']}",
        f"- Ended: {session.get('ended_at') or 'in progress'}"
        + (f" ({session['end_reason']})" if session.get("end_reason") else ""),
        f"- Providers: {', '.join(f'{k}={v}' for k, v in providers.items())}",
        f"- Prompt version: {session['config'].get('prompt_version', '-')}",
        "",
        "## Transcript",
        "",
    ]
    for turn in data["transcript"]:
        who = "Caller" if turn["role"] == "user" else "Agent"
        note = " _(interrupted)_" if turn["interrupted"] else ""
        lines.append(f"**{who}** (turn {turn['turn_id']}){note}: {turn['text'] or '_(nothing heard)_'}")
        for call in turn["tool_calls"]:
            outcome = "error" if call.get("is_error") else "result"
            args = json.dumps(call.get("arguments", {}))
            lines.append(f"  - tool `{call['name']}({args})` {outcome}: `{call.get('result')}`")
        lines.append("")
    if data["metrics"]:
        lines += [
            "## Latency (ms)",
            "",
            "| Turn | VAD | STT | LLM | TTS | Total |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for m in data["metrics"]:
            cells = [m.get(stage) for stage in STAGES]
            lines.append(f"| {m['turn_id']} | " + " | ".join("-" if c is None else f"{c:.0f}" for c in cells) + " |")
        lines.append("")
    if data["errors"]:
        lines += ["## Errors", ""]
        lines += [f"- turn {e['turn_id']}: {e['stage']} `{e['code']}`: {e['message']}" for e in data["errors"]]
        lines.append("")
    return "\n".join(lines)


def prometheus_text(counts: Mapping[str, Mapping[str, int]], recent: list[Mapping[str, Any]], active: int) -> str:
    """Prometheus text exposition: call/turn/error totals and recent stage latency."""

    def label(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    out = [
        "# HELP voice_agent_active_calls Calls currently in progress.",
        "# TYPE voice_agent_active_calls gauge",
        f"voice_agent_active_calls {active}",
        "# HELP voice_agent_calls_total Calls by how they ended.",
        "# TYPE voice_agent_calls_total counter",
        *(f'voice_agent_calls_total{{end_reason="{label(k)}"}} {v}' for k, v in sorted(counts["calls"].items())),
        "# HELP voice_agent_turns_total Conversation turns by speaker.",
        "# TYPE voice_agent_turns_total counter",
        *(f'voice_agent_turns_total{{role="{label(k)}"}} {v}' for k, v in sorted(counts["turns"].items())),
        "# HELP voice_agent_barge_ins_total Agent replies the caller interrupted.",
        "# TYPE voice_agent_barge_ins_total counter",
        f"voice_agent_barge_ins_total {sum(counts['interrupted'].values())}",
        "# HELP voice_agent_errors_total Recoverable errors by pipeline stage.",
        "# TYPE voice_agent_errors_total counter",
        *(f'voice_agent_errors_total{{stage="{label(k)}"}} {v}' for k, v in sorted(counts["errors"].items())),
        "# HELP voice_agent_stage_latency_ms Per-stage latency over recent turns.",
        "# TYPE voice_agent_stage_latency_ms summary",
    ]
    for stage in STAGES:
        values = [float(r[stage]) for r in recent if r.get(stage) is not None]
        name = stage.removesuffix("_ms")
        for q in (0.5, 0.9, 0.99):
            if values:
                out.append(
                    f'voice_agent_stage_latency_ms{{stage="{name}",quantile="{q}"}} {percentile(values, q * 100):.1f}'
                )
        out.append(f'voice_agent_stage_latency_ms_count{{stage="{name}"}} {len(values)}')
        out.append(f'voice_agent_stage_latency_ms_sum{{stage="{name}"}} {sum(values):.1f}')
    return "\n".join(out) + "\n"
