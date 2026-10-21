"""Median and p90 latency per pipeline stage over the most recent turns.

python scripts/latency_report.py                      # last 20 turns, all calls
python scripts/latency_report.py --session <id>       # one call
python scripts/latency_report.py --out docs/latency-report.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice_agent.config import Settings  # noqa: E402
from voice_agent.metrics import STAGES, summarize  # noqa: E402
from voice_agent.store import Store  # noqa: E402

LABELS = {
    "vad_end_ms": "VAD end of speech",
    "stt_final_ms": "STT final transcript",
    "llm_first_token_ms": "LLM first token",
    "tts_first_audio_ms": "TTS first audio",
    "total_ms": "Total (speech end -> first audio)",
}
BUDGET_MS = 1500


def build_report(rows: list[dict], title: str) -> str:
    summary = summarize(rows)
    sessions = len({row["session_id"] for row in rows})
    lines = [
        f"# {title}",
        "",
        f"{len(rows)} turns from {sessions} call(s), generated {datetime.now():%Y-%m-%d %H:%M}.",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=None, help="SQLite file (default: DB_PATH from the environment)")
    parser.add_argument("--last", type=int, default=20, help="number of most recent turns")
    parser.add_argument("--session", default=None, help="only this session id")
    parser.add_argument("--out", default=None, help="also write the report to this Markdown file")
    args = parser.parse_args()

    db_path = args.db or Settings.from_env().db_path
    if not Path(db_path).exists():
        sys.exit(f"no database at {db_path}; make some calls first")
    store = Store(db_path)
    rows = store.recent_metrics(limit=args.last, session_id=args.session)
    store.close()
    if not rows:
        sys.exit("no turns recorded yet")

    report = build_report(rows, "Latency report")
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
