"""Median and p90 latency per pipeline stage over the most recent turns.

python scripts/latency_report.py                      # last 20 turns, all calls
python scripts/latency_report.py --session <id>       # one call
python scripts/latency_report.py --out docs/latency-report.md

To generate 20 turns automatically, see ``python -m voice_agent.benchmark``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice_agent.config import Settings  # noqa: E402
from voice_agent.report import build_report  # noqa: E402
from voice_agent.store import Store  # noqa: E402


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

    report = build_report(rows)
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
