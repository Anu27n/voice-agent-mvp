"""SQLite persistence for sessions, turns, per-turn metrics and errors.

Rows are written as the call happens (not only at the end), so a crash or a
dropped socket still leaves the transcript up to that point. Calls run in a
worker thread to keep the event loop free.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    end_reason  TEXT,
    config      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    turn_id     INTEGER NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'agent')),
    text        TEXT NOT NULL,
    interrupted INTEGER NOT NULL DEFAULT 0,
    tool_calls  TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL,
    UNIQUE (session_id, turn_id, role)
);
CREATE TABLE IF NOT EXISTS metrics (
    session_id         TEXT NOT NULL REFERENCES sessions(id),
    turn_id            INTEGER NOT NULL,
    vad_end_ms         REAL,
    stt_final_ms       REAL,
    llm_first_token_ms REAL,
    tts_first_audio_ms REAL,
    total_ms           REAL,
    created_at         TEXT NOT NULL,
    PRIMARY KEY (session_id, turn_id)
);
CREATE TABLE IF NOT EXISTS errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    turn_id     INTEGER,
    stage       TEXT NOT NULL,
    code        TEXT NOT NULL,
    message     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""

METRIC_COLUMNS = ("vad_end_ms", "stt_final_ms", "llm_first_token_ms", "tts_first_audio_ms", "total_ms")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Store:
    def __init__(self, path: str | Path) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    async def _run(self, sql: str, params: tuple = ()) -> None:
        await asyncio.to_thread(self._execute, sql, params)

    def _execute(self, sql: str, params: tuple) -> None:
        with self._lock, self._conn:
            self._conn.execute(sql, params)

    def _query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------------ writes

    async def create_session(self, session_id: str, started_at: str, config: dict[str, Any]) -> None:
        await self._run(
            "INSERT INTO sessions (id, started_at, config) VALUES (?, ?, ?)",
            (session_id, started_at, json.dumps(config)),
        )

    async def end_session(self, session_id: str, reason: str) -> None:
        await self._run(
            "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL",
            (utcnow(), reason, session_id),
        )

    async def add_turn(
        self,
        session_id: str,
        turn_id: int,
        role: str,
        text: str,
        *,
        interrupted: bool = False,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        await self._run(
            "INSERT OR REPLACE INTO turns (session_id, turn_id, role, text, interrupted, tool_calls, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, turn_id, role, text, int(interrupted), json.dumps(tool_calls or []), utcnow()),
        )

    async def add_metrics(self, session_id: str, turn_id: int, metrics: dict[str, float | None]) -> None:
        await self._run(
            "INSERT OR REPLACE INTO metrics (session_id, turn_id, vad_end_ms, stt_final_ms, llm_first_token_ms, "
            "tts_first_audio_ms, total_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, turn_id, *(metrics.get(c) for c in METRIC_COLUMNS), utcnow()),
        )

    async def add_error(self, session_id: str, turn_id: int | None, stage: str, code: str, message: str) -> None:
        await self._run(
            "INSERT INTO errors (session_id, turn_id, stage, code, message, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, turn_id, stage, code, message, utcnow()),
        )

    # ------------------------------------------------------------------- reads

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_session, session_id)

    def _get_session(self, session_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM sessions WHERE id = ?", (session_id,))
        if not rows:
            return None
        session = rows[0]
        session["config"] = json.loads(session["config"])
        turns = self._query(
            "SELECT turn_id, role, text, interrupted, tool_calls, created_at FROM turns "
            "WHERE session_id = ? ORDER BY turn_id, CASE role WHEN 'user' THEN 0 ELSE 1 END",
            (session_id,),
        )
        for turn in turns:
            turn["interrupted"] = bool(turn["interrupted"])
            turn["tool_calls"] = json.loads(turn["tool_calls"])
        metrics = self._query(
            f"SELECT turn_id, {', '.join(METRIC_COLUMNS)} FROM metrics WHERE session_id = ? ORDER BY turn_id",
            (session_id,),
        )
        errors = self._query(
            "SELECT turn_id, stage, code, message, created_at FROM errors WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        return {"session": session, "transcript": turns, "metrics": metrics, "errors": errors}

    def recent_metrics(self, limit: int = 20, session_id: str | None = None) -> list[dict[str, Any]]:
        where, params = ("WHERE session_id = ?", (session_id, limit)) if session_id else ("", (limit,))
        return self._query(
            f"SELECT session_id, turn_id, {', '.join(METRIC_COLUMNS)} FROM metrics {where} "
            "ORDER BY created_at DESC LIMIT ?",
            params,
        )

    async def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_sessions, limit)

    def _list_sessions(self, limit: int) -> list[dict[str, Any]]:
        rows = self._query(
            """
            SELECT s.id, s.started_at, s.ended_at, s.end_reason, s.config,
                   (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id AND t.role = 'user') AS user_turns,
                   (SELECT COUNT(*) FROM errors e WHERE e.session_id = s.id) AS errors,
                   (SELECT text FROM turns t WHERE t.session_id = s.id AND t.role = 'user'
                     ORDER BY t.turn_id LIMIT 1) AS first_words
            FROM sessions s ORDER BY s.started_at DESC LIMIT ?
            """,
            (limit,),
        )
        for row in rows:
            row["providers"] = json.loads(row.pop("config")).get("providers")
            totals = [
                r["total_ms"]
                for r in self._query(
                    "SELECT total_ms FROM metrics WHERE session_id = ? AND total_ms IS NOT NULL", (row["id"],)
                )
            ]
            row["median_total_ms"] = round(statistics.median(totals), 1) if totals else None
        return rows

    def counts(self) -> dict[str, dict[str, int]]:
        """Totals for the metrics endpoint."""

        def grouped(sql: str) -> dict[str, int]:
            return {str(row["key"]): row["n"] for row in self._query(sql)}

        return {
            "calls": grouped(
                "SELECT COALESCE(end_reason, 'in_progress') AS key, COUNT(*) AS n FROM sessions GROUP BY 1"
            ),
            "turns": grouped("SELECT role AS key, COUNT(*) AS n FROM turns GROUP BY 1"),
            "interrupted": grouped("SELECT 'agent' AS key, COUNT(*) AS n FROM turns WHERE interrupted = 1"),
            "errors": grouped("SELECT stage AS key, COUNT(*) AS n FROM errors GROUP BY 1"),
        }
