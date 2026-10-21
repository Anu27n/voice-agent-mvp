"""REST endpoints: access code, call limit, history, transcripts, metrics."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from conftest import fast_settings
from voice_agent.main import create_app


def client(tmp_path, **overrides) -> TestClient:
    settings = fast_settings(db_path=str(tmp_path / "calls.db"), mock_tts_ms_per_char=2.0, **overrides)
    return TestClient(create_app(settings))


def typed_call(api: TestClient, text: str, headers: dict | None = None, code: str | None = None) -> str:
    """One call with one typed turn, ended by the client. Returns the session id."""
    session_id = api.post("/sessions", json={"timezone": "UTC"}, headers=headers or {}).json()["session_id"]
    query = f"session_id={session_id}" + (f"&code={code}" if code else "")
    with api.websocket_connect(f"/ws/voice?{query}") as ws:
        assert json.loads(ws.receive_text()) == {"type": "state", "state": "listening"}
        ws.send_text(json.dumps({"type": "user_text", "text": text}))
        while True:
            message = ws.receive()
            if message.get("text") and json.loads(message["text"])["type"] == "metrics":
                break
        ws.send_text(json.dumps({"type": "end_call"}))
        while json.loads(ws.receive_text())["type"] != "call_ended":
            pass
    return session_id


def test_health_reports_demo_mode_and_limits(tmp_path):
    with client(tmp_path) as api:
        health = api.get("/healthz").json()
    assert health["demo_mode"] is True
    assert health["providers"] == {"stt": "mock", "llm": "mock", "tts": "mock"}
    assert health["access_code_required"] is False
    assert health["max_active_calls"] == 10


def test_history_transcript_and_metrics_after_a_call(tmp_path):
    with client(tmp_path) as api:
        session_id = typed_call(api, "What is 12 times 7?")

        calls = api.get("/sessions").json()["sessions"]
        assert calls[0]["id"] == session_id
        assert calls[0]["user_turns"] == 1 and calls[0]["end_reason"] == "user_ended"
        assert calls[0]["first_words"] == "What is 12 times 7?"
        assert calls[0]["state"] == "ended"

        markdown = api.get(f"/sessions/{session_id}/transcript.md")
        assert markdown.headers["content-type"].startswith("text/markdown")
        assert f"# Call {session_id}" in markdown.text
        assert "**Caller** (turn 1): What is 12 times 7?" in markdown.text
        assert "tool `calculate(" in markdown.text and "That comes to 84." in markdown.text
        assert "| Turn | VAD | STT | LLM | TTS | Total |" in markdown.text

        summary = api.get("/metrics/summary?last=20").json()
        assert summary["turns"] == 1
        assert summary["stages"]["llm_first_token_ms"]["n"] == 1
        assert summary["stages"]["total_ms"]["n"] == 0  # typed: no speech stages

        prometheus = api.get("/metrics").text
        assert 'voice_agent_calls_total{end_reason="user_ended"} 1' in prometheus
        assert 'voice_agent_turns_total{role="user"} 1' in prometheus
        assert "voice_agent_active_calls 0" in prometheus
        assert 'voice_agent_stage_latency_ms_count{stage="llm_first_token"} 1' in prometheus

        assert api.get("/sessions/nope/transcript.md").status_code == 404
        assert api.get("/history.html").status_code == 200


def test_access_code_protects_the_api_and_the_socket(tmp_path):
    with client(tmp_path, access_code="open-sesame") as api:
        assert api.get("/healthz").status_code == 200  # health checks stay open
        assert api.get("/healthz").json()["access_code_required"] is True
        assert api.post("/sessions").status_code == 401
        assert api.get("/sessions").status_code == 401
        assert api.get("/metrics").status_code == 401
        assert api.post("/sessions", headers={"X-Access-Code": "wrong"}).status_code == 401
        assert api.get("/sessions?code=open-sesame").status_code == 200

        session_id = api.post("/sessions", headers={"X-Access-Code": "open-sesame"}).json()["session_id"]
        with api.websocket_connect(f"/ws/voice?session_id={session_id}") as ws:
            assert json.loads(ws.receive_text())["code"] == "unauthorized"
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_text()
            assert closed.value.code == 4401

        typed_call(api, "Hello there.", headers={"X-Access-Code": "open-sesame"}, code="open-sesame")


def test_concurrent_call_limit(tmp_path):
    with client(tmp_path, max_active_calls=2) as api:
        assert api.post("/sessions").status_code == 201
        assert api.post("/sessions").status_code == 201
        refused = api.post("/sessions")
        assert refused.status_code == 429
        assert "Too many calls" in refused.json()["detail"]


def test_json_log_lines():
    import logging

    from voice_agent.__main__ import JsonFormatter

    record = logging.LogRecord("voice_agent.pipeline", logging.INFO, __file__, 1, "turn %d done", (3,), None)
    line = json.loads(JsonFormatter().format(record))
    assert line["level"] == "info" and line["logger"] == "voice_agent.pipeline" and line["message"] == "turn 3 done"
    assert line["time"].endswith("+00:00")
