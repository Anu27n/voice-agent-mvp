// Call history: recent calls, their transcripts and per-stage latency.

import { api, formatMs, withCode } from "./common.js";

const $ = (id) => document.getElementById(id);
const STAGES = [
  ["vad_end_ms", "VAD end of speech"],
  ["stt_final_ms", "STT final transcript"],
  ["llm_first_token_ms", "LLM first token"],
  ["tts_first_audio_ms", "TTS first audio"],
  ["total_ms", "Total (speech end to first audio)"],
];
const BUDGET_MS = 1500;

async function getJson(path) {
  const res = await api(path);
  if (!res.ok) throw new Error(res.status === 401 ? "an access code is required" : `HTTP ${res.status}`);
  return res.json();
}

function cell(row, text, className) {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  row.append(td);
  return td;
}

function showError(err) {
  $("banner-text").textContent = `Couldn't load the history: ${err.message}.`;
  $("banner").dataset.kind = "error";
  $("banner").hidden = false;
}

async function loadSummary() {
  const { stages } = await getJson("/metrics/summary?last=20");
  const body = $("summary").querySelector("tbody");
  body.replaceChildren();
  for (const [key, label] of STAGES) {
    const row = document.createElement("tr");
    cell(row, label);
    cell(row, String(stages[key].n), "num");
    cell(row, formatMs(stages[key].median), "num");
    cell(row, formatMs(stages[key].p90), "num");
    body.append(row);
  }
  const total = stages.total_ms.median;
  $("verdict").textContent =
    total == null
      ? "No spoken turns yet."
      : `Median total ${Math.round(total)} ms: ${total <= BUDGET_MS ? "within" : "over"} the ${BUDGET_MS} ms budget.`;
}

function duration(start, end) {
  if (!end) return "in progress";
  const seconds = Math.max(0, Math.round((new Date(end) - new Date(start)) / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

async function loadCalls() {
  const { sessions } = await getJson("/sessions?limit=50");
  const body = $("calls").querySelector("tbody");
  body.replaceChildren();
  $("calls-empty").hidden = sessions.length > 0;
  for (const s of sessions) {
    const row = document.createElement("tr");
    row.tabIndex = 0;
    cell(row, new Date(s.started_at).toLocaleString());
    cell(row, duration(s.started_at, s.ended_at), "num");
    cell(row, String(s.user_turns), "num");
    cell(row, formatMs(s.median_total_ms), "num");
    cell(row, (s.end_reason || s.state || "").replaceAll("_", " "));
    const words = s.first_words || "";
    cell(row, words.length > 60 ? `${words.slice(0, 57)}...` : words, "muted");
    row.addEventListener("click", () => showCall(s.id));
    row.addEventListener("keydown", (event) => event.key === "Enter" && showCall(s.id));
    body.append(row);
  }
}

async function showCall(id) {
  let data;
  try {
    data = await getJson(`/sessions/${encodeURIComponent(id)}`);
  } catch (err) {
    showError(err);
    return;
  }
  const { session, transcript, metrics, errors } = data;
  location.hash = id;
  $("detail").hidden = false;
  $("detail-title").textContent = `Call ${id.slice(0, 8)}`;
  const providers = Object.entries(session.config.providers || {})
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
  const tz = session.config.caller_timezone ? ` | caller time zone ${session.config.caller_timezone}` : "";
  const reason = session.end_reason ? ` (${session.end_reason.replaceAll("_", " ")})` : "";
  $("detail-meta").textContent =
    `${new Date(session.started_at).toLocaleString()} | ${duration(session.started_at, session.ended_at)}${reason}` +
    ` | ${providers}${tz} | ${errors.length} error${errors.length === 1 ? "" : "s"}`;
  $("download-md").href = withCode(`/sessions/${encodeURIComponent(id)}/transcript.md`);
  $("download-json").href = withCode(`/sessions/${encodeURIComponent(id)}`);

  const box = $("detail-transcript");
  box.replaceChildren();
  for (const turn of transcript) {
    const el = document.createElement("div");
    el.className = `turn ${turn.role === "user" ? "user" : "agent"}${turn.interrupted ? " interrupted" : ""}`;
    const text = document.createElement("p");
    text.className = "text";
    text.textContent = turn.text || "(nothing heard)";
    el.append(text);
    for (const call of turn.tool_calls) {
      const chip = document.createElement("div");
      chip.className = `tool ${call.is_error ? "failed" : "ok"}`;
      chip.textContent = `${call.name}(${JSON.stringify(call.arguments)}) -> ${call.result}`;
      el.append(chip);
    }
    box.append(el);
  }

  const body = $("detail-metrics").querySelector("tbody");
  body.replaceChildren();
  for (const m of metrics) {
    const row = document.createElement("tr");
    cell(row, String(m.turn_id), "num");
    for (const [key] of STAGES) cell(row, formatMs(m[key]), "num");
    body.append(row);
  }
  $("detail").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function refresh() {
  try {
    await loadSummary(); // one at a time, so an access-code prompt appears only once
    await loadCalls();
  } catch (err) {
    showError(err);
  }
}

$("refresh").addEventListener("click", refresh);
await refresh();
if (location.hash.length > 1) showCall(location.hash.slice(1));
