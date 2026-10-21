// Voice Agent web client: mic capture, playback, captions, status, barge-in,
// typed messages and mute.
//
// Protocol (see docs/DESIGN.md): the client streams PCM16 mono frames at the
// rate POST /sessions returns; the server sends JSON events and binary
// agent_audio frames of [uint32 LE turn_id][PCM16 mono at output_sample_rate].

import { api, withCode } from "./common.js";

const $ = (id) => document.getElementById(id);
const ui = {
  start: $("start"),
  end: $("end"),
  mute: $("mute"),
  status: $("status"),
  statusLabel: $("status-label"),
  demo: $("demo"),
  banner: $("banner"),
  bannerText: $("banner-text"),
  textOnly: $("text-only"),
  transcript: $("transcript"),
  empty: $("empty"),
  partial: $("partial"),
  composer: $("composer"),
  message: $("message"),
  send: $("send"),
  level: $("level"),
  bar: $("bar"),
  stats: $("stats"),
};

// Barge-in: the user counts as talking over the agent when the mic is louder
// than BARGE_IN_DB *and* BARGE_IN_MARGIN_DB above the echo of the agent's own
// voice (whatever the browser's echo cancellation leaves), for BARGE_IN_MS.
const BARGE_IN_DB = -38;
const BARGE_IN_MARGIN_DB = 12;
const BARGE_IN_MS = 160; // keeps stop time well under 300 ms
const ECHO_LEARN_MS = 240; // at the start of playback, only measure the echo
const ECHO_MAX_DB = -32; // cap, so a user already talking when playback starts can still interrupt
const JITTER_S = 0.06; // delay before the first chunk plays, to absorb network jitter
const BUDGET_MS = 1500;

const STATUS_LABELS = {
  idle: "Idle",
  connecting: "Connecting...",
  listening: "Listening",
  thinking: "Thinking...",
  speaking: "Speaking",
  ended: "Call ended",
  disconnected: "Disconnected",
};
const END_REASONS = {
  user_ended: "Call ended.",
  agent_ended: "The agent said goodbye and ended the call.",
  silence_timeout: "Call ended after a long silence.",
  max_duration: "Call ended: maximum call length reached.",
  disconnected: "Call ended: connection lost.",
};

class Call {
  constructor({ textOnly = false } = {}) {
    this.textOnly = textOnly;
    this.sources = new Set(); // scheduled agent audio
    this.cursor = 0; // AudioContext time where the next chunk should start
    this.playingTurn = null;
    this.playedEndedMs = 0;
    this.interrupted = new Set();
    this.muted = false;
    this.loudMs = 0;
    this.ambientDb = -60; // mic level while the agent is silent
    this.echoDb = -60; // mic level while the agent is talking and the user isn't
    this.playbackMs = 0; // how long the current stretch of playback has run
    this.ended = false;
    this.closed = false;
  }

  async start() {
    setStatus("connecting");
    if (!this.textOnly) {
      // Ask for the microphone first: if it's denied, no session is created.
      try {
        this.mic = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        });
      } catch (err) {
        const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
        showBanner(
          denied
            ? "Microphone access was denied. Allow it in your browser's site settings, then press Start Call again."
            : `Couldn't open the microphone: ${err.message || err}`,
          "error",
          0,
          { offerTextOnly: true },
        );
        setStatus("idle");
        return false;
      }
    }

    try {
      const res = await api("/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ timezone: callerTimezone() }),
      });
      if (res.status === 429) throw new Error("too many calls are in progress right now. Try again in a minute");
      if (res.status === 401) throw new Error("an access code is required");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.info = await res.json();
    } catch (err) {
      this.stopMic();
      showBanner(`Couldn't start a call: ${err.message}.`, "error");
      setStatus("idle");
      return false;
    }

    this.ctx = new AudioContext();
    await this.ctx.resume();
    if (this.mic) {
      await this.ctx.audioWorklet.addModule("mic-worklet.js");
      const source = this.ctx.createMediaStreamSource(this.mic);
      this.node = new AudioWorkletNode(this.ctx, "mic-capture", {
        processorOptions: { targetRate: this.info.input_sample_rate, frameMs: this.info.frame_ms },
      });
      this.node.port.onmessage = (event) => this.onMicFrame(event.data);
      const mute = this.ctx.createGain();
      mute.gain.value = 0; // keeps the worklet running without playing the mic back
      source.connect(this.node).connect(mute).connect(this.ctx.destination);
    }

    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const url = `${scheme}://${location.host}/ws/voice?session_id=${encodeURIComponent(this.info.session_id)}`;
    this.ws = new WebSocket(withCode(url));
    this.ws.binaryType = "arraybuffer";
    this.ws.onmessage = (event) =>
      typeof event.data === "string" ? this.onEvent(JSON.parse(event.data)) : this.onAudio(event.data);
    this.ws.onclose = () => this.onClose();
    return true;
  }

  send(event) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(event));
  }

  end() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.send({ type: "end_call" });
    else this.cleanup();
  }

  sendText(text) {
    if (this.isPlaying()) this.bargeIn(); // typing over the agent interrupts it, like talking does
    this.send({ type: "user_text", text });
  }

  setMuted(muted) {
    this.muted = muted;
    this.loudMs = 0;
    if (muted) showLevel(-100);
  }

  // ---------------------------------------------------------------- mic

  onMicFrame({ pcm, db }) {
    if (this.muted) return;
    showLevel(db);
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(pcm);
    this.detectBargeIn(db, this.info.frame_ms);
  }

  detectBargeIn(db, step) {
    if (!this.isPlaying()) {
      this.loudMs = 0;
      this.playbackMs = 0;
      this.ambientDb += 0.05 * (db - this.ambientDb);
      return;
    }
    if (this.playbackMs === 0) this.echoDb = this.ambientDb;
    this.playbackMs += step;
    if (this.playbackMs <= ECHO_LEARN_MS) {
      this.echoDb = Math.min(ECHO_MAX_DB, Math.max(this.echoDb, this.echoDb + 0.3 * (db - this.echoDb)));
      return;
    }
    const threshold = Math.max(BARGE_IN_DB, this.echoDb + BARGE_IN_MARGIN_DB);
    if (db > threshold) {
      this.loudMs += step;
      if (this.loudMs >= BARGE_IN_MS) this.bargeIn();
    } else {
      this.loudMs = Math.max(0, this.loudMs - step);
      this.echoDb = Math.min(ECHO_MAX_DB, this.echoDb + 0.05 * (db - this.echoDb)); // keep tracking the echo
    }
  }

  bargeIn() {
    const turnId = this.playingTurn;
    const playedMs = Math.round(this.playedMs());
    this.stopPlayback();
    this.interrupted.add(turnId);
    this.loudMs = 0;
    this.send({ type: "interrupt", turn_id: turnId, played_ms: playedMs });
    markInterrupted(turnId);
  }

  // ----------------------------------------------------------- playback

  onAudio(buffer) {
    const turnId = new DataView(buffer).getUint32(0, true);
    if (this.interrupted.has(turnId) || !this.ctx) return;
    const pcm = new Int16Array(buffer, 4);
    const samples = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) samples[i] = pcm[i] / 32768;
    const chunk = this.ctx.createBuffer(1, samples.length, this.info.output_sample_rate);
    chunk.copyToChannel(samples, 0);

    if (turnId !== this.playingTurn) {
      this.playingTurn = turnId;
      this.playedEndedMs = 0;
    }
    const now = this.ctx.currentTime;
    const startAt = this.cursor > now ? this.cursor : now + JITTER_S;
    const src = this.ctx.createBufferSource();
    src.buffer = chunk;
    src.connect(this.ctx.destination);
    src.start(startAt);
    this.cursor = startAt + chunk.duration;
    const item = { src, start: startAt, duration: chunk.duration };
    this.sources.add(item);
    src.onended = () => {
      if (this.sources.delete(item)) this.playedEndedMs += item.duration * 1000;
    };
  }

  isPlaying() {
    return this.sources.size > 0;
  }

  playedMs() {
    const now = this.ctx.currentTime;
    let ms = this.playedEndedMs;
    for (const s of this.sources) ms += Math.max(0, Math.min(now - s.start, s.duration)) * 1000;
    return ms;
  }

  remainingPlaybackS() {
    return this.ctx && this.isPlaying() ? Math.max(0, this.cursor - this.ctx.currentTime) : 0;
  }

  stopPlayback() {
    for (const s of this.sources) {
      s.src.onended = null;
      try {
        s.src.stop();
      } catch {
        // already stopped
      }
    }
    this.sources.clear();
    this.cursor = 0;
  }

  // ------------------------------------------------------------- events

  onEvent(event) {
    switch (event.type) {
      case "state":
        setStatus(event.state);
        break;
      case "transcript_partial":
        ui.partial.textContent = event.text;
        break;
      case "transcript_final":
        ui.partial.textContent = "";
        addUserTurn(event.turn_id, event.text, event.source === "text");
        break;
      case "agent_text":
        addAgentText(event.turn_id, event.text);
        break;
      case "tool_call":
        addToolCall(event.turn_id, event.name, event.arguments);
        break;
      case "tool_result":
        addToolResult(event.turn_id, event.name, event.is_error, event.result);
        break;
      case "metrics":
        recordMetrics(event);
        break;
      case "error":
        showBanner(event.message, "warn", 6000);
        break;
      case "call_ended":
        this.ended = true;
        setStatus("ended");
        showBanner(END_REASONS[event.reason] || "Call ended.", "info", 8000);
        break;
    }
  }

  onClose() {
    if (!this.ended) {
      setStatus("disconnected");
      showBanner("Disconnected from the server. The conversation so far has been saved.", "error");
      this.cleanup();
      return;
    }
    // Let the goodbye finish playing before tearing the audio down.
    this.stopMic();
    if (call === this) call = null;
    setCallControls(false);
    const remaining = this.remainingPlaybackS();
    if (remaining > 0) setTimeout(() => this.cleanup(), Math.min(remaining, 10) * 1000 + 100);
    else this.cleanup();
  }

  stopMic() {
    if (this.mic) this.mic.getTracks().forEach((track) => track.stop());
    if (this.node) this.node.port.onmessage = null;
  }

  cleanup() {
    if (this.closed) return;
    this.closed = true;
    this.stopMic();
    this.stopPlayback();
    if (this.ctx) this.ctx.close();
    showLevel(-100);
    setCallControls(false);
    if (call === this) call = null;
  }
}

// --------------------------------------------------------------- view

let call = null;
const agentBubbles = new Map();
const latencies = [];

async function startCall({ textOnly = false } = {}) {
  if (call) return;
  hideBanner();
  ui.start.disabled = true;
  call = new Call({ textOnly });
  const ok = await call.start();
  if (!ok) {
    call = null;
    ui.start.disabled = false;
    return;
  }
  setCallControls(true, { textOnly });
  if (textOnly) ui.message.focus();
}

ui.start.addEventListener("click", () => startCall());
ui.textOnly.addEventListener("click", () => startCall({ textOnly: true }));

ui.end.addEventListener("click", () => {
  ui.end.disabled = true;
  if (call) call.end();
});

ui.mute.addEventListener("click", () => toggleMute());

ui.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = ui.message.value.trim();
  if (!text || !call) return;
  call.sendText(text);
  ui.message.value = "";
});

document.addEventListener("keydown", (event) => {
  const typing = event.target === ui.message;
  if (event.key === "Escape" && call) {
    event.preventDefault();
    ui.end.click();
  } else if (!typing && (event.key === "m" || event.key === "M") && call && !ui.mute.disabled) {
    toggleMute();
  } else if (!typing && event.key === "/" && call) {
    event.preventDefault();
    ui.message.focus();
  }
});

function toggleMute() {
  if (!call || call.textOnly) return;
  const muted = !call.muted;
  call.setMuted(muted);
  ui.mute.textContent = muted ? "Unmute" : "Mute";
  ui.mute.setAttribute("aria-pressed", String(muted));
  ui.mute.classList.toggle("on", muted);
}

function setCallControls(active, { textOnly = false } = {}) {
  ui.start.disabled = active;
  ui.end.disabled = !active;
  ui.mute.disabled = !active || textOnly;
  ui.message.disabled = !active;
  ui.send.disabled = !active;
  if (!active) {
    ui.mute.textContent = "Mute";
    ui.mute.setAttribute("aria-pressed", "false");
    ui.mute.classList.remove("on");
  }
}

function callerTimezone() {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || null;
  } catch {
    return null;
  }
}

function setStatus(state) {
  ui.status.dataset.state = state;
  ui.statusLabel.textContent = STATUS_LABELS[state] || state;
}

let bannerTimer = null;
function showBanner(text, kind = "info", hideAfterMs = 0, { offerTextOnly = false } = {}) {
  clearTimeout(bannerTimer);
  ui.bannerText.textContent = text;
  ui.banner.dataset.kind = kind;
  ui.textOnly.hidden = !offerTextOnly;
  ui.banner.hidden = false;
  if (hideAfterMs) bannerTimer = setTimeout(hideBanner, hideAfterMs);
}

function hideBanner() {
  ui.banner.hidden = true;
}

function showLevel(db) {
  const pct = Math.max(0, Math.min(100, ((db + 70) / 60) * 100));
  ui.level.style.width = `${pct}%`;
}

function bubble(role, text) {
  ui.empty.hidden = true;
  const el = document.createElement("div");
  el.className = `turn ${role}`;
  const body = document.createElement("p");
  body.className = "text";
  body.textContent = text;
  el.append(body);
  ui.transcript.append(el);
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
  return el;
}

function addUserTurn(turnId, text, typed) {
  const el = bubble("user", text);
  el.dataset.turn = turnId;
  if (typed) el.classList.add("typed");
}

function agentBubble(turnId) {
  let el = agentBubbles.get(turnId);
  if (!el) {
    el = bubble("agent", "");
    el.dataset.turn = turnId;
    agentBubbles.set(turnId, el);
  }
  return el;
}

function addAgentText(turnId, text) {
  const body = agentBubble(turnId).querySelector(".text");
  body.textContent = body.textContent ? `${body.textContent} ${text}` : text;
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
}

function addToolCall(turnId, name, args) {
  const chip = document.createElement("div");
  chip.className = "tool pending";
  chip.dataset.tool = name;
  const argText = Object.entries(args || {})
    .map(([k, v]) => `${k}: ${v}`)
    .join(", ");
  chip.textContent = `${name}(${argText})`;
  agentBubble(turnId).append(chip);
}

function addToolResult(turnId, name, isError, result) {
  const chip = agentBubble(turnId).querySelector(`.tool.pending[data-tool="${name}"]`);
  if (!chip) return;
  chip.classList.remove("pending");
  chip.classList.add(isError ? "failed" : "ok");
  chip.textContent += ` -> ${summarizeResult(name, isError, result)}`;
}

function summarizeResult(name, isError, result) {
  if (isError) return result.error;
  switch (name) {
    case "get_current_time":
      return result.spoken;
    case "get_order_status":
      return result.status;
    case "calculate":
      return String(result.result);
    case "get_weather":
      return `${result.temperature}° ${result.conditions}, ${result.location}`;
    case "end_call":
      return "hanging up after the goodbye";
    default:
      return JSON.stringify(result);
  }
}

function markInterrupted(turnId) {
  const el = agentBubbles.get(turnId);
  if (el) el.classList.add("interrupted");
}

function recordMetrics(m) {
  const stages = [
    ["vad", m.vad_end_ms],
    ["stt", m.stt_final_ms],
    ["llm", m.llm_first_token_ms],
    ["tts", m.tts_first_audio_ms],
  ];
  const measured = stages.filter(([, ms]) => ms != null);
  if (!measured.length) return;
  // Typed turns have no speech stages, so their total is LLM + TTS.
  const total = m.total_ms ?? measured.reduce((sum, [, ms]) => sum + ms, 0);
  if (m.total_ms != null) latencies.push(m.total_ms);
  const scale = Math.max(total, BUDGET_MS) * 1.1;
  ui.bar.querySelectorAll(".seg").forEach((el) => el.remove());
  for (const [name, ms] of measured) {
    const seg = document.createElement("span");
    seg.className = `seg ${name}`;
    seg.style.width = `${(ms / scale) * 100}%`;
    seg.title = `${name.toUpperCase()} ${Math.round(ms)} ms`;
    ui.bar.append(seg);
  }
  ui.bar.querySelector(".budget").style.left = `${(BUDGET_MS / scale) * 100}%`;
  const parts = measured.map(([n, ms]) => `${n.toUpperCase()} ${Math.round(ms)}`);
  let text = `Turn ${m.turn_id}: ${parts.join(" + ")} = ${Math.round(total)} ms`;
  text += m.total_ms == null ? " (typed, no speech stages)." : ".";
  if (latencies.length) {
    const sorted = [...latencies].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    text += ` Median over ${latencies.length} spoken turn${latencies.length === 1 ? "" : "s"}: ${Math.round(median)} ms.`;
  }
  ui.stats.textContent = text;
}

async function showDemoNotice() {
  try {
    const res = await fetch("/healthz");
    if (res.ok && (await res.json()).demo_mode) ui.demo.hidden = false;
  } catch {
    // the notice is informational only
  }
}

showDemoNotice();
