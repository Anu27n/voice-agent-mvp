// Voice Agent web client: mic capture, playback, captions, status and barge-in.
//
// Protocol (see docs/DESIGN.md): the client streams PCM16 mono frames at the
// rate POST /sessions returns; the server sends JSON events and binary
// agent_audio frames of [uint32 LE turn_id][PCM16 mono at output_sample_rate].

const $ = (id) => document.getElementById(id);
const ui = {
  start: $("start"),
  end: $("end"),
  status: $("status"),
  statusLabel: $("status-label"),
  banner: $("banner"),
  transcript: $("transcript"),
  empty: $("empty"),
  partial: $("partial"),
  level: $("level"),
  bar: $("bar"),
  stats: $("stats"),
};

const BARGE_IN_DB = -38; // mic level that counts as the user talking over the agent...
const BARGE_IN_MS = 160; // ...for this long (keeps stop time well under 300 ms)
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
  silence_timeout: "Call ended after a long silence.",
  max_duration: "Call ended: maximum call length reached.",
  disconnected: "Call ended: connection lost.",
};

class Call {
  constructor() {
    this.sources = new Set(); // scheduled agent audio
    this.cursor = 0; // AudioContext time where the next chunk should start
    this.playingTurn = null;
    this.playedEndedMs = 0;
    this.interrupted = new Set();
    this.loudMs = 0;
    this.ended = false;
    this.closed = false;
  }

  async start() {
    setStatus("connecting");
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
      );
      setStatus("idle");
      return false;
    }

    try {
      const res = await fetch("/sessions", { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.info = await res.json();
    } catch (err) {
      this.stopMic();
      showBanner(`Couldn't start a session: ${err.message}`, "error");
      setStatus("idle");
      return false;
    }

    this.ctx = new AudioContext();
    await this.ctx.resume();
    await this.ctx.audioWorklet.addModule("mic-worklet.js");
    const source = this.ctx.createMediaStreamSource(this.mic);
    this.node = new AudioWorkletNode(this.ctx, "mic-capture", {
      processorOptions: { targetRate: this.info.input_sample_rate, frameMs: this.info.frame_ms },
    });
    this.node.port.onmessage = (event) => this.onMicFrame(event.data);
    const mute = this.ctx.createGain();
    mute.gain.value = 0; // keeps the worklet running without playing the mic back
    source.connect(this.node).connect(mute).connect(this.ctx.destination);

    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const url = `${scheme}://${location.host}/ws/voice?session_id=${encodeURIComponent(this.info.session_id)}`;
    this.ws = new WebSocket(url);
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

  // ---------------------------------------------------------------- mic

  onMicFrame({ pcm, db }) {
    showLevel(db);
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(pcm);
    if (!this.isPlaying()) {
      this.loudMs = 0;
      return;
    }
    const step = this.info.frame_ms;
    this.loudMs = db > BARGE_IN_DB ? this.loudMs + step : Math.max(0, this.loudMs - step);
    if (this.loudMs >= BARGE_IN_MS) this.bargeIn();
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
        addUserTurn(event.turn_id, event.text);
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
        showBanner(END_REASONS[event.reason] || "Call ended.", "info", 6000);
        break;
    }
  }

  onClose() {
    if (!this.ended) {
      setStatus("disconnected");
      showBanner("Disconnected from the server. The conversation so far has been saved.", "error");
    }
    this.cleanup();
  }

  stopMic() {
    if (this.mic) this.mic.getTracks().forEach((track) => track.stop());
  }

  cleanup() {
    if (this.closed) return;
    this.closed = true;
    this.stopMic();
    this.stopPlayback();
    if (this.node) this.node.port.onmessage = null;
    if (this.ctx) this.ctx.close();
    showLevel(-100);
    ui.start.disabled = false;
    ui.end.disabled = true;
    if (call === this) call = null;
  }
}

// --------------------------------------------------------------- view

let call = null;
const agentBubbles = new Map();
const latencies = [];

ui.start.addEventListener("click", async () => {
  if (call) return;
  hideBanner();
  ui.start.disabled = true;
  call = new Call();
  const ok = await call.start();
  if (!ok) {
    call = null;
    ui.start.disabled = false;
    return;
  }
  ui.end.disabled = false;
});

ui.end.addEventListener("click", () => {
  ui.end.disabled = true;
  if (call) call.end();
});

function setStatus(state) {
  ui.status.dataset.state = state;
  ui.statusLabel.textContent = STATUS_LABELS[state] || state;
}

let bannerTimer = null;
function showBanner(text, kind = "info", hideAfterMs = 0) {
  clearTimeout(bannerTimer);
  ui.banner.textContent = text;
  ui.banner.dataset.kind = kind;
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

function addUserTurn(turnId, text) {
  bubble("user", text).dataset.turn = turnId;
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
  const chips = agentBubble(turnId).querySelectorAll(`.tool.pending[data-tool="${name}"]`);
  const chip = chips[0];
  if (!chip) return;
  chip.classList.remove("pending");
  chip.classList.add(isError ? "failed" : "ok");
  const summary = isError ? result.error : result.spoken || result.status || JSON.stringify(result);
  chip.textContent += ` -> ${summary}`;
}

function markInterrupted(turnId) {
  const el = agentBubbles.get(turnId);
  if (el) el.classList.add("interrupted");
}

function recordMetrics(m) {
  if (m.total_ms == null) return;
  latencies.push(m.total_ms);
  const stages = [
    ["vad", m.vad_end_ms],
    ["stt", m.stt_final_ms],
    ["llm", m.llm_first_token_ms],
    ["tts", m.tts_first_audio_ms],
  ];
  const scale = Math.max(m.total_ms, BUDGET_MS) * 1.1;
  ui.bar.querySelectorAll(".seg").forEach((el) => el.remove());
  for (const [name, ms] of stages) {
    if (ms == null) continue;
    const seg = document.createElement("span");
    seg.className = `seg ${name}`;
    seg.style.width = `${(ms / scale) * 100}%`;
    seg.title = `${name.toUpperCase()} ${Math.round(ms)} ms`;
    ui.bar.append(seg);
  }
  ui.bar.querySelector(".budget").style.left = `${(BUDGET_MS / scale) * 100}%`;
  const sorted = [...latencies].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  const parts = stages.filter(([, ms]) => ms != null).map(([n, ms]) => `${n.toUpperCase()} ${Math.round(ms)}`);
  ui.stats.textContent =
    `Turn ${m.turn_id}: ${parts.join(" + ")} = ${Math.round(m.total_ms)} ms. ` +
    `Median over ${latencies.length} turn${latencies.length === 1 ? "" : "s"}: ${Math.round(median)} ms.`;
}
