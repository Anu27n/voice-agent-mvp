"""Runtime configuration.

Everything is read from environment variables (optionally via a ``.env`` file in
the working directory). ``Settings`` is a frozen dataclass so tests can build one
directly with keyword overrides instead of touching the environment.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

DEFAULT_SYSTEM_PROMPT = """\
You are a friendly, concise voice assistant on a live phone-style call.
Everything you write is converted to speech, so:
- Reply in one to three short sentences unless the caller asks for more detail.
- Use plain spoken language: no markdown, bullet points, emoji, or URLs.
- Say numbers, dates, and times the way a person would say them out loud.
- When you need a tool, you may say one brief sentence first, such as "Let me check that."
- If a tool returns an error, explain the problem simply and suggest what the caller can do next.
- If the caller's words seem garbled or cut off, ask them to repeat instead of guessing.
- Use your tools for the time, orders, weather and arithmetic instead of guessing.
- When the caller says goodbye or wants to hang up, say a brief goodbye and end the call."""


class ConfigError(Exception):
    """Raised when the configuration is missing something a provider needs."""


@dataclass(frozen=True)
class Settings:
    # Which adapter to use for each stage. Built-in names: see providers/registry.py.
    # A dotted path ("package.module:ClassName") loads a custom adapter. "auto"
    # picks the real provider whose API key is set, and the mock otherwise.
    stt_provider: str = "auto"
    llm_provider: str = "auto"
    tts_provider: str = "auto"

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    prompt_version: str = ""  # derived from the prompt text when left empty

    # Audio format, pinned end to end. Client -> server is PCM16LE mono at
    # input_sample_rate in frame_ms frames; server -> client is PCM16LE mono at
    # tts_sample_rate.
    input_sample_rate: int = 16000
    frame_ms: int = 20
    tts_sample_rate: int = 24000

    # Voice activity detection.
    vad_silence_ms: int = 600  # trailing silence that ends an utterance
    vad_start_ms: int = 100  # sustained voice needed to start an utterance
    vad_min_utterance_ms: int = 250  # shorter voiced segments are dropped as noise
    vad_max_utterance_ms: int = 30000  # force an end if a segment never goes quiet
    vad_threshold_db: float = -45.0  # absolute floor (dBFS) for a frame to count as voice
    vad_margin_db: float = 10.0  # ...and it must also be this far above the noise floor
    vad_preroll_ms: int = 300  # audio kept from before the detected onset
    barge_in_min_ms: int = 200  # speech needed to cancel a reply that is still thinking

    # Call limits.
    silence_timeout_s: float = 120.0
    max_call_duration_s: float = 600.0

    # Per-stage timeouts; a timeout is handled like any other stage error.
    stt_timeout_s: float = 4.0  # from end of speech to final transcript
    llm_timeout_s: float = 10.0  # to first token, and between tokens
    tts_timeout_s: float = 5.0  # to first audio chunk, and between chunks
    tool_timeout_s: float = 8.0  # weather does two HTTP calls, with one retry each
    max_tool_rounds: int = 3

    # Anthropic (LLM).
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "low"
    llm_max_tokens: int = 4096
    llm_fallbacks: str = "default"  # server-side refusal fallback; "off" disables it

    # Deepgram (STT and TTS).
    deepgram_api_key: str | None = None
    deepgram_stt_model: str = "nova-3"
    deepgram_tts_model: str = "aura-2-thalia-en"
    stt_language: str = "en"

    # ElevenLabs (alternative TTS).
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model: str = "eleven_flash_v2_5"

    # Mock providers (local development and tests).
    mock_stt_script: tuple[str, ...] = ()
    mock_stt_latency_ms: int = 120
    mock_llm_latency_ms: int = 300
    mock_tts_latency_ms: int = 120
    mock_tts_ms_per_char: float = 55.0

    # Tools.
    default_timezone: str = ""  # empty -> the server's local time zone
    enable_weather: bool = True  # live weather via Open-Meteo (no API key needed)
    allow_agent_hangup: bool = True  # the agent may end the call when the caller says goodbye

    # Public deployments.
    access_code: str | None = None  # when set, every API call and the socket must present it
    max_active_calls: int = 10  # POST /sessions answers 429 beyond this

    # Storage and server.
    db_path: str = "data/voice_agent.db"
    host: str = "127.0.0.1"
    port: int = 8000
    web_dir: str = ""  # directory of the web client; empty -> ./web next to the package

    def __post_init__(self) -> None:
        if not self.prompt_version:
            digest = hashlib.sha256(self.system_prompt.encode()).hexdigest()[:8]
            object.__setattr__(self, "prompt_version", f"sha-{digest}")

    @property
    def frame_bytes(self) -> int:
        return self.input_sample_rate * self.frame_ms // 1000 * 2

    @property
    def demo_mode(self) -> bool:
        """True when any stage runs on a mock provider."""
        resolved = self.resolve_auto()
        return "mock" in (resolved.stt_provider, resolved.llm_provider, resolved.tts_provider)

    def resolve_auto(self) -> "Settings":
        """Replace "auto" provider names with the provider whose key is configured."""
        stt = self.stt_provider
        llm = self.llm_provider
        tts = self.tts_provider
        if stt == "auto":
            stt = "deepgram" if self.deepgram_api_key else "mock"
        if llm == "auto":
            llm = "anthropic" if self.anthropic_api_key else "mock"
        if tts == "auto":
            tts = "deepgram" if self.deepgram_api_key else ("elevenlabs" if self.elevenlabs_api_key else "mock")
        if (stt, llm, tts) == (self.stt_provider, self.llm_provider, self.tts_provider):
            return self
        return replace(self, stt_provider=stt, llm_provider=llm, tts_provider=tts)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, dotenv: bool = True) -> "Settings":
        if env is None:
            if dotenv:
                from dotenv import load_dotenv

                load_dotenv(Path.cwd() / ".env", override=False)
            env = os.environ

        values: dict[str, Any] = {}
        for f in fields(cls):
            raw = env.get(f.name.upper())
            if raw is None or raw == "":
                continue
            values[f.name] = _coerce(f.type, raw)

        prompt_file = env.get("SYSTEM_PROMPT_FILE")
        if prompt_file and "system_prompt" not in values:
            values["system_prompt"] = Path(prompt_file).read_text(encoding="utf-8").strip()
        return cls(**values)

    def public_config(self) -> dict[str, Any]:
        """The configuration stored with each session (no secrets)."""
        resolved = self.resolve_auto()
        return {
            "providers": {"stt": resolved.stt_provider, "llm": resolved.llm_provider, "tts": resolved.tts_provider},
            "llm_model": self.llm_model if resolved.llm_provider == "anthropic" else None,
            "tools": {"weather": self.enable_weather, "agent_hangup": self.allow_agent_hangup},
            "prompt_version": self.prompt_version,
            "audio": {
                "input_sample_rate": self.input_sample_rate,
                "frame_ms": self.frame_ms,
                "tts_sample_rate": self.tts_sample_rate,
            },
            "vad": {
                "silence_ms": self.vad_silence_ms,
                "start_ms": self.vad_start_ms,
                "min_utterance_ms": self.vad_min_utterance_ms,
                "threshold_db": self.vad_threshold_db,
            },
            "limits": {
                "silence_timeout_s": self.silence_timeout_s,
                "max_call_duration_s": self.max_call_duration_s,
            },
        }


def _coerce(annotation: Any, raw: str) -> Any:
    kind = str(annotation)
    if "tuple" in kind:
        return tuple(part.strip() for part in raw.split("|"))
    if kind.startswith("int"):
        return int(raw)
    if kind.startswith("float"):
        return float(raw)
    if kind.startswith("bool"):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return raw
