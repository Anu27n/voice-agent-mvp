"""Runtime configuration.

Everything is read from environment variables (optionally via a ``.env`` file in
the working directory). ``Settings`` is a frozen dataclass so tests can build one
directly with keyword overrides instead of touching the environment.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, fields
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
You can look up the current time in any time zone and the status of a customer's order."""


class ConfigError(Exception):
    """Raised when the configuration is missing something a provider needs."""


@dataclass(frozen=True)
class Settings:
    # Which adapter to use for each stage. Built-in names: see providers/registry.py.
    # A dotted path ("package.module:ClassName") loads a custom adapter.
    stt_provider: str = "mock"
    llm_provider: str = "mock"
    tts_provider: str = "mock"

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
    tool_timeout_s: float = 5.0
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

    # Storage and server.
    db_path: str = "data/voice_agent.db"
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        if not self.prompt_version:
            digest = hashlib.sha256(self.system_prompt.encode()).hexdigest()[:8]
            object.__setattr__(self, "prompt_version", f"sha-{digest}")

    @property
    def frame_bytes(self) -> int:
        return self.input_sample_rate * self.frame_ms // 1000 * 2

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
        return {
            "providers": {"stt": self.stt_provider, "llm": self.llm_provider, "tts": self.tts_provider},
            "llm_model": self.llm_model if self.llm_provider == "anthropic" else None,
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
