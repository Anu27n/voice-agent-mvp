"""Configuration parsing and provider selection: swapping a provider is config only."""

import pytest

from conftest import fast_settings
from voice_agent.config import DEFAULT_SYSTEM_PROMPT, ConfigError, Settings
from voice_agent.providers import build_llm, build_stt, build_tts
from voice_agent.providers.base import TTSProvider
from voice_agent.providers.deepgram import DeepgramTTS
from voice_agent.providers.elevenlabs import ElevenLabsTTS
from voice_agent.providers.mock import MockLLM, MockSTT, MockTTS


def test_settings_from_env_coerces_types():
    settings = Settings.from_env(
        {
            "STT_PROVIDER": "deepgram",
            "VAD_SILENCE_MS": "700",
            "VAD_THRESHOLD_DB": "-40.5",
            "MAX_CALL_DURATION_S": "300",
            "MOCK_STT_SCRIPT": "one | two|three",
            "DEEPGRAM_API_KEY": "",  # empty means unset
        }
    )
    assert settings.stt_provider == "deepgram"
    assert settings.vad_silence_ms == 700 and settings.vad_threshold_db == -40.5
    assert settings.max_call_duration_s == 300.0
    assert settings.mock_stt_script == ("one", "two", "three")
    assert settings.deepgram_api_key is None
    assert settings.system_prompt == DEFAULT_SYSTEM_PROMPT


def test_prompt_version_follows_the_prompt(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("You are a pirate.", encoding="utf-8")
    custom = Settings.from_env({"SYSTEM_PROMPT_FILE": str(prompt)})
    assert custom.system_prompt == "You are a pirate."
    assert custom.prompt_version != Settings().prompt_version
    assert Settings.from_env({"PROMPT_VERSION": "v7"}).prompt_version == "v7"


def test_public_config_has_no_secrets():
    config = fast_settings(
        anthropic_api_key="sk-secret", deepgram_api_key="dg-secret", access_code="secret-code"
    ).public_config()
    assert "secret" not in repr(config)
    # "auto" is stored as the provider it resolved to.
    assert config["providers"] == {"stt": "deepgram", "llm": "anthropic", "tts": "deepgram"}
    assert config["llm_model"] == "claude-opus-5-5"


def test_builds_mock_providers_by_default():
    settings = fast_settings()
    assert settings.stt_provider == "auto" and settings.demo_mode
    assert isinstance(build_stt(settings), MockSTT)
    assert isinstance(build_llm(settings), MockLLM)
    assert isinstance(build_tts(settings), MockTTS)


@pytest.mark.parametrize(
    "keys, expected",
    [
        ({}, ("mock", "mock", "mock")),
        ({"anthropic_api_key": "a"}, ("mock", "anthropic", "mock")),
        ({"deepgram_api_key": "d"}, ("deepgram", "mock", "deepgram")),
        ({"elevenlabs_api_key": "e"}, ("mock", "mock", "elevenlabs")),
        ({"deepgram_api_key": "d", "anthropic_api_key": "a"}, ("deepgram", "anthropic", "deepgram")),
    ],
)
def test_auto_picks_the_provider_whose_key_is_set(keys, expected):
    resolved = fast_settings(**keys).resolve_auto()
    assert (resolved.stt_provider, resolved.llm_provider, resolved.tts_provider) == expected
    assert resolved.demo_mode == ("mock" in expected)


def test_explicit_provider_names_are_kept():
    resolved = fast_settings(stt_provider="mock", deepgram_api_key="d").resolve_auto()
    assert resolved.stt_provider == "mock" and resolved.tts_provider == "deepgram"


def test_swapping_tts_is_configuration_only():
    deepgram = build_tts(fast_settings(tts_provider="deepgram", deepgram_api_key="k"))
    eleven = build_tts(fast_settings(tts_provider="elevenlabs", elevenlabs_api_key="k"))
    assert isinstance(deepgram, DeepgramTTS) and isinstance(eleven, ElevenLabsTTS)


class CustomTTS(TTSProvider):
    name = "custom"

    def __init__(self, settings):
        self.sample_rate = settings.tts_sample_rate

    async def synthesize(self, text):
        yield b"\x00\x00"


def test_custom_adapter_by_dotted_path_needs_no_registration():
    tts = build_tts(fast_settings(tts_provider="test_config_registry:CustomTTS"))
    assert isinstance(tts, CustomTTS)


def test_unknown_or_misconfigured_providers_fail_clearly():
    with pytest.raises(ConfigError, match="unknown TTS provider 'nope'"):
        build_tts(fast_settings(tts_provider="nope"))
    with pytest.raises(ConfigError, match="DEEPGRAM_API_KEY"):
        build_stt(fast_settings(stt_provider="deepgram"))
    with pytest.raises(ConfigError, match="cannot load"):
        build_llm(fast_settings(llm_provider="no.such.module:Thing"))
    with pytest.raises(ConfigError, match="does not implement"):
        build_tts(fast_settings(tts_provider="voice_agent.config:Settings"))
