"""Name -> adapter lookup, driven by STT_PROVIDER / LLM_PROVIDER / TTS_PROVIDER.

Built-in adapters register themselves with the decorators below. A custom
adapter needs no registration at all: set e.g. ``TTS_PROVIDER=my_pkg.tts:MyTTS``
and the class is imported and constructed with the Settings object.
"""

from __future__ import annotations

import importlib
from typing import Callable, TypeVar

from ..config import ConfigError, Settings
from .base import LLMProvider, STTProvider, TTSProvider

T = TypeVar("T")
Factory = Callable[[Settings], T]

_STT: dict[str, Factory[STTProvider]] = {}
_LLM: dict[str, Factory[LLMProvider]] = {}
_TTS: dict[str, Factory[TTSProvider]] = {}

_BUILTIN_MODULES = ("mock", "anthropic_llm", "deepgram", "elevenlabs")


def register_stt(name: str) -> Callable[[Factory[STTProvider]], Factory[STTProvider]]:
    return _register(_STT, name)


def register_llm(name: str) -> Callable[[Factory[LLMProvider]], Factory[LLMProvider]]:
    return _register(_LLM, name)


def register_tts(name: str) -> Callable[[Factory[TTSProvider]], Factory[TTSProvider]]:
    return _register(_TTS, name)


def build_stt(settings: Settings) -> STTProvider:
    settings = settings.resolve_auto()
    return _build("STT", _STT, settings.stt_provider, settings, STTProvider)


def build_llm(settings: Settings) -> LLMProvider:
    settings = settings.resolve_auto()
    return _build("LLM", _LLM, settings.llm_provider, settings, LLMProvider)


def build_tts(settings: Settings) -> TTSProvider:
    settings = settings.resolve_auto()
    return _build("TTS", _TTS, settings.tts_provider, settings, TTSProvider)


def available() -> dict[str, list[str]]:
    _load_builtins()
    return {"stt": sorted(_STT), "llm": sorted(_LLM), "tts": sorted(_TTS)}


def _register(table: dict[str, Factory[T]], name: str) -> Callable[[Factory[T]], Factory[T]]:
    def decorator(factory: Factory[T]) -> Factory[T]:
        table[name] = factory
        return factory

    return decorator


def _load_builtins() -> None:
    for module in _BUILTIN_MODULES:
        importlib.import_module(f"{__package__}.{module}")


def _build(kind: str, table: dict[str, Factory[T]], name: str, settings: Settings, base: type) -> T:
    if ":" in name:
        module_name, _, attr = name.partition(":")
        try:
            factory = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            raise ConfigError(f"cannot load {kind} provider {name!r}: {exc}") from exc
    else:
        _load_builtins()
        if name not in table:
            raise ConfigError(f"unknown {kind} provider {name!r}; built-in options: {', '.join(sorted(table))}")
        factory = table[name]
    provider = factory(settings)
    if not isinstance(provider, base):
        raise ConfigError(f"{kind} provider {name!r} does not implement {base.__name__}")
    return provider
