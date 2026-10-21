"""ElevenLabs streaming TTS. A drop-in alternative to Deepgram TTS: switching is
``TTS_PROVIDER=elevenlabs`` plus ``ELEVENLABS_API_KEY``, nothing else."""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from ..audio import rechunk_pcm16
from ..config import ConfigError, Settings
from .base import TTSError, TTSProvider
from .registry import register_tts

TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
PCM_SAMPLE_RATES = {16000, 22050, 24000, 44100}


@register_tts("elevenlabs")
class ElevenLabsTTS(TTSProvider):
    name = "elevenlabs"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None, url: str = TTS_URL) -> None:
        if not settings.elevenlabs_api_key:
            raise ConfigError("TTS_PROVIDER=elevenlabs needs ELEVENLABS_API_KEY")
        if settings.tts_sample_rate not in PCM_SAMPLE_RATES:
            raise ConfigError(f"ElevenLabs PCM output supports sample rates {sorted(PCM_SAMPLE_RATES)}")
        self.sample_rate = settings.tts_sample_rate
        self._api_key = settings.elevenlabs_api_key
        self._voice_id = settings.elevenlabs_voice_id
        self._model = settings.elevenlabs_model
        self._url = url
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        url = self._url.format(voice_id=self._voice_id)
        params = {"output_format": f"pcm_{self.sample_rate}"}
        headers = {"xi-api-key": self._api_key}
        body = {"text": text, "model_id": self._model}
        try:
            async with self._client.stream("POST", url, params=params, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode(errors="replace")[:300]
                    raise TTSError(f"tts_http_{resp.status_code}", f"ElevenLabs TTS error: {detail}")
                async for chunk in rechunk_pcm16(resp.aiter_bytes(), self.sample_rate // 20 * 2):
                    yield chunk
        except httpx.HTTPError as exc:
            raise TTSError("tts_request_failed", f"ElevenLabs TTS request failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()
