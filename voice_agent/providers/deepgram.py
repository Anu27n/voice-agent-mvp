"""Deepgram adapters: streaming STT over WebSocket and streaming TTS over HTTP.

STT opens one WebSocket per utterance when the VAD detects speech (connection
setup overlaps with the user still talking; audio is queued meanwhile). At the
end of the utterance it sends ``CloseStream``; Deepgram flushes its last final
results and closes the socket, and the joined finals are the transcript.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator
from urllib.parse import urlencode

import httpx

from ..audio import rechunk_pcm16
from ..config import ConfigError, Settings
from .base import PartialCallback, STTError, STTProvider, STTStream, TTSError, TTSProvider
from .registry import register_stt, register_tts

log = logging.getLogger(__name__)

STT_URL = "wss://api.deepgram.com/v1/listen"
TTS_URL = "https://api.deepgram.com/v1/speak"
TTS_SAMPLE_RATES = {8000, 16000, 24000, 32000, 48000}


def parse_results(message: dict[str, Any]) -> tuple[str, bool] | None:
    """Return (transcript, is_final) for a Deepgram ``Results`` message, else None."""
    if message.get("type") != "Results":
        return None
    alternatives = message.get("channel", {}).get("alternatives") or [{}]
    return (alternatives[0].get("transcript") or "").strip(), bool(message.get("is_final"))


@register_stt("deepgram")
class DeepgramSTT(STTProvider):
    name = "deepgram"

    def __init__(self, settings: Settings, url: str = STT_URL) -> None:
        if not settings.deepgram_api_key:
            raise ConfigError("STT_PROVIDER=deepgram needs DEEPGRAM_API_KEY")
        self._api_key = settings.deepgram_api_key
        self._model = settings.deepgram_stt_model
        self._language = settings.stt_language
        self._url = url

    def stream_url(self, sample_rate: int) -> str:
        params = {
            "model": self._model,
            "language": self._language,
            "encoding": "linear16",
            "sample_rate": sample_rate,
            "channels": 1,
            "interim_results": "true",
            "punctuate": "true",
            "smart_format": "true",
            # Our own VAD decides when the utterance ends.
            "endpointing": "false",
        }
        return f"{self._url}?{urlencode(params)}"

    async def open_stream(self, *, sample_rate: int, on_partial: PartialCallback) -> STTStream:
        from websockets.asyncio.client import connect
        from websockets.exceptions import InvalidStatus, WebSocketException

        try:
            ws = await connect(
                self.stream_url(sample_rate),
                additional_headers={"Authorization": f"Token {self._api_key}"},
                open_timeout=5,
                max_size=2**22,
            )
        except InvalidStatus as exc:
            status = exc.response.status_code
            raise STTError("stt_rejected", f"Deepgram refused the connection (HTTP {status})") from exc
        except (OSError, TimeoutError, WebSocketException) as exc:
            raise STTError("stt_connect_failed", f"could not connect to Deepgram: {exc}") from exc
        return _DeepgramStream(ws, on_partial)


class _DeepgramStream(STTStream):
    def __init__(self, ws: Any, on_partial: PartialCallback) -> None:
        self._ws = ws
        self._on_partial = on_partial
        self._finals: list[str] = []
        self._interim = ""
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        from websockets.exceptions import ConnectionClosedError

        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                parsed = parse_results(json.loads(raw))
                if parsed is None:
                    continue
                transcript, is_final = parsed
                if is_final:
                    if transcript:
                        self._finals.append(transcript)
                    self._interim = ""
                else:
                    self._interim = transcript
                partial = " ".join([*self._finals, self._interim]).strip()
                if partial:
                    try:
                        await self._on_partial(partial)
                    except Exception:
                        log.exception("partial transcript callback failed")
        except ConnectionClosedError as exc:
            raise STTError("stt_connection_lost", f"Deepgram closed the stream: {exc}") from exc

    async def send(self, pcm: bytes) -> None:
        if self._reader.done():
            self._reader.result()  # surfaces the reader's error, if any
            raise STTError("stt_connection_lost", "Deepgram stream is closed")
        from websockets.exceptions import ConnectionClosed

        try:
            await self._ws.send(pcm)
        except ConnectionClosed as exc:
            raise STTError("stt_connection_lost", "Deepgram stream is closed") from exc

    async def finish(self) -> str:
        from websockets.exceptions import ConnectionClosed

        try:
            await self._ws.send(json.dumps({"type": "CloseStream"}))
        except ConnectionClosed:
            pass
        await self._reader
        return " ".join(self._finals).strip()

    async def aclose(self) -> None:
        if not self._reader.done():
            self._reader.cancel()
        await asyncio.gather(self._reader, return_exceptions=True)
        await self._ws.close()


@register_tts("deepgram")
class DeepgramTTS(TTSProvider):
    name = "deepgram"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None, url: str = TTS_URL) -> None:
        if not settings.deepgram_api_key:
            raise ConfigError("TTS_PROVIDER=deepgram needs DEEPGRAM_API_KEY")
        if settings.tts_sample_rate not in TTS_SAMPLE_RATES:
            raise ConfigError(f"Deepgram TTS supports sample rates {sorted(TTS_SAMPLE_RATES)}")
        self.sample_rate = settings.tts_sample_rate
        self._api_key = settings.deepgram_api_key
        self._model = settings.deepgram_tts_model
        self._url = url
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        params = {"model": self._model, "encoding": "linear16", "sample_rate": self.sample_rate, "container": "none"}
        headers = {"Authorization": f"Token {self._api_key}"}
        body = {"text": text}
        try:
            async with self._client.stream("POST", self._url, params=params, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")[:300]
                    raise TTSError(f"tts_http_{resp.status_code}", f"Deepgram TTS error: {body}")
                async for chunk in rechunk_pcm16(resp.aiter_bytes(), self.sample_rate // 20 * 2):
                    yield chunk
        except httpx.HTTPError as exc:
            raise TTSError("tts_request_failed", f"Deepgram TTS request failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()
