"""Deepgram STT against a local fake Deepgram WebSocket server; Deepgram and
ElevenLabs TTS against mocked HTTP."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from websockets.asyncio.server import serve

from conftest import fast_settings, speech
from voice_agent.audio import rechunk_pcm16
from voice_agent.config import ConfigError
from voice_agent.providers.base import STTError, TTSError
from voice_agent.providers.deepgram import DeepgramSTT, DeepgramTTS, parse_results
from voice_agent.providers.elevenlabs import ElevenLabsTTS


def results(transcript, is_final):
    return json.dumps(
        {"type": "Results", "is_final": is_final, "channel": {"alternatives": [{"transcript": transcript}]}}
    )


@pytest.fixture
async def fake_deepgram():
    seen = {}

    async def handler(ws):
        seen["path"] = ws.request.path
        seen["auth"] = ws.request.headers.get("Authorization")
        audio = 0
        async for message in ws:
            if isinstance(message, bytes):
                audio += len(message)
                if audio >= 6400 and "interim" not in seen:
                    seen["interim"] = True
                    await ws.send(results("what time", False))
                    await ws.send(results("What time", True))
                    await ws.send(results("is", False))
            elif json.loads(message)["type"] == "CloseStream":
                seen["audio_bytes"] = audio
                await ws.send(results("is it?", True))
                await ws.send(json.dumps({"type": "Metadata"}))
                await ws.close()
                return

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/v1/listen", seen


async def test_deepgram_stt_streams_partials_and_returns_the_final_transcript(fake_deepgram):
    url, seen = fake_deepgram
    stt = DeepgramSTT(fast_settings(deepgram_api_key="dg-test"), url=url)
    partials = []

    async def on_partial(text):
        partials.append(text)

    stream = await stt.open_stream(sample_rate=16000, on_partial=on_partial)
    for frame in speech(400):
        await stream.send(frame)
    transcript = await stream.finish()
    await stream.aclose()

    assert transcript == "What time is it?"
    assert partials == ["what time", "What time", "What time is", "What time is it?"]
    assert seen["auth"] == "Token dg-test"
    query = parse_qs(urlparse(seen["path"]).query)
    assert query["encoding"] == ["linear16"] and query["sample_rate"] == ["16000"]
    assert query["interim_results"] == ["true"] and query["endpointing"] == ["false"]
    assert seen["audio_bytes"] == 400 * 32


async def test_deepgram_stt_connection_failure_is_an_stt_error():
    stt = DeepgramSTT(fast_settings(deepgram_api_key="dg-test"), url="ws://127.0.0.1:9/v1/listen")

    async def ignore(_):
        return None

    with pytest.raises(STTError):
        await stt.open_stream(sample_rate=16000, on_partial=ignore)


def test_parse_results_ignores_other_messages():
    assert parse_results({"type": "Metadata"}) is None
    assert parse_results(json.loads(results(" hi ", True))) == ("hi", True)


def tts_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_deepgram_tts_request_and_audio():
    seen = {}
    pcm = b"\x01\x02" * 6000

    def handler(request):
        seen["request"] = request
        return httpx.Response(200, content=pcm)

    tts = DeepgramTTS(fast_settings(deepgram_api_key="dg-test"), client=tts_client(handler))
    audio = b"".join([chunk async for chunk in tts.synthesize("Hello there.")])
    assert audio == pcm
    request = seen["request"]
    assert request.headers["Authorization"] == "Token dg-test"
    assert request.url.params["encoding"] == "linear16"
    assert request.url.params["sample_rate"] == "24000" and request.url.params["container"] == "none"
    assert json.loads(request.content) == {"text": "Hello there."}


async def test_deepgram_tts_http_error():
    tts = DeepgramTTS(
        fast_settings(deepgram_api_key="dg-test"),
        client=tts_client(lambda request: httpx.Response(401, text="bad key")),
    )
    with pytest.raises(TTSError) as info:
        [chunk async for chunk in tts.synthesize("Hi")]
    assert info.value.code == "tts_http_401"


async def test_elevenlabs_tts_request_and_audio():
    seen = {}

    def handler(request):
        seen["request"] = request
        return httpx.Response(200, content=b"\x00\x01" * 3000)

    settings = fast_settings(elevenlabs_api_key="el-test", elevenlabs_voice_id="voice123")
    tts = ElevenLabsTTS(settings, client=tts_client(handler))
    audio = b"".join([chunk async for chunk in tts.synthesize("Hi.")])
    assert len(audio) == 6000
    request = seen["request"]
    assert request.url.path == "/v1/text-to-speech/voice123/stream"
    assert request.url.params["output_format"] == "pcm_24000"
    assert request.headers["xi-api-key"] == "el-test"
    assert json.loads(request.content) == {"text": "Hi.", "model_id": "eleven_flash_v2_5"}


def test_tts_adapters_validate_their_config():
    with pytest.raises(ConfigError):
        DeepgramTTS(fast_settings(deepgram_api_key=None))
    with pytest.raises(ConfigError):
        ElevenLabsTTS(fast_settings(elevenlabs_api_key="k", tts_sample_rate=48000))


async def test_rechunk_keeps_whole_samples():
    async def odd_chunks():
        for piece in (b"\x01", b"\x02\x03", b"\x04\x05\x06", b"\x07"):
            yield piece

    chunks = [c async for c in rechunk_pcm16(odd_chunks(), 4)]
    assert all(len(c) % 2 == 0 for c in chunks)
    assert b"".join(chunks) == b"\x01\x02\x03\x04\x05\x06"  # the dangling half-sample is dropped
