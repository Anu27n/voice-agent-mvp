"""PCM16 helpers shared by the VAD, the providers and the tests."""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import AsyncIterator, Iterator

import numpy as np

SAMPLE_WIDTH = 2  # bytes per PCM16 sample
SILENCE_DBFS = -100.0


def pcm16_duration_ms(num_bytes: int, sample_rate: int) -> float:
    return num_bytes / SAMPLE_WIDTH / sample_rate * 1000.0


def frame_dbfs(frame: bytes) -> float:
    """Root-mean-square level of a PCM16 frame in dB relative to full scale."""
    samples = np.frombuffer(frame, dtype="<i2").astype(np.float32)
    if samples.size == 0:
        return SILENCE_DBFS
    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms <= 0.0:
        return SILENCE_DBFS
    return max(SILENCE_DBFS, 20.0 * math.log10(rms / 32768.0))


def iter_frames(pcm: bytes, sample_rate: int, frame_ms: int) -> Iterator[bytes]:
    """Split PCM into fixed-size frames; a short tail is zero-padded."""
    size = sample_rate * frame_ms // 1000 * SAMPLE_WIDTH
    for start in range(0, len(pcm), size):
        frame = pcm[start : start + size]
        if len(frame) < size:
            frame = frame + b"\x00" * (size - len(frame))
        yield frame


def read_wav(path: str | Path) -> tuple[bytes, int]:
    """Read a mono PCM16 WAV file and return (pcm_bytes, sample_rate)."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != SAMPLE_WIDTH:
            raise ValueError(f"{path}: expected mono 16-bit PCM")
        return wav.readframes(wav.getnframes()), wav.getframerate()


def write_wav(path: str | Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def to_pcm16(samples: np.ndarray) -> bytes:
    """Convert float samples in [-1, 1] to PCM16LE bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


async def rechunk_pcm16(chunks: AsyncIterator[bytes], min_bytes: int) -> AsyncIterator[bytes]:
    """Re-slice a byte stream into sample-aligned chunks of at least ``min_bytes``.

    HTTP streams split audio at arbitrary byte boundaries; the client needs whole
    samples, and fewer, larger WebSocket messages are cheaper to schedule.
    """
    min_bytes = max(SAMPLE_WIDTH, min_bytes - min_bytes % SAMPLE_WIDTH)
    buf = bytearray()
    async for chunk in chunks:
        buf.extend(chunk)
        if len(buf) >= min_bytes:
            cut = len(buf) - len(buf) % SAMPLE_WIDTH
            yield bytes(buf[:cut])
            del buf[:cut]
    cut = len(buf) - len(buf) % SAMPLE_WIDTH
    if cut:
        yield bytes(buf[:cut])
