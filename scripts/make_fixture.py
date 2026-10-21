"""Regenerate tests/fixtures/two_turns.wav.

Two spoken questions separated by silence, with a 60 ms click between them that
the VAD must ignore, over a low noise floor. Speech is synthesised with Windows
SAPI (System.Speech) through PowerShell; on another OS, record any 16 kHz mono
16-bit WAV with the same two questions instead.

    python scripts/make_fixture.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice_agent.audio import read_wav, to_pcm16, write_wav  # noqa: E402

SAMPLE_RATE = 16000
PHRASES = ("What time is it right now?", "Can you check the status of order A1001?")
OUTPUT = ROOT / "tests" / "fixtures" / "two_turns.wav"


def synthesize(text: str, path: Path) -> None:
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({SAMPLE_RATE}, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
        f"$s.SetOutputToWaveFile('{path}', $f); $s.Speak('{text}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)


def main() -> None:
    if sys.platform != "win32":
        sys.exit("This script uses Windows SAPI; record the fixture manually on other systems.")
    rng = np.random.default_rng(7)

    def noise(seconds: float, level: float = 0.0015) -> np.ndarray:
        return rng.normal(0.0, level, int(seconds * SAMPLE_RATE))

    parts = [noise(0.5)]
    for index, phrase in enumerate(PHRASES):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phrase.wav"
            synthesize(phrase, path)
            pcm, rate = read_wav(path)
        if rate != SAMPLE_RATE:
            raise SystemExit(f"expected {SAMPLE_RATE} Hz from SAPI, got {rate}")
        speech = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        parts.append(speech + noise(len(speech) / SAMPLE_RATE))
        if index == 0:
            click = rng.normal(0.0, 0.3, int(0.06 * SAMPLE_RATE))
            parts += [noise(1.5), click, noise(1.2)]
    parts.append(noise(1.0))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_wav(OUTPUT, to_pcm16(np.concatenate(parts)), SAMPLE_RATE)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
