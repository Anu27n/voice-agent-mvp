"""Energy-based voice activity detection.

The detector works in *audio time* (frame counts), never wall-clock time, so a
recording streamed faster than real time segments exactly like live audio.

A frame counts as voice when its level is above both an absolute floor and an
adaptive noise floor plus a margin. An utterance starts after ``start_ms`` of
consecutive voiced frames (so clicks and taps never start one) and ends after
``silence_ms`` of consecutive unvoiced frames.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .audio import frame_dbfs


class VADEvent(enum.Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass(frozen=True)
class VADConfig:
    frame_ms: int = 20
    start_ms: int = 100
    silence_ms: int = 600
    max_utterance_ms: int = 30000
    threshold_db: float = -45.0
    margin_db: float = 10.0
    initial_noise_db: float = -70.0


class EnergyVAD:
    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self.noise_floor_db = self.config.initial_noise_db
        self.in_speech = False
        self.last_frame_voiced = False
        self.last_level_db = self.config.initial_noise_db
        self._voiced_run = 0  # consecutive voiced frames while waiting for a start
        self._silence_run = 0  # consecutive unvoiced frames inside an utterance
        self._segment_frames = 0
        self._segment_voiced = 0
        self.last_segment_voiced_ms = 0.0

    @property
    def threshold_db(self) -> float:
        return max(self.config.threshold_db, self.noise_floor_db + self.config.margin_db)

    @property
    def speech_ms(self) -> float:
        """Length of the current segment so far, including its onset."""
        return self._segment_frames * self.config.frame_ms if self.in_speech else 0.0

    @property
    def voiced_ms(self) -> float:
        return self._segment_voiced * self.config.frame_ms if self.in_speech else 0.0

    def process(self, frame: bytes) -> VADEvent | None:
        cfg = self.config
        level = frame_dbfs(frame)
        self.last_level_db = level
        voiced = level > self.threshold_db
        self.last_frame_voiced = voiced
        self._track_noise(level)

        if not self.in_speech:
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            if self._voiced_run * cfg.frame_ms >= cfg.start_ms:
                self.in_speech = True
                self._segment_frames = self._voiced_run
                self._segment_voiced = self._voiced_run
                self._silence_run = 0
                self._voiced_run = 0
                return VADEvent.SPEECH_START
            return None

        self._segment_frames += 1
        if voiced:
            self._segment_voiced += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        too_long = self._segment_frames * cfg.frame_ms >= cfg.max_utterance_ms
        if self._silence_run * cfg.frame_ms >= cfg.silence_ms or too_long:
            self.last_segment_voiced_ms = self._segment_voiced * cfg.frame_ms
            self.in_speech = False
            self._segment_frames = 0
            self._segment_voiced = 0
            self._silence_run = 0
            return VADEvent.SPEECH_END
        return None

    def _track_noise(self, level: float) -> None:
        # Follow the floor down quickly and up slowly (even more slowly during
        # speech), so a steady noise source is eventually absorbed into the floor
        # while words barely move it.
        if level < self.noise_floor_db:
            alpha = 0.3
        else:
            alpha = 0.002 if self.in_speech else 0.02
        self.noise_floor_db += alpha * (level - self.noise_floor_db)
