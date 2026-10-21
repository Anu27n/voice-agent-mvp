"""Where each spoken sentence sits in a reply's audio, so a barge-in can work out
what the caller actually heard."""

from __future__ import annotations

from dataclasses import dataclass, field

# Typical speaking rate, used to size a sentence whose audio is still arriving.
EST_MS_PER_CHAR = 65.0


@dataclass
class SpokenSegment:
    text: str
    round_index: int  # which LLM round produced it (tool calls split a reply into rounds)
    start_ms: float  # offset of its first sample in the reply's audio
    duration_ms: float = 0.0
    complete: bool = False  # all of its audio has been synthesized

    @property
    def expected_ms(self) -> float:
        if self.complete:
            return self.duration_ms
        return max(self.duration_ms, len(self.text) * EST_MS_PER_CHAR)


@dataclass
class SpeechTimeline:
    segments: list[SpokenSegment] = field(default_factory=list)
    total_ms: float = 0.0

    def begin(self, text: str, round_index: int = 0) -> SpokenSegment:
        segment = SpokenSegment(text, round_index, self.total_ms)
        self.segments.append(segment)
        return segment

    def add_audio(self, segment: SpokenSegment, ms: float) -> None:
        segment.duration_ms += ms
        self.total_ms += ms

    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments if s.duration_ms > 0)

    def spoken_text(self, heard_ms: float, rounds: set[int] | None = None) -> str:
        """Text heard in the first ``heard_ms`` of audio. A sentence cut off part-way
        keeps the proportional share of its words, marked with an ellipsis."""
        parts: list[str] = []
        for segment in self.segments:
            if rounds is not None and segment.round_index not in rounds:
                continue
            if segment.duration_ms <= 0 or segment.start_ms >= heard_ms:
                continue
            fraction = (heard_ms - segment.start_ms) / segment.expected_ms
            if fraction >= 0.999:
                parts.append(segment.text)
                continue
            words = segment.text.split()
            keep = int(len(words) * fraction)
            if keep:
                parts.append(" ".join(words[:keep]) + "...")
        return " ".join(parts)
