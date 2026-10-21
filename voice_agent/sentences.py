"""Turn a stream of LLM tokens into speakable sentences.

Each sentence goes to TTS as soon as it is complete, so the agent starts
talking while the rest of the reply is still being generated.
"""

from __future__ import annotations

import re

# Sentence-final punctuation, optional closing quotes/brackets/emphasis, then whitespace.
_BOUNDARY = re.compile(r"[.!?]+[\"'”’)\]*_]*(?=\s)")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "jr", "sr", "approx"}


class SentenceSplitter:
    def __init__(self, max_chars: int = 220) -> None:
        self.max_chars = max_chars
        self._buf = ""

    def push(self, text: str) -> list[str]:
        """Add streamed text; return any sentences it completed."""
        self._buf += text
        out: list[str] = []
        while True:
            sentence = self._take_sentence()
            if sentence is None:
                break
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> list[str]:
        """Return whatever is left (end of the reply or a tool-call boundary)."""
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []

    def _take_sentence(self) -> str | None:
        buf = self._buf
        newline = buf.find("\n")
        for match in _BOUNDARY.finditer(buf):
            end = match.end()
            if newline != -1 and newline < end:
                break
            if self._is_abbreviation(buf[: match.start()]):
                continue
            self._buf = buf[end:].lstrip()
            return buf[:end].strip()
        if newline != -1:
            self._buf = buf[newline + 1 :]
            return buf[:newline].strip()
        if len(buf) > self.max_chars:
            # A run-on reply with no punctuation: break at the last comma or space.
            cut = max(buf.rfind(", ", 0, self.max_chars), buf.rfind(" ", 0, self.max_chars))
            if cut > 0:
                self._buf = buf[cut + 1 :].lstrip()
                return buf[: cut + 1].strip()
        return None

    @staticmethod
    def _is_abbreviation(before: str) -> bool:
        words = before.split()
        if not words:
            return False
        word = words[-1].lower().strip("(\"'")
        # Initials ("J. Smith") and known abbreviations don't end a sentence.
        return word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha() and word not in {"i", "a"})
