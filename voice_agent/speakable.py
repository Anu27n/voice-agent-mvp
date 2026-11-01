"""Make model text safe to read aloud.

The system prompt asks for plain spoken language, but models still slip in
markdown, list markers, emoji or URLs, which TTS reads literally ("asterisk
asterisk..."). Each sentence is cleaned before it is captioned and spoken.
"""

from __future__ import annotations

import re
import unicodedata

_CODE_FENCE = re.compile(r"```[a-zA-Z0-9_-]*")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")  # [label](url) -> label
_URL = re.compile(r"\bhttps?://(?:www\.)?([^\s/]+)[^\s]*")  # https://www.example.com/x -> example.com
_STRONG = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_EMPHASIS = re.compile(r"(?<![\w*])\*(?=\S)([^*]+?)(?<=\S)\*(?![\w*])|(?<!\w)_(?=\S)([^_]+?)(?<=\S)_(?!\w)")
_LINE_MARKUP = re.compile(r"^\s*(?:#{1,6}\s+|>\s+|[-*+]\s+|\d+[.)]\s+)")  # heading, quote, list item
_NUMBER_SIGN = re.compile(r"#(?=\d)")
_TIMES = re.compile(r"(?<=\d)\s*[*×]\s*(?=\d)")
_STRAY = re.compile(r"[*`#|~^<>{}\[\]\\]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+(?=[.,!?;:])")


def speakable(text: str) -> str:
    text = _CODE_FENCE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(r"\1", text)
    text = _STRONG.sub(r"\2", text)
    text = _EMPHASIS.sub(lambda m: m.group(1) or m.group(2), text)
    text = _LINE_MARKUP.sub("", text)
    text = _NUMBER_SIGN.sub("number ", text)
    text = _TIMES.sub(" times ", text)
    text = _STRAY.sub(" ", text)
    text = "".join(ch for ch in text if not _is_pictograph(ch))
    text = re.sub(r"\s+", " ", text)
    return _SPACE_BEFORE_PUNCT.sub("", text).strip()


def _is_pictograph(ch: str) -> bool:
    # Emoji and other symbols TTS can't say, plus emoji modifiers/joiners.
    return unicodedata.category(ch) == "So" or ch in "‍️"
