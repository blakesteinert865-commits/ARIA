"""Small final-response cleaner for Aria's local language model.

Keep finished sentences when the model ends in the middle of another sentence.
This is output formatting only; it cannot fix factual errors or add missing facts.
No effect on replies produced by Crutch/Hassil deterministic tools.
"""
from __future__ import annotations

import re

MAX_RESPONSE_CHARS = 512

# Sentence-final marks, optionally followed by closing quotation/bracket marks.
_FINAL = re.compile(r"[.!?。！？][\"'”’)]*\s*$")
_BOUNDARY = re.compile(r"[.!?。！？][\"'”’)]*(?=\s|$)")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "etc", "vs", "fig",
    "no", "e.g", "i.e", "u.s", "u.k", "a.m", "p.m",
}


def _likely_abbreviation(text: str, dot_index: int) -> bool:
    """Avoid treating common abbreviations and initials as sentence endings."""
    prefix = text[:dot_index + 1]
    match = re.search(r"([A-Za-z](?:\.[A-Za-z])+|[A-Za-z]+)\.$", prefix)
    if match:
        word = match.group(1).lower()
        if word in _ABBREVIATIONS or len(word) == 1:
            return True
    # Numbered sections like "1. The nucleus" are not sentence boundaries.
    if re.search(r"(?:^|\s)\d\.$", prefix):
        return True
    return False

def trim_unfinished_sentence(text: str) -> str:
    """Discard an incomplete last clause while preserving finished sentences.

    A response ending in terminal punctuation is left alone. If there are no
    reliable complete-sentence boundaries, preserve the original reply instead
    of deleting everything. Only call this on model output, not tool answers.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # 1. New creative requirement: Tight character slice right at the 512-character mark
    if len(text) > MAX_RESPONSE_CHARS:
        text = text[:MAX_RESPONSE_CHARS]

    stripped = text.rstrip()
    if _FINAL.search(stripped):
        return stripped

    last_end = None
    for match in _BOUNDARY.finditer(stripped):
        if stripped[match.start()] == "." and _likely_abbreviation(stripped, match.start()):
            continue
        last_end = match.end()

    return stripped[:last_end].rstrip() if last_end is not None else stripped

