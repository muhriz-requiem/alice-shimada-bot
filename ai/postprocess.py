"""
ai/postprocess.py
==================
Cleans up raw model output before it reaches Discord. Small, pure
functions only -- no I/O, easy to unit test.
"""

from __future__ import annotations

import re

import config

# In-character fallback used whenever there's no real reply to send
# (empty completion, every line filtered as a leak, backend failure).
# Deliberately a plain sentence: this text may also be spoken by TTS, so
# no stammer fragments (spoken as letter names), and no bare-ellipsis
# fallback that could synthesize to empty audio.
FALLBACK_REPLY = config.FALLBACK_REPLY

# Unifies ellipsis forms so the model's optional quiet-moment "..."
# reaches Discord consistently: the single unicode "…" glyph some
# models emit becomes the typed "...", and runs of 4+ dots collapse.
_ELLIPSIS_GLYPH = re.compile(r"\u2026")
_ELLIPSIS_LONG = re.compile(r"\.{4,}")

# Safety net against the character's old canon stammer style: a single
# letter, a hyphen, then a capitalized word ("F-Father", "I-I'm") --
# the fragment is dropped and only the word remains. Requiring the
# capital after the hyphen protects real hyphenated words where the
# first segment happens to be one letter (T-shirt, V-Neck, T-34 is
# safe too because the second segment starts with a digit).
_STAMMER_PATTERN = re.compile(r"(?<![A-Za-z])([A-Za-z])-([A-Z])")

_LEAK_PHRASES = [
    "system prompt",
    "assistant:",
    "analysis:",
    "thinking:",
    "as an ai",
    "i am an ai",
    "language model",
    "could you please provide",
    "how can i assist",
    "how may i assist",
    "let me know how i can help",
    "i'm here to help",
    "your message got cut off",
    "message got cut off",
    "how can i help you today",
]

# Raw completion artifacts that should never reach Discord. Their
# presence means the model's output ran past the intended in-character
# reply -- a leaked special token (e.g. "<|endoftext|>"), or a stray
# citation-style URL Alice has no in-character reason to produce.
# Everything from the earliest match onward is discarded; the text
# before it is reliably the real reply.
_HARD_CUT_PATTERNS = [
    re.compile(r"<\|[^>]{0,40}\|>"),           # <|endoftext|>, <|im_end|>, etc.
    re.compile(r"\bUrl:\s*https?://\S+", re.IGNORECASE),
    re.compile(r"https?://\S+"),
]


# Matches a whole word immediately followed by a repeat of that same
# word (case-insensitive), e.g. "I I", "Just just", "I I'm" (the \b
# after the backreference stops the match right after the second "I",
# so a trailing contraction like "'m" is left intact afterward rather
# than swallowed or duplicated).
_WORD_REPEAT_PATTERN = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)


def collapse_repeats(text: str) -> str:
    """
    Collapses a whole-word repeat ("I I'm", "Just just") down to a
    single word ("I'm", "Just").

    Belt-and-suspenders alongside the system-prompt rule: the model
    sometimes echoes the first word of a phrase before continuing, so
    this guarantees a clean read regardless of what the model outputs.
    The old behavior converted these repeats into canon stammer format
    ("I-I'm") -- Alice no longer stammers, so the repeat just goes.
    """
    if not text:
        return text

    return _WORD_REPEAT_PATTERN.sub(lambda m: m.group(1), text)


def strip_stammer(text: str) -> str:
    """
    Removes any stammer fragment ("F-Father", "H-Hi", "I-I'm"),
    keeping only the word itself ("Father", "Hi", "I'm").

    Belt-and-suspenders alongside the system-prompt ban: the model
    may still emit the old canon format from habit, and anything
    spoken by TTS would read the lone letter as a letter name
    ("eff-Father"). Single-letter segments only, so real hyphenated
    words (T-shirt, sensha-do, Boko-chan) pass through untouched.
    """
    if not text:
        return text

    return _STAMMER_PATTERN.sub(r"\2", text)


def normalize_ellipsis(text: str) -> str:
    """
    Normalizes ellipsis variants to the typed "...": the unicode
    "…" glyph becomes "...", and runs of four or more dots collapse
    to exactly three. Whitespace around the ellipsis is preserved --
    "..." is now allowed in Alice's text as a quiet-moment marker.
    """
    if not text:
        return text

    text = _ELLIPSIS_GLYPH.sub("...", text)
    text = _ELLIPSIS_LONG.sub("...", text)
    return text


def clean_response(text: str, character_name: str = "") -> str:
    """
    Strips whitespace, hard-cuts anything after a leaked special token
    or stray URL, filters out lines containing meta-commentary /
    assistant-speak leaks, strips any stammer fragments that slipped
    through despite the system prompt, and normalizes ellipsis forms
    -- so the model breaking character never reaches Discord as-is.
    """
    if not text:
        return FALLBACK_REPLY

    text = text.strip()

    earliest_cut = len(text)
    for pattern in _HARD_CUT_PATTERNS:
        match = pattern.search(text)
        if match:
            earliest_cut = min(earliest_cut, match.start())
    text = text[:earliest_cut].strip()

    if not text:
        return FALLBACK_REPLY

    leak_markers = list(_LEAK_PHRASES)
    if character_name:
        leak_markers.append(f"we are {character_name.lower()}")

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    clean_lines = [
        ln for ln in lines
        if not any(marker in ln.lower() for marker in leak_markers)
    ]

    if not clean_lines:
        # Every line looked like a leak -- fall back to the original
        # first line rather than returning nothing.
        result = lines[0] if lines else FALLBACK_REPLY
    else:
        result = "\n".join(clean_lines)

    result = collapse_repeats(result)
    result = strip_stammer(result)
    result = normalize_ellipsis(result)
    return result if result else FALLBACK_REPLY
