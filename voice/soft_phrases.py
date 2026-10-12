"""
voice/soft_phrases.py
=====================
Curated clarification phrases Alice uses when her name is mentioned
but not directly addressed (e.g. "Alice is quiet" -> "Hmm? ... was I
being talked about?"). Short, character-appropriate, non-intrusive.

Static phrase pool (EN + JA) -- no AI round-trip, instant TTS, preserves
response latency. JA pool matches Alice's character (Japanese).
"""

from __future__ import annotations

import random

from bot.logging_setup import get_logger

log = get_logger("voice.soft_phrases")

# Japanese phrases (Alice's character voice)
SOFT_PHRASES_JA = [
    "えっ？……あたしの名前を呼んだ？",
    "ふうん？……あたし、何か言われてた？",
    "あら？……もしかして、あたしのこと？",
    "えへへ？……なんでもないけど、呼んだ？",
    "……えっ？あたし、聞こえてた？",
    "えっ、ほんと？……何？",
]

# English phrases (spoken through Alice's voice)
SOFT_PHRASES_EN = [
    "Hmm? ... did I hear my name?",
    "Eh? ... was I being talked about?",
    "Mmm? ... I thought someone said my name.",
    "Oh? ... did you call me?",
    "Heh? ... is everything okay?",
    "Hmm? ... I almost thought you said something.",
]

# Very brief/tentative clarifications used in multi-speaker clips
SOFT_PHRASES_BRIEF_JA = [
    "えっ…?",
    "ふーん…?",
    "あら?",
    "ん…?",
    "えへ…?",
]

# Very brief/tentative clarifications used in multi-speaker clips
SOFT_PHRASES_BRIEF_EN = [
    "Hmm?",
    "Eh?",
    "Oh?",
    "Mmm?",
    "...?",
]


def pick_soft_phrase(language: str = "ja", tentative: bool = False) -> str:
    """
    Pick a random clarification phrase for the given language.

    Args:
        language: "ja" for Japanese, anything else for English.
        tentative: pick a brief/tentative phrase for multi-speaker clips.

    Returns:
        A phrase string.
    """
    if tentative:
        pool = (
            SOFT_PHRASES_BRIEF_JA if language.startswith("ja") else SOFT_PHRASES_BRIEF_EN
        )
    else:
        pool = SOFT_PHRASES_JA if language.startswith("ja") else SOFT_PHRASES_EN
    choice = random.choice(pool)
    log.debug(f"Picked soft phrase ({language}, tentative={tentative}): {choice!r}")
    return choice
