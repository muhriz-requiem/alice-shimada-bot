"""
ai/router.py
============
A cheap, non-LLM heuristic that decides whether a message is a short
"fast path" message (greetings, one-word replies, etc.) that doesn't
need memory-graph lookups or a large context payload, versus a normal
message that benefits from full context.

The original project routed this decision through a second local model
(phi3:mini). That meant running two models instead of one, which
contradicted the "single model" goal and added latency + a second
failure point for no real benefit -- in practice the only thing the
router's output changed was this same fast/slow split. So it's been
replaced with a plain heuristic here: same behavior, one dependency,
no extra round-trip.
"""

from __future__ import annotations

_FAST_GREETINGS = {
    "hi", "hey", "hello", "yo", "sup", "hiya", "heya",
    "ok", "okay", "k", "kk", "lol", "lmao", "haha",
    "yes", "no", "yeah", "yep", "nah", "nope",
    "thanks", "thank you", "ty", "np",
    "good morning", "good night", "gn", "gm",
    "bye", "goodbye", "cya", "see ya",
}


def is_fast_path(text: str) -> bool:
    """
    True for short, low-context messages where it's safe to skip
    memory-graph retrieval and keep the request small and quick.
    """
    if not text:
        return True

    stripped = text.strip().lower().strip("!?.,~ ")

    if not stripped:
        return True

    if stripped in _FAST_GREETINGS:
        return True

    # Very short messages with no question mark are usually small talk.
    word_count = len(stripped.split())
    if word_count <= 3 and "?" not in text:
        return True

    return False
