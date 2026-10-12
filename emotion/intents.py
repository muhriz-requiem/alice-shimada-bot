"""
emotion/intents.py
====================
Generic keyword-based intent detection, driven entirely by the active
character's `intents` data (see character/data/<name>/intents.json)
rather than hardcoded phrase lists -- so a different character can
define its own trigger words without touching this file.
"""

from __future__ import annotations


def detect_intent(message: str, intents_config: dict[str, list[str]]) -> dict[str, bool]:
    msg = message.lower()
    return {
        intent_name: any(kw.lower() in msg for kw in keywords)
        for intent_name, keywords in intents_config.items()
    }
