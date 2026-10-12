"""
voice/backchannel.py
=====================
Generates brief acknowledgments (backchannels) for natural conversation.
These are short "mm", "uh-huh" sounds that play while the other person
is still speaking, making the conversation feel more natural.
"""

from __future__ import annotations

import random
from typing import Optional

from bot.logging_setup import get_logger
import config

log = get_logger("voice.backchannel")

# Japanese backchannels (for Alice's character)
BACKCHANNELS_JA = [
    "うん",      # "mm"
    "ううん",    # "uh-huh"
    "そう",      # "right"
    "なるほど",  # "I see"
    "へえ",      # "oh"
    "ふうん",    # "hmm"
    "ああ",      # "ah"
    "おお",      # "ohh"
]

# English fallback
BACKCHANNELS_EN = [
    "mm",
    "uh-huh",
    "right",
    "I see",
    "oh",
    "hmm",
    "ah",
]


class BackchannelGenerator:
    """Generates contextual backchannel responses."""

    def __init__(self):
        self._last_backchannel: dict[int, float] = {}  # guild_id -> timestamp
        self._backchannel_count: dict[int, int] = {}

    def should_backchannel(
        self,
        guild_id: int,
        text: str,
        alice_silence: float = 0.0,
    ) -> bool:
        """Determine if a backchannel is appropriate right now."""
        if not config.VOICE_BACKCHANNEL_ENABLED:
            return False

        # Don't backchannel too frequently
        import time
        now = time.time()
        last = self._last_backchannel.get(guild_id, 0)
        if now - last < 3.0:  # At least 3 seconds between backchannels
            return False

        # Don't backchannel if Alice just had a full response
        if alice_silence < 2.0:
            return False

        return True

    def pick_backchannel(self, guild_id: int, language: str = "ja") -> str:
        """Pick a random backchannel appropriate for the context."""
        if language.startswith("ja"):
            pool = BACKCHANNELS_JA
        else:
            pool = BACKCHANNELS_EN

        choice = random.choice(pool)

        # Record usage
        import time
        self._last_backchannel[guild_id] = time.time()
        self._backchannel_count[guild_id] = self._backchannel_count.get(guild_id, 0) + 1

        return choice

    def get_stats(self, guild_id: int) -> dict:
        """Get backchannel statistics."""
        return {
            "count": self._backchannel_count.get(guild_id, 0),
            "last": self._last_backchannel.get(guild_id, 0),
        }

    def cleanup(self, guild_id: int) -> None:
        """Clean up guild data."""
        self._last_backchannel.pop(guild_id, None)
        self._backchannel_count.pop(guild_id, None)
