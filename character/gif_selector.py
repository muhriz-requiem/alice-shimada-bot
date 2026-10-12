"""
character/gif_selector.py
===========================
Picks a reaction GIF based on detected intent/emotion, with a per-user
cooldown so the bot doesn't spam a GIF on every single message. State
is in-memory only (resets on restart) -- same behavior as the original
bot, and low-stakes enough not to need persistence.

Now supports Redis-backed cooldown for horizontal scaling.
"""

from __future__ import annotations

import random
import time

import config
from character.base import Character
from state import (
    check_gif_cooldown,
    check_gif_cooldown_async,
    is_redis_available,
    set_gif_cooldown,
    set_gif_cooldown_async,
)


class GifSelector:
    _redis_available: bool | None = None

    @classmethod
    def _check_redis(cls) -> bool:
        if cls._redis_available is None:
            cls._redis_available = is_redis_available()
        return cls._redis_available

    def __init__(self, character: Character):
        self._character = character
        self._last_shown: dict[str, float] = {}
        self._redis_enabled = self._check_redis()

    def _on_cooldown(self, user_id: str) -> bool:
        if self._redis_enabled:
            return check_gif_cooldown(user_id, config.GIF_COOLDOWN_SECONDS)
        last = self._last_shown.get(user_id, 0.0)
        return (time.monotonic() - last) < config.GIF_COOLDOWN_SECONDS

    async def _on_cooldown_async(self, user_id: str) -> bool:
        if self._redis_enabled:
            return await check_gif_cooldown_async(user_id, config.GIF_COOLDOWN_SECONDS)
        last = self._last_shown.get(user_id, 0.0)
        return (time.monotonic() - last) < config.GIF_COOLDOWN_SECONDS

    def _set_cooldown(self, user_id: str) -> None:
        if self._redis_enabled:
            set_gif_cooldown(user_id)
        else:
            self._last_shown[user_id] = time.monotonic()

    async def _set_cooldown_async(self, user_id: str) -> None:
        if self._redis_enabled:
            await set_gif_cooldown_async(user_id)
        else:
            self._last_shown[user_id] = time.monotonic()

    def pick_for_intent(self, intent_name: str) -> str | None:
        """
        Direct lookup for an explicitly detected intent (e.g.
        "goodnight", "agree_special", "agree"). Bypasses the cooldown
        entirely and doesn't touch cooldown state -- an explicit intent
        is a strong, deliberate signal from the user, so it should
        always get a reaction, matching the original bot's behavior.
        """
        pool = self._character.gifs.get(intent_name)
        return random.choice(pool) if pool else None

    def pick_for_emotion(self, user_id: str, emotion: str | None) -> str | None:
        """
        Cooldown-gated fallback based on the character's ambient
        emotional state, used only when no explicit intent fired.
        """
        if not emotion or emotion not in self._character.gifs:
            return None

        if self._on_cooldown(user_id):
            return None

        pool = self._character.gifs[emotion]
        if not pool:
            return None

        self._set_cooldown(user_id)
        return random.choice(pool)

    async def pick_for_emotion_async(self, user_id: str, emotion: str | None) -> str | None:
        """
        Async cooldown-gated fallback based on the character's ambient
        emotional state, used only when no explicit intent fired.
        """
        if not emotion or emotion not in self._character.gifs:
            return None

        if await self._on_cooldown_async(user_id):
            return None

        pool = self._character.gifs[emotion]
        if not pool:
            return None

        await self._set_cooldown_async(user_id)
        return random.choice(pool)
