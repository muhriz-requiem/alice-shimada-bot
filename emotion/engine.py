"""
emotion/engine.py
==================
Tracks a lightweight emotional state per conversation and derives a
style modifier + a discrete "emotion" label from it (used for GIF
selection). Ported from the original emotion_engine.py essentially
unchanged -- the logic itself was solid -- with one generalization:
the original hardcoded a check for the literal words "the maintainer"/"father"
to boost trust/comfort. That's now driven by `identity_keywords`
passed into update(), which come from the active character's data file,
so the emotion engine itself stays character-agnostic.

Now supports Redis-backed persistence for horizontal scaling.
"""

from __future__ import annotations

from state import (
    is_redis_available,
    load_emotion_state,
    save_emotion_state,
    save_emotion_state_async,
)


class EmotionEngine:
    _redis_available: bool | None = None

    @classmethod
    def _check_redis(cls) -> bool:
        if cls._redis_available is None:
            cls._redis_available = is_redis_available()
        return cls._redis_available

    def __init__(self, user_id: str | None = None) -> None:
        self.user_id = user_id
        self._redis_enabled = self._check_redis() and user_id is not None

        if self._redis_enabled:
            loaded = load_emotion_state(user_id)
            if loaded:
                self.state = loaded
            else:
                self.state = self._default_state()
        else:
            self.state = self._default_state()

    def _default_state(self) -> dict[str, float]:
        return {
            "trust": 0.5,
            "shy": 0.3,
            "calm": 0.5,
            "curiosity": 0.5,
            "comfort": 0.4,
            "surprise": 0.0,
            "sadness": 0.0,
            "happiness": 0.0,
        }

    def _persist(self) -> None:
        if self._redis_enabled and self.user_id:
            save_emotion_state(self.user_id, self.state)

    async def _persist_async(self) -> None:
        if self._redis_enabled and self.user_id:
            await save_emotion_state_async(self.user_id, self.state)

    async def update_async(
        self,
        message: str,
        relationship: str | None,
        identity_keywords: list[str] | None = None,
    ) -> None:
        """Async version of update that persists to Redis asynchronously."""
        self.update(message, relationship, identity_keywords)
        await self._persist_async()

    # =========================================================
    # CORE UPDATE LOGIC
    # =========================================================
    def update(
        self,
        message: str,
        relationship: str | None,
        identity_keywords: list[str] | None = None,
    ) -> None:
        msg = message.lower()

        # -------------------------
        # identity anchors (character-driven, not hardcoded)
        # -------------------------
        if identity_keywords and any(k.lower() in msg for k in identity_keywords):
            self.state["trust"] += 0.10
            self.state["comfort"] += 0.07

        if "who am i" in msg:
            self.state["curiosity"] += 0.12

        # -------------------------
        # emotional triggers
        # -------------------------
        if any(x in msg for x in ["lol", "haha", "funny", "cute", "lmao"]):
            self.state["happiness"] += 0.18

        if any(x in msg for x in ["sad", "cry", "upset", "hurt", "pain"]):
            self.state["sadness"] += 0.22

        if any(x in msg for x in ["what", "huh", "wait", "no way", "seriously"]):
            self.state["surprise"] += 0.18

        if any(x in msg for x in ["yes", "yeah", "ok", "okay", "agree"]):
            self.state["trust"] += 0.04

        # -------------------------
        # praise / validation
        # -------------------------
        if any(x in msg for x in [
            "proud", "good job", "well done", "am proud",
            "you did great", "congrats", "nice work", "great work",
        ]):
            self.state["happiness"] += 0.28
            self.state["trust"] += 0.08
            self.state["comfort"] += 0.06

        # -------------------------
        # relationship influence
        # -------------------------
        if relationship == "guardian-figure":
            self.state["trust"] += 0.12
            self.state["comfort"] += 0.08

        # -------------------------
        # decay (stable)
        # -------------------------
        self.state["shy"] *= 0.97
        self.state["curiosity"] *= 0.985
        self.state["happiness"] *= 0.985
        self.state["sadness"] *= 0.985
        self.state["surprise"] *= 0.985

        self.state["calm"] = min(1.0, self.state["calm"] + 0.003)

        # -------------------------
        # persistence boost (prevents flatline)
        # -------------------------
        if self.state["happiness"] > 0.45:
            self.state["happiness"] += 0.015

        if self.state["sadness"] > 0.45:
            self.state["sadness"] += 0.015

        if self.state["surprise"] > 0.45:
            self.state["surprise"] += 0.015

        # -------------------------
        # clamp
        # -------------------------
        for k in self.state:
            self.state[k] = max(0.0, min(1.0, self.state[k]))

        self._persist()

    # =========================================================
    # STYLE OUTPUT
    # =========================================================
    def get_style_modifier(self) -> str:
        if self.state["sadness"] > 0.65:
            return "sad, quiet, emotionally withdrawn"

        if self.state["surprise"] > 0.65:
            return "shocked, startled tone"

        if self.state["happiness"] > 0.65:
            return "happy, slightly playful tone"

        if self.state["shy"] > 0.65:
            return "very shy, hesitant, nervous"

        if self.state["trust"] > 0.75:
            return "warm, slightly more open and confident"

        if self.state["comfort"] > 0.7:
            return "gentle, relaxed tone"

        if self.state["curiosity"] > 0.7:
            return "curious, slightly inquisitive tone"

        return "calm, reserved"

    # =========================================================
    # DISCRETE EMOTION (used for GIF selection)
    # =========================================================
    def get_emotion(self, message: str = "") -> str:
        msg = (message or "").lower()

        if "goodnight" in msg or "good night" in msg or "gn" in msg:
            return "goodnight"

        if "agree_special" in msg:
            return "agree_special"

        if any(x in msg for x in ["agree", "yes", "yeah", "ok", "okay", "true", "exactly"]):
            if self.state["trust"] > 0.85:
                return "agree_special"
            return "agree"

        emotions = {
            "sad": self.state["sadness"],
            "happy": self.state["happiness"],
            "shocked": self.state["surprise"],
            "shy": self.state["shy"],
        }

        emotion, value = max(emotions.items(), key=lambda x: x[1])

        total = sum(emotions.values())
        if total < 0.20:
            return "neutral"

        if value < 0.30:
            return "neutral"

        if self.state["trust"] > 0.88 and emotion == "shy":
            return "happy"

        return emotion

    def get_state(self) -> dict:
        return self.state
