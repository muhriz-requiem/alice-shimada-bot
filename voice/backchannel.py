"""
voice/backchannel.py
=====================
Generates brief acknowledgments (backchannels) for natural conversation.
These are short "mm", "uh-huh" sounds that play while the other person
is still speaking, making the conversation feel more natural.
"""

from __future__ import annotations

import asyncio
import random


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
    "yeah",
]

# Very short address-acknowledgments ("Hm?" behavior, like ChatGPT voice
# mode): played the moment Alice decides she was directly addressed,
# while STT -> AI -> TTS works underneath. Kept tiny (2-5 chars, <1s).
ACK_JA = [
    "ん？",
    "あれ？",
    "え？",
]

ACK_EN = [
    "Hm?",
    "Hmm?",
    "Yes?",
]

# Max banked-clip duration at 16kHz: the bank is for *brief* sounds, so
# anything longer means the phrase is wrong for the bank (skip it
# rather than degrading playback start with a long tail).
_MAX_BANKED_SECONDS = 2.5


class BackchannelBank:
    """
    Prefetched bank of spoken acknowledgments in Alice's own timbre.

    Built once when the TTS engine is confirmed ready (readiness
    on-ready hook): each phrase is synthesized, RVC-converted where the
    engine needs it, and cached as (float32 audio, sample_rate,
    wav_bytes). At call time a backchannel is a dict lookup + one
    vc.play instead of a full TTS round-trip, which is what makes the
    acknowledgment effectively instant.
    """

    def __init__(self, size_per_lang: int = 8):
        self.size_per_lang = max(0, size_per_lang)
        # lang -> list[(audio_f32_16k, sample_rate, wav_bytes)]
        self._backchannels: dict[str, list] = {}
        self._acks: dict[str, list] = {}
        self._built = False

    @property
    def ready(self) -> bool:
        return self._built and (
            bool(self._backchannels) or bool(self._acks)
        )

    def _pool(self, lang: str, pool: list[str]) -> list[str]:
        if not pool or self.size_per_lang <= 0:
            return []
        # Distinct phrases first, cycled only if the pool is smaller
        # than the requested bank size.
        full = pool * (self.size_per_lang // len(pool) + 1)
        return full[: self.size_per_lang]

    async def _synthesize(
        self,
        store: dict[str, list],
        lang: str,
        phrase: str,
    ) -> None:
        from voice import rvc, tts

        try:
            wav_bytes, _, engine = await tts.generate_tts(phrase)
            # Normalize to 16k mono float -- the same representation
            # _play_backchannel builds today, so playback behavior is
            # identical whether an entry comes from the bank or live.
            audio = tts.wav_bytes_to_array(wav_bytes, target_sr=16000)
            sr = 16000
            # Apply RVC only for the edge engine (sovits is already
            # Alice) -- same rule as the live backchannel path.
            if engine != "sovits" and rvc.is_available():
                audio = await asyncio.to_thread(rvc.convert, audio, 16000)
                audio = await asyncio.to_thread(rvc.soften_human, audio, 16000)

            if len(audio) / sr > _MAX_BANKED_SECONDS:
                log.debug(f"Backchannel too long for bank: {phrase!r}")
                return

            wav_final = tts.array_to_wav_bytes(audio, sample_rate=sr)
            store.setdefault(lang, []).append((audio, sr, wav_final))
        except Exception as e:  # noqa: BLE001 - one phrase failing must not kill the bank
            log.warning(f"Backchannel bank entry failed ({lang!r} {phrase!r}): {e}")

    async def build(self) -> None:
        """Synthesize the bank. Best-effort: entries that fail to
        synthesize are skipped; the live path covers what's missing."""
        langs = []
        if BACKCHANNELS_JA:
            langs.append("ja")
        if BACKCHANNELS_EN:
            langs.append("en")

        for lang in langs:
            pool = BACKCHANNELS_JA if lang == "ja" else BACKCHANNELS_EN
            ack_pool = ACK_JA if lang == "ja" else ACK_EN

            for phrase in self._pool(lang, pool):
                await self._synthesize(self._backchannels, lang, phrase)
            for phrase in ack_pool:
                await self._synthesize(self._acks, lang, phrase)

        self._built = True
        total = sum(len(v) for v in self._backchannels.values())
        total += sum(len(v) for v in self._acks.values())
        log.info(
            f"Backchannel bank built: {total} clips "
            f"(langs={list(self._backchannels) + list(self._acks)})"
        )

    def _normalize_lang(self, lang: str | None) -> str:
        language = (lang or "en").lower()
        return "ja" if language.startswith("ja") else "en"

    def _pick(self, store: dict[str, list], lang: str | None):
        normalized = self._normalize_lang(lang)
        entries = store.get(normalized) or next(iter(store.values()), None)
        if not entries:
            return None
        audio, sr, wav = random.choice(entries)
        return audio, sr, wav

    def pick_backchannel(self, lang: str | None = None):
        """Random banked backchannel: (audio_f32, sr, wav_bytes) or None."""
        return self._pick(self._backchannels, lang)

    def pick_ack(self, lang: str | None = None):
        """Random banked address-ack ("Hm?"): (audio_f32, sr, wav_bytes) or None."""
        return self._pick(self._acks, lang)

    def stats(self) -> dict:
        return {
            "backchannels": {k: len(v) for k, v in self._backchannels.items()},
            "acks": {k: len(v) for k, v in self._acks.items()},
            "built": self._built,
        }


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
