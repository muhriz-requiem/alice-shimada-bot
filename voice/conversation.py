"""
voice/conversation.py
=====================
Tracks conversation history and context for voice interactions.
Maintains a rolling buffer of recent utterances per guild with
speaker identity, timestamps, and transcription data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from bot.logging_setup import get_logger
import config

log = get_logger("voice.conversation")


@dataclass
class Utterance:
    """A single spoken utterance."""
    user_id: int
    text: str
    timestamp: float = field(default_factory=time.time)
    language: str = "unknown"
    is_bot: bool = False
    responded_to: bool = False

    @property
    def age(self) -> float:
        """Seconds since this utterance was made."""
        return time.time() - self.timestamp


@dataclass
class SpeakerState:
    """Tracks the current speaking state of a user."""
    user_id: int
    last_spoke_at: float = 0.0
    total_utterances: int = 0
    total_words: int = 0

    @property
    def silence_duration(self) -> float:
        """Seconds since this user last spoke."""
        return time.time() - self.last_spoke_at if self.last_spoke_at > 0 else float("inf")


class ConversationTracker:
    """
    Tracks conversation context per guild for smart response decisions.
    
    Maintains:
    - Rolling buffer of last N utterances
    - Per-speaker state (who spoke last, how often)
    - Conversation metadata (topic, mood, active speakers)
    """

    def __init__(self, max_utterances: int = 10):
        self.max_utterances = max_utterances
        self._utterances: dict[int, list[Utterance]] = {}  # guild_id -> utterances
        self._speakers: dict[int, dict[int, SpeakerState]] = {}  # guild_id -> user_id -> state
        self._alice_last_spoke: dict[int, float] = {}  # guild_id -> timestamp

    def add_utterance(
        self,
        guild_id: int,
        user_id: int,
        text: str,
        language: str = "unknown",
        is_bot: bool = False,
    ) -> Utterance:
        """Record a new utterance."""
        utterance = Utterance(
            user_id=user_id,
            text=text,
            language=language,
            is_bot=is_bot,
        )

        if guild_id not in self._utterances:
            self._utterances[guild_id] = []
        self._utterances[guild_id].append(utterance)

        # Trim to max
        if len(self._utterances[guild_id]) > self.max_utterances:
            self._utterances[guild_id] = self._utterances[guild_id][-self.max_utterances:]

        # Update speaker state
        if guild_id not in self._speakers:
            self._speakers[guild_id] = {}
        if user_id not in self._speakers[guild_id]:
            self._speakers[guild_id][user_id] = SpeakerState(user_id=user_id)

        speaker = self._speakers[guild_id][user_id]
        speaker.last_spoke_at = utterance.timestamp
        speaker.total_utterances += 1
        speaker.total_words += len(text.split())

        if is_bot:
            self._alice_last_spoke[guild_id] = utterance.timestamp

        return utterance

    def get_recent(
        self, guild_id: int, count: Optional[int] = None
    ) -> list[Utterance]:
        """Get recent utterances for a guild."""
        utterances = self._utterances.get(guild_id, [])
        if count is not None:
            return utterances[-count:]
        return list(utterances)

    def get_last_alice_spoke(self, guild_id: int) -> float:
        """Get timestamp of when Alice last spoke in this guild."""
        return self._alice_last_spoke.get(guild_id, 0.0)

    def get_last_speaker(self, guild_id: int) -> Optional[int]:
        """Get the user_id of the last non-bot speaker."""
        for u in reversed(self._utterances.get(guild_id, [])):
            if not u.is_bot:
                return u.user_id
        return None

    def get_last_text(self, guild_id: int) -> str:
        """Get the text of the last non-bot utterance."""
        for u in reversed(self._utterances.get(guild_id, [])):
            if not u.is_bot:
                return u.text
        return ""

    def get_speaker_count(self, guild_id: int) -> int:
        """Get the number of unique speakers in recent history."""
        speakers = set()
        for u in self._utterances.get(guild_id, []):
            speakers.add(u.user_id)
        return len(speakers)

    def get_active_speakers(self, guild_id: int, within: float = 5.0) -> list[int]:
        """Get speakers who spoke within the last N seconds."""
        now = time.time()
        speakers = []
        for u in reversed(self._utterances.get(guild_id, [])):
            if u.age <= within and not u.is_bot:
                if u.user_id not in speakers:
                    speakers.append(u.user_id)
        return speakers

    def get_speaker_state(self, guild_id: int, user_id: int) -> Optional[SpeakerState]:
        """Get the state of a specific speaker."""
        return self._speakers.get(guild_id, {}).get(user_id)

    def get_context_string(self, guild_id: int, count: int = 5) -> str:
        """Get a formatted context string of recent utterances for the AI."""
        recent = self.get_recent(guild_id, count)
        lines = []
        for u in recent:
            role = "Alice" if u.is_bot else f"User_{u.user_id}"
            lines.append(f"{role}: {u.text}")
        return "\n".join(lines)

    def get_alice_silence_duration(self, guild_id: int) -> float:
        """Seconds since Alice last spoke."""
        last = self.get_last_alice_spoke(guild_id)
        if last == 0:
            return float("inf")
        return time.time() - last

    def get_conversation_gap(self, guild_id: int) -> float:
        """Seconds since the last utterance from anyone."""
        utterances = self._utterances.get(guild_id, [])
        if not utterances:
            return float("inf")
        return time.time() - utterances[-1].timestamp

    def was_directly_addressed(self, guild_id: int, text: str) -> bool:
        """Check if the last utterance directly mentions Alice."""
        keywords = [kw.lower() for kw in config.VOICE_MENTION_KEYWORDS]
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)

    def cleanup(self, guild_id: int) -> None:
        """Clean up data for a guild."""
        self._utterances.pop(guild_id, None)
        self._speakers.pop(guild_id, None)
        self._alice_last_spoke.pop(guild_id, None)
