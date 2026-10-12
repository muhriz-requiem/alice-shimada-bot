"""
voice/state.py
===============
Voice connection state management. Tracks which guilds are connected,
current voice channels, voice session metadata, and turn states.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

import discord

from bot.logging_setup import get_logger

log = get_logger("voice.state")


class TurnState(Enum):
    """Current turn state of the voice conversation."""
    IDLE = "idle"                    # No one is speaking
    SINGLE_SPEAKER = "single"       # One person talking
    MULTIPLE_SPEAKERS = "multi"     # Multiple people talking
    PAUSE = "pause"                  # Brief silence after speech
    LULL = "lull"                    # Extended silence, open floor
    BOT_RESPONDING = "responding"   # Alice is speaking
    BARGE_IN = "barge_in"           # User interrupted Alice


@dataclass
class SpeakerInfo:
    """Information about an active speaker."""
    user_id: int
    first_spoke_at: float = field(default_factory=time.time)
    last_spoke_at: float = 0.0
    utterance_count: int = 0

    @property
    def is_active(self) -> bool:
        """Whether the speaker has spoken recently."""
        return time.time() - self.last_spoke_at < 2.0

    @property
    def silence_duration(self) -> float:
        """Seconds since last speech."""
        if self.last_spoke_at == 0:
            return float("inf")
        return time.time() - self.last_spoke_at


@dataclass
class VoiceSession:
    """Represents an active voice session in a guild."""
    guild_id: int
    channel_id: int
    connected_at: float = field(default_factory=time.time)
    last_spoke_at: float = 0.0
    total_speech_seconds: float = 0.0
    messages_processed: int = 0
    turn_state: TurnState = TurnState.IDLE
    turn_state_changed_at: float = field(default_factory=time.time)
    speakers: dict[int, SpeakerInfo] = field(default_factory=dict)
    bot_playing: bool = False

    @property
    def duration(self) -> float:
        return time.time() - self.connected_at

    @property
    def is_active(self) -> bool:
        return time.time() - self.last_spoke_at < 300  # 5 min timeout

    @property
    def turn_duration(self) -> float:
        """How long the current turn state has been active."""
        return time.time() - self.turn_state_changed_at

    @property
    def active_speakers(self) -> list[int]:
        """List of currently active speaker IDs."""
        return [uid for uid, info in self.speakers.items() if info.is_active]

    @property
    def active_speaker_count(self) -> int:
        """Number of currently active speakers."""
        return len(self.active_speakers)


class VoiceStateManager:
    """Manages voice connection state across all guilds."""

    def __init__(self):
        self._sessions: dict[int, VoiceSession] = {}
        self._voice_clients: dict[int, Union[discord.VoiceClient, object]] = {}

    def connect(
        self,
        guild_id: int,
        channel_id: int,
        voice_client: Union[discord.VoiceClient, object],
    ) -> VoiceSession:
        """Record a new voice connection."""
        session = VoiceSession(guild_id=guild_id, channel_id=channel_id)
        self._sessions[guild_id] = session
        self._voice_clients[guild_id] = voice_client
        log.info(f"Voice connected: guild={guild_id}, channel={channel_id}")
        return session

    def disconnect(self, guild_id: int) -> Optional[VoiceSession]:
        """Record a voice disconnection."""
        session = self._sessions.pop(guild_id, None)
        self._voice_clients.pop(guild_id, None)
        if session:
            log.info(
                f"Voice disconnected: guild={guild_id}, "
                f"duration={session.duration:.0f}s, "
                f"messages={session.messages_processed}"
            )
        return session

    def get_session(self, guild_id: int) -> Optional[VoiceSession]:
        """Get the active voice session for a guild."""
        return self._sessions.get(guild_id)

    def get_voice_client(self, guild_id: int):
        """Get the voice client for a guild (VoiceClient or VoiceRecvClient)."""
        vc = self._voice_clients.get(guild_id)
        if vc and hasattr(vc, 'is_connected') and vc.is_connected():
            return vc
        return None

    def is_connected(self, guild_id: int) -> bool:
        """Check if the bot is connected to a voice channel in a guild."""
        vc = self._voice_clients.get(guild_id)
        return vc is not None and hasattr(vc, 'is_connected') and vc.is_connected()

    def get_connected_guilds(self) -> list[int]:
        """Get list of guild IDs where the bot is connected to voice."""
        return [
            gid for gid, vc in self._voice_clients.items()
            if hasattr(vc, 'is_connected') and vc.is_connected()
        ]

    def update_spoke(self, guild_id: int) -> None:
        """Update the last-spoke timestamp for a session."""
        session = self._sessions.get(guild_id)
        if session:
            session.last_spoke_at = time.time()

    def increment_messages(self, guild_id: int) -> None:
        """Increment the message counter for a session."""
        session = self._sessions.get(guild_id)
        if session:
            session.messages_processed += 1

    def update_speaker(
        self, guild_id: int, user_id: int, is_bot: bool = False
    ) -> None:
        """Update speaker activity."""
        session = self._sessions.get(guild_id)
        if not session:
            return

        if user_id not in session.speakers:
            session.speakers[user_id] = SpeakerInfo(user_id=user_id)

        speaker = session.speakers[user_id]
        speaker.last_spoke_at = time.time()
        speaker.utterance_count += 1

    def set_turn_state(self, guild_id: int, state: TurnState) -> None:
        """Update the turn state for a guild."""
        session = self._sessions.get(guild_id)
        if session and session.turn_state != state:
            old = session.turn_state
            session.turn_state = state
            session.turn_state_changed_at = time.time()
            log.debug(f"Turn state: {old.value} -> {state.value} (guild={guild_id})")

    def set_bot_playing(self, guild_id: int, playing: bool) -> None:
        """Mark whether the bot is currently playing audio."""
        session = self._sessions.get(guild_id)
        if session:
            session.bot_playing = playing

    def cleanup_stale(self, timeout_seconds: float = 600) -> list[int]:
        """Remove stale sessions where voice client disconnected."""
        stale = []
        for guild_id in list(self._sessions.keys()):
            vc = self._voice_clients.get(guild_id)
            if vc is None or not hasattr(vc, 'is_connected') or not vc.is_connected():
                stale.append(guild_id)
                self._sessions.pop(guild_id, None)
                self._voice_clients.pop(guild_id, None)
                log.info(f"Cleaned up stale voice session: guild={guild_id}")
        return stale


# Global singleton
voice_state = VoiceStateManager()
