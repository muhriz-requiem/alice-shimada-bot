"""
voice/turn_detector.py
======================
Semantic turn completion detection for voice conversations.
Uses sample-based timing for accurate silence detection and
determines when a speaker's turn is complete.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional

import numpy as np

from bot.logging_setup import get_logger
import config

log = get_logger("voice.turn_detector")


class TurnState(Enum):
    """Current turn state of the conversation."""
    IDLE = "idle"                    # No one is speaking
    SINGLE_SPEAKER = "single"       # One person talking
    MULTIPLE_SPEAKERS = "multi"     # Multiple people talking (overlapping)
    PAUSE = "pause"                  # Silence after speech, evaluating
    LULL = "lull"                    # Extended silence (>2s), conversation open
    BOT_RESPONDING = "responding"   # Bot is generating/playing TTS
    BARGE_IN = "barge_in"           # User interrupted during bot response


class SpeakerTracker:
    """Tracks per-user audio activity with sample-based timing."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self._speakers: dict[int, SpeakerActivity] = {}

    def update(self, user_id: int, pcm_chunk: np.ndarray) -> SpeakerActivity:
        """Update a speaker's activity with new audio data."""
        if user_id not in self._speakers:
            self._speakers[user_id] = SpeakerActivity(user_id, self.sample_rate)

        activity = self._speakers[user_id]
        activity.add_audio(pcm_chunk)
        return activity

    def get_activity(self, user_id: int) -> Optional[SpeakerActivity]:
        """Get a speaker's current activity."""
        return self._speakers.get(user_id)

    def get_speaking_users(self) -> list[int]:
        """Get list of currently speaking user IDs."""
        return [
            uid for uid, act in self._speakers.items()
            if act.is_speaking
        ]

    def get_speaker_count(self) -> int:
        """Number of users currently speaking."""
        return len(self.get_speaking_users())

    def get_last_speaker(self) -> Optional[int]:
        """Get the user_id who spoke most recently."""
        if not self._speakers:
            return None
        return max(self._speakers.keys(), key=lambda uid: self._speakers[uid].last_audio_at)

    def cleanup_stale(self, timeout: float = 60.0) -> None:
        """Remove speakers who haven't spoken recently."""
        now = time.time()
        stale = [
            uid for uid, act in self._speakers.items()
            if now - act.last_audio_at > timeout
        ]
        for uid in stale:
            del self._speakers[uid]


class SpeakerActivity:
    """Tracks a single speaker's audio activity."""

    def __init__(self, user_id: int, sample_rate: int = 48000):
        self.user_id = user_id
        self.sample_rate = sample_rate
        self.last_audio_at: float = 0.0
        self.first_audio_at: float = 0.0
        self._total_samples: int = 0
        self._silence_samples: int = 0
        self._is_silence: bool = True

    def add_audio(self, pcm_chunk) -> None:
        """Add audio data and update activity."""
        import numpy as np

        now = time.time()
        if self.first_audio_at == 0:
            self.first_audio_at = now

        # Calculate RMS energy
        rms = float(np.sqrt(np.mean(pcm_chunk.astype(np.float64) ** 2)))

        # VAD: check if this chunk has speech
        threshold = config.VOICE_VAD_THRESHOLD / 1000.0
        has_speech = rms > threshold

        chunk_samples = len(pcm_chunk)
        self._total_samples += chunk_samples

        if has_speech:
            self.last_audio_at = now
            self._silence_samples = 0
            self._is_silence = False
        else:
            self._silence_samples += chunk_samples
            if self._silence_samples > self.sample_rate * config.VOICE_SILENCE_THRESHOLD:
                self._is_silence = True
            # Don't update last_audio_at for silence chunks

    @property
    def is_speaking(self) -> bool:
        """Whether this speaker is currently producing speech."""
        if self.last_audio_at == 0:
            return False
        return time.time() - self.last_audio_at < config.VOICE_SILENCE_THRESHOLD

    @property
    def silence_duration(self) -> float:
        """Duration of current silence in seconds."""
        if self.last_audio_at == 0:
            return float("inf")
        return time.time() - self.last_audio_at

    @property
    def total_duration(self) -> float:
        """Total speech duration in seconds."""
        return self._total_samples / self.sample_rate

    @property
    def is_long_pause(self) -> bool:
        """Whether silence exceeds the pause threshold."""
        return self.silence_duration >= config.VOICE_TURN_PAUSE_THRESHOLD

    @property
    def is_conversation_lull(self) -> bool:
        """Whether silence exceeds the lull threshold."""
        return self.silence_duration >= config.VOICE_LULL_THRESHOLD



class TurnDetector:
    """
    Detects turn completion in voice conversations.
    
    Uses sample-based timing for accurate detection (30x faster than wall-clock).
    
    Turn states:
    - IDLE: No one has spoken recently
    - SINGLE_SPEAKER: One person talking, turn not complete
    - PAUSE: Speaker stopped, waiting for resume or turn complete
    - LULL: Extended silence, conversation open for anyone
    - BOT_RESPONDING: Alice is speaking
    - BARGE_IN: User interrupted during Alice's response
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.trackers: dict[int, SpeakerTracker] = {}  # guild_id -> tracker
        self._turn_states: dict[int, TurnState] = {}
        self._turn_started: dict[int, float] = {}
        self._bot_playing: dict[int, bool] = {}

    def get_tracker(self, guild_id: int) -> SpeakerTracker:
        """Get or create speaker tracker for a guild."""
        if guild_id not in self.trackers:
            self.trackers[guild_id] = SpeakerTracker(self.sample_rate)
        return self.trackers[guild_id]

    def update(self, guild_id: int, user_id: int, pcm_chunk) -> TurnState:
        """
        Update turn detection with new audio data.
        Returns the current turn state.
        """
        tracker = self.get_tracker(guild_id)
        activity = tracker.update(user_id, pcm_chunk)

        # Determine turn state
        speaking_count = tracker.get_speaker_count()

        if self._bot_playing.get(guild_id, False):
            if speaking_count > 0 and activity.silence_duration < 0.5:
                state = TurnState.BARGE_IN
            else:
                state = TurnState.BOT_RESPONDING
        elif speaking_count == 0:
            # Check lull vs pause vs idle
            if activity.is_conversation_lull:
                state = TurnState.LULL
            elif activity.is_long_pause:
                state = TurnState.PAUSE
            else:
                state = TurnState.IDLE
        elif speaking_count == 1:
            state = TurnState.SINGLE_SPEAKER
        else:
            state = TurnState.MULTIPLE_SPEAKERS

        old_state = self._turn_states.get(guild_id)
        if old_state != state:
            self._turn_states[guild_id] = state
            self._turn_started[guild_id] = time.time()
            log.debug(
                f"Turn state changed: {old_state} -> {state} "
                f"(speakers={speaking_count})"
            )

        return state

    def get_state(self, guild_id: int) -> TurnState:
        """Get current turn state for a guild."""
        return self._turn_states.get(guild_id, TurnState.IDLE)

    def get_turn_duration(self, guild_id: int) -> float:
        """How long the current turn state has been active."""
        started = self._turn_started.get(guild_id, 0)
        if started == 0:
            return 0.0
        return time.time() - started

    def is_turn_complete(self, guild_id: int) -> bool:
        """
        Check if the current speaker's turn is complete.
        Turn is complete when silence exceeds threshold after speech.
        """
        state = self.get_state(guild_id)
        if state in (TurnState.PAUSE, TurnState.LULL):
            return True
        return False

    def mark_bot_playing(self, guild_id: int, playing: bool) -> None:
        """Mark whether the bot is currently playing audio."""
        self._bot_playing[guild_id] = playing
        if playing:
            self._turn_states[guild_id] = TurnState.BOT_RESPONDING
        # When stopping, state will be updated on next audio update

    def set_activity_state(self, guild_id: int, state: TurnState) -> None:
        """
        Sync turn state from an external activity detector (e.g. the Silero-VAD
        based monitor loop). The monitor is the authoritative real-time source,
        so this keeps turn_detector in sync with the shared voice_state.
        """
        old_state = self._turn_states.get(guild_id)
        if old_state != state:
            self._turn_states[guild_id] = state
            self._turn_started[guild_id] = time.time()
            log.debug(f"Turn state synced: {old_state} -> {state}")

    def get_last_speaker(self, guild_id: int) -> Optional[int]:
        """Get the last person who spoke."""
        tracker = self.trackers.get(guild_id)
        if tracker:
            return tracker.get_last_speaker()
        return None

    def get_active_speakers(self, guild_id: int) -> list[int]:
        """Get currently active speakers."""
        tracker = self.trackers.get(guild_id)
        if tracker:
            return tracker.get_speaking_users()
        return []

    def cleanup(self, guild_id: int) -> None:
        """Clean up guild data."""
        self.trackers.pop(guild_id, None)
        self._turn_states.pop(guild_id, None)
        self._turn_started.pop(guild_id, None)
        self._bot_playing.pop(guild_id, None)
