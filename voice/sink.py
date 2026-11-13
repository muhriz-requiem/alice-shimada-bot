"""
voice/sink.py
==============
Audio receive sink for Discord voice using discord-ext-voice-recv.
Handles receiving audio from Discord, buffering it per-user, and
detecting speech completion via sample-based VAD.
"""

from __future__ import annotations

import collections
import time
from collections.abc import Callable

import numpy as np
from discord.ext import voice_recv

import config
from bot.logging_setup import get_logger
from voice.vad import SileroVAD

log = get_logger("voice.sink")


class AudioBuffer:
    """Circular buffer for incoming audio chunks from a single user."""

    def __init__(self, max_duration: float = 30.0, sample_rate: int = 48000):
        self.max_samples = int(max_duration * sample_rate)
        self.sample_rate = sample_rate
        self._buffer: collections.deque[np.ndarray] = collections.deque()
        self._total_samples = 0
        self._last_write = time.time()
        self._first_write = 0.0
        # Monotonically increasing count of add() calls since creation.
        # Never reset by clear(): a predictive STT snapshot taken from
        # this buffer is fresh only if the revision is unchanged when
        # the utterance completes. Resetting it here would make a stale
        # match indistinguishable from a fresh one after wraparound.
        self.revision = 0

    def add(self, pcm_data: np.ndarray) -> None:
        """Add PCM float32 samples to the buffer."""
        now = time.time()
        if self._first_write == 0:
            self._first_write = now

        self.revision += 1
        self._buffer.append(pcm_data)
        self._total_samples += len(pcm_data)
        self._last_write = now

        # Trim oldest chunks if over limit
        while self._total_samples > self.max_samples and self._buffer:
            old = self._buffer.popleft()
            self._total_samples -= len(old)

    def get_and_clear(self) -> np.ndarray | None:
        """Get all buffered audio and clear the buffer."""
        if not self._buffer:
            return None

        audio = np.concatenate(self._buffer)
        self._buffer.clear()
        self._total_samples = 0
        self._first_write = 0.0
        return audio

    def get_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return self._total_samples / self.sample_rate

    def get_snapshot(self) -> tuple[np.ndarray | None, int]:
        """Copy the buffered audio without clearing the buffer.

        Used by predictive STT: the decode runs on this frozen copy while
        live speech keeps appending. Returns (audio, revision); revision
        is the caller's freshness token -- a later get_audio()+stage must
        compare it against the buffer's current revision before trusting
        the decode result.
        """
        return (
            np.concatenate(self._buffer) if self._buffer else None,
            self.revision,
        )

    @property
    def silence_duration(self) -> float:
        """Time since last audio was written."""
        return time.time() - self._last_write

    @property
    def has_speech(self) -> bool:
        """Whether the buffer currently contains speech."""
        return self._total_samples > 0

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._total_samples = 0
        self._first_write = 0.0
        self._last_write = 0.0


class AliceVoiceSink(voice_recv.AudioSink):
    """
    Audio sink that receives Discord voice audio and buffers it per-user.

    Integrates with discord-ext-voice-recv's AudioSink API to properly
    receive and decode audio from Discord voice channels.

    Audio flow:
    1. Discord sends Opus packets -> voice_recv decodes to PCM
    2. write() receives PCM float32 samples per user
    3. Audio is buffered per-user until speech completes
    4. Completed utterances are passed to the callback for processing
    """

    def __init__(self, guild_id: int) -> None:
        super().__init__()
        self.guild_id = guild_id

        self.buffers: dict[int, AudioBuffer] = {}
        self._vads: dict[int, SileroVAD] = {}
        self._speaking: set[int] = set()
        self._min_speech = config.VOICE_MIN_SPEECH_DURATION
        self._max_speech = config.VOICE_MAX_SPEECH_DURATION
        self._silence_threshold = config.VOICE_SILENCE_THRESHOLD
        self._write_stats: dict[int, dict] = {}  # user_id -> {total, speech, first}

    def wants_opus(self) -> bool:
        """We want decoded PCM, not raw Opus."""
        return False

    def write(self, user, data: voice_recv.VoiceData) -> None:
        """
        Called by voice_recv for each audio packet received.

        Args:
            user: The User/Member who sent the audio (or None).
            data: VoiceData containing pcm_data (np.ndarray of float32).
        """
        if user is None:
            return

        user_id = user.id
        pcm: np.ndarray | bytes | bytearray = data.pcm  # bytes or np.ndarray

        if pcm is None or len(pcm) == 0:
            return

        # Convert bytes (s16le PCM) to float32 numpy array
        if isinstance(pcm, (bytes, bytearray)):
            pcm = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        # Deinterleave stereo -> mono.
        # Discord Opus always decodes as CHANNELS=2 interleaved LRLR...
        # frombuffer yields a 1D array, so ndim>1 never fires and LRLR
        # would be treated as mono garbage (breaks VAD + Whisper).
        if pcm.ndim == 1 and len(pcm) >= 2 and len(pcm) % 2 == 0:
            pcm = pcm.reshape(-1, 2).mean(axis=1)
        elif pcm.ndim > 1:
            pcm = pcm.mean(axis=1)

        # Track write stats
        st = self._write_stats.setdefault(user_id, {"total": 0, "speech": 0, "logged": False})
        st["total"] += 1
        if not st["logged"]:
            st["logged"] = True
            log.info(
                "sink.write first packet: user=%s, samples=%d, dtype=%s, peak=%.4f",
                user_id, len(pcm), pcm.dtype, float(np.max(np.abs(pcm))),
            )

        # Get or create VAD for this user
        if user_id not in self._vads:
            self._vads[user_id] = SileroVAD(sample_rate=16000)
            log.info("Created SileroVAD for user=%s", user_id)

        vad = self._vads[user_id]

        # Check if this chunk has speech using Silero VAD
        # Silero expects 16kHz, so downsample for VAD check
        if len(pcm) > 0:
            # Simple downsample for VAD: take every 3rd sample (48k -> 16k)
            pcm_16k = pcm[::3][:512]  # Silero expects exactly 512 samples
            has_speech = vad.is_speech(pcm_16k) if len(pcm_16k) > 0 else False
        else:
            has_speech = False

        if has_speech:
            st["speech"] += 1
            if st["speech"] <= 3 or st["speech"] % 100 == 0:
                log.info(
                    "VAD speech #%d user=%s (total pkts=%d)",
                    st["speech"], user_id, st["total"],
                )
            self._speaking.add(user_id)
            # Get or create buffer for this user
            if user_id not in self.buffers:
                self.buffers[user_id] = AudioBuffer(
                    max_duration=self._max_speech,
                    sample_rate=48000,
                )
                log.info("Started audio buffer for user=%s", user_id)

            self.buffers[user_id].add(pcm)
        else:
            if st["total"] <= 3:
                log.info("VAD silence #%d user=%s (total pkts=%d)", st["total"], user_id, st["total"])
            self._speaking.discard(user_id)

    def check_speech_complete(self, user_id: int, silence_threshold: float | None = None) -> bool:
        """
        Check if a user has stopped speaking.
        Uses Silero VAD for accurate speech completion detection.
        Returns True if speech is complete and ready for processing.

        silence_threshold overrides config.VOICE_SILENCE_THRESHOLD for
        this call -- the live-mode monitor passes a shorter threshold
        for private calls (adaptive turn-end) while shared calls keep
        the conservative default.
        """
        if user_id not in self.buffers:
            return False

        buf = self.buffers[user_id]
        threshold = (
            self._silence_threshold if silence_threshold is None
            else silence_threshold
        )

        # Check if we have any audio
        if buf.get_duration() < self._min_speech:
            return False

        # Check VAD state — speech complete when silence exceeds threshold
        vad = self._vads.get(user_id)
        if vad and not vad.is_speaking:
            if vad.silence_duration >= threshold:
                return True

        # Fallback: buffer-based silence detection
        if buf.silence_duration >= threshold:
            return True

        # Check if we've hit max duration
        return buf.get_duration() >= self._max_speech

    def observation(self, user_id: int) -> dict | None:
        """Live-mode predictive STT candidate check.

        Returns {silence_duration, buffer_duration, revision, min_speech} for a
        user with buffered speech, or None if there is no buffer. Called by the
        pipeline's monitor loop to decide whether to start a predictive decode
        snapshot.
        """
        buf = self.buffers.get(user_id)
        if buf is None or buf.get_duration() < self._min_speech:
            return None
        vad = self._vads.get(user_id)
        silence = (
            vad.silence_duration if vad and not vad.is_speaking
            else buf.silence_duration
        )
        return {
            "silence_duration": silence,
            "buffer_duration": buf.get_duration(),
            "revision": buf.revision,
        }

    def snapshot_audio(self, user_id: int) -> tuple[np.ndarray, int] | None:
        """Buffer copy for a predictive decode: (audio, revision)."""
        buf = self.buffers.get(user_id)
        if buf is None:
            return None
        audio, revision = buf.get_snapshot()
        if audio is None:
            return None
        return audio, revision

    def get_revision(self, user_id: int) -> int | None:
        """Current buffer revision for a user (None without a buffer)."""
        buf = self.buffers.get(user_id)
        return buf.revision if buf else None

    def get_audio(self, user_id: int) -> np.ndarray | None:
        """Get buffered audio for a user and clear the buffer."""
        if user_id not in self.buffers:
            return None
        return self.buffers[user_id].get_and_clear()

    def clear_user(self, user_id: int) -> None:
        """Clear buffer for a specific user and remove from buffers dict."""
        self._speaking.discard(user_id)
        if user_id in self.buffers:
            self.buffers[user_id].clear()
            del self.buffers[user_id]
        if user_id in self._vads:
            self._vads[user_id].reset()

    def cleanup(self) -> None:
        """Clean up all buffers when sink is closed."""
        self.buffers.clear()
        self._vads.clear()
        self._speaking.clear()
        log.info(f"Sink cleanup for guild {self.guild_id}")


class SimpleSink(voice_recv.AudioSink):
    """
    Simple callback-based sink for quick testing.
    Just calls the callback with (user_id, pcm_data) for each packet.
    """

    def __init__(self, callback: Callable[[int, np.ndarray], None]):
        super().__init__()
        self.callback = callback

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data: voice_recv.VoiceData) -> None:
        if user is None or data.pcm is None:
            return
        pcm = data.pcm
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        self.callback(user.id, pcm)

    def cleanup(self) -> None:
        pass
