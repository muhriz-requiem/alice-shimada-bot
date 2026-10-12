"""
voice/streaming_source.py
==========================
Custom discord.AudioSource that accepts PCM chunks while playing.
Enables sentence-by-sentence TTS streaming — first audio plays
while remaining sentences are still being synthesized.
"""

from __future__ import annotations

import asyncio
import collections
import io
import struct
import threading
import time
from typing import Optional

import numpy as np

from bot.logging_setup import get_logger

log = get_logger("voice.streaming_source")

# Discord expects 20ms frames at 48kHz stereo
FRAME_SIZE = 960  # 48000 * 0.02
SAMPLE_RATE = 48000
CHANNELS = 2
BYTES_PER_FRAME = FRAME_SIZE * CHANNELS * 2  # 16-bit stereo


import discord

class StreamingAudioSource(discord.AudioSource):
    """
    A discord.AudioSource that accepts PCM float32 mono chunks
    and streams them to Discord as 48kHz stereo 16-bit frames.

    Usage:
        source = StreamingAudioSource()
        vc.play(source, after=cleanup)
        # From async code:
        source.add_chunk(pcm_float32_array)
        source.mark_done()
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._queue: collections.deque[bytes] = collections.deque()
        self._lock = threading.Lock()
        self._done = False
        self._sent_first_frame = False
        self._started_at: float = 0
        self._bytes_played: int = 0
        self._silence_frames_sent: int = 0
        self._max_silence_frames = 50  # ~1 second of silence before stopping

    def is_opus(self) -> bool:
        """We send raw PCM, not Opus (discord.AudioSource API)."""
        return False

    def read(self) -> Optional[bytes]:
        """
        Read the next 20ms frame for Discord.
        Called by discord.py's voice client in a loop.
        """
        with self._lock:
            if self._queue:
                self._silence_frames_sent = 0
                frame = self._queue.popleft()
                self._bytes_played += len(frame)
                if not self._sent_first_frame:
                    self._sent_first_frame = True
                    self._started_at = time.time()
                return frame

            if self._done:
                # Trailing silence so Discord flushes cleanly, then end
                self._silence_frames_sent += 1
                if self._silence_frames_sent > self._max_silence_frames:
                    return b''  # AudioSource end-of-stream
                return b'\x00' * BYTES_PER_FRAME

            # Buffer starvation while TTS still synthesizing — keep alive.
            # First sentence can take several seconds (Edge-TTS + RVC).
            self._silence_frames_sent += 1
            return b'\x00' * BYTES_PER_FRAME

    def add_pcm(self, audio: np.ndarray, source_sr: int = 24000) -> None:
        """
        Add a chunk of PCM float32 mono audio.
        Automatically resamples to 48kHz stereo 16-bit.
        """
        if len(audio) == 0:
            return

        # Resample to 48kHz if needed
        if source_sr != SAMPLE_RATE:
            import scipy.signal
            audio = scipy.signal.resample(
                audio, int(len(audio) * SAMPLE_RATE / source_sr)
            ).astype(np.float32)

        # Clip to prevent distortion
        audio = np.clip(audio, -1.0, 1.0)

        # Convert to int16 stereo (duplicate mono to both channels)
        audio_int16 = (audio * 32767).astype(np.int16)
        stereo = np.column_stack((audio_int16, audio_int16))
        pcm_bytes = stereo.tobytes()

        # Split into 20ms frames (960 samples * 2 channels * 2 bytes = 3840 bytes)
        frame_bytes = BYTES_PER_FRAME
        with self._lock:
            for i in range(0, len(pcm_bytes), frame_bytes):
                frame = pcm_bytes[i:i + frame_bytes]
                if len(frame) < frame_bytes:
                    # Pad last frame with silence
                    frame = frame + b'\x00' * (frame_bytes - len(frame))
                self._queue.append(frame)
                self._silence_frames_sent = 0

    def add_wav_bytes(self, wav_bytes: bytes) -> None:
        """Add WAV bytes (from TTS) to the stream."""
        audio = _wav_bytes_to_float(wav_bytes)
        self.add_pcm(audio, source_sr=24000)

    def add_mp3_bytes(self, mp3_bytes: bytes) -> None:
        """Add MP3 bytes (from Edge-TTS) to the stream."""
        audio = _mp3_to_float(mp3_bytes)
        self.add_pcm(audio, source_sr=24000)

    def mark_done(self) -> None:
        """Signal that no more audio will be added."""
        with self._lock:
            self._done = True

    def abort(self) -> None:
        """Immediately stop — drop all buffered audio."""
        with self._lock:
            self._queue.clear()
            self._done = True

    @property
    def is_done(self) -> bool:
        with self._lock:
            return self._done and len(self._queue) == 0

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def time_to_first_frame(self) -> float:
        if self._started_at == 0:
            return 0.0
        return self._started_at - self._created_at if hasattr(self, '_created_at') else 0.0

    def cleanup(self) -> None:
        """Clean up resources."""
        with self._lock:
            self._queue.clear()
            self._done = True


def _wav_bytes_to_float(wav_bytes: bytes) -> np.ndarray:
    """Convert WAV bytes to float32 numpy array."""
    import wave
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio


def _mp3_to_float(mp3_bytes: bytes) -> np.ndarray:
    """Convert MP3 bytes to float32 numpy array via ffmpeg."""
    import subprocess
    from voice.utils import find_ffmpeg
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    proc = subprocess.run(
        [ffmpeg, "-i", "pipe:0", "-ar", "24000", "-ac", "1", "-f", "wav", "pipe:1"],
        input=mp3_bytes,
        capture_output=True,
        timeout=10,
    )
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg MP3 decode failed")

    return _wav_bytes_to_float(proc.stdout)
