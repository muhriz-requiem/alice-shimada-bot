"""
voice/mixer.py
===============
Audio mixer for overlapping voice sounds.
Handles playing backchannels while another speaker is active,
and supports barge-in by stopping current playback.
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import Optional

import numpy as np

from bot.logging_setup import get_logger
import config
from voice.state import voice_state

log = get_logger("voice.mixer")


class VoiceMixer:
    """
    Manages audio playback with support for:
    - Normal TTS playback
    - Backchannel overlapping (brief ack while others speak)
    - Barge-in (stop current playback when user interrupts)
    """

    def __init__(self):
        self._playing: dict[int, bool] = {}  # guild_id -> is_playing
        self._current_task: dict[int, asyncio.Task] = {}
        self._play_start: dict[int, float] = {}
        self._total_play_time: dict[int, float] = {}

    async def play_tts(
        self,
        guild_id: int,
        audio_data: bytes,
        interruptible: bool = True,
    ) -> bool:
        """
        Play TTS audio to a voice channel.
        
        Args:
            guild_id: Guild ID
            audio_data: WAV bytes to play
            interruptible: Whether this can be barge-in'd
            
        Returns:
            True if playback completed, False if interrupted
        """
        # Stop any current playback
        if interruptible:
            await self.stop(guild_id)

        self._playing[guild_id] = True
        self._play_start[guild_id] = time.time()

        try:
            import discord
            from voice.utils import find_ffmpeg

            vc = voice_state.get_voice_client(guild_id)
            if vc is None:
                self._playing[guild_id] = False
                return False

            audio_file = io.BytesIO(audio_data)
            done = asyncio.Event()

            def after_play(error):
                if error:
                    log.error(f"Playback error: {error}")
                done.set()

            vc.play(
                discord.FFmpegPCMAudio(audio_file, pipe=True, executable=find_ffmpeg()),
                after=after_play,
            )

            # Wait for completion or interruption
            try:
                await asyncio.wait_for(done.wait(), timeout=30.0)
                return True
            except asyncio.TimeoutError:
                log.warning(f"Playback timeout for guild {guild_id}")
                vc.stop()
                return False

        except Exception as e:
            log.error(f"TTS playback failed: {e}")
            return False
        finally:
            self._playing[guild_id] = False
            duration = time.time() - self._play_start.get(guild_id, time.time())
            self._total_play_time[guild_id] = (
                self._total_play_time.get(guild_id, 0) + duration
            )

    async def play_backchannel(
        self,
        guild_id: int,
        audio_data: bytes,
    ) -> bool:
        """
        Play a backchannel acknowledgment.
        These are short and don't interrupt ongoing playback.
        """
        # Don't interrupt if already playing TTS
        if self._playing.get(guild_id, False):
            return False

        return await self.play_tts(guild_id, audio_data, interruptible=False)

    async def stop(self, guild_id: int) -> None:
        """Stop current playback (barge-in)."""
        vc = voice_state.get_voice_client(guild_id)
        if vc and vc.is_playing():
            vc.stop()
            log.info(f"Stopped playback for guild {guild_id}")

        self._playing[guild_id] = False

        # Cancel any pending task
        task = self._current_task.pop(guild_id, None)
        if task and not task.done():
            task.cancel()

    def is_playing(self, guild_id: int) -> bool:
        """Check if audio is currently playing."""
        return self._playing.get(guild_id, False)

    def get_play_duration(self, guild_id: int) -> float:
        """How long the current playback has been going."""
        if not self._playing.get(guild_id, False):
            return 0.0
        start = self._play_start.get(guild_id, 0)
        return time.time() - start if start > 0 else 0.0

    def get_stats(self, guild_id: int) -> dict:
        """Get mixer statistics."""
        return {
            "is_playing": self.is_playing(guild_id),
            "total_play_time": self._total_play_time.get(guild_id, 0),
        }

    def cleanup(self, guild_id: int) -> None:
        """Clean up guild data."""
        self._playing.pop(guild_id, None)
        self._current_task.pop(guild_id, None)
        self._play_start.pop(guild_id, None)
        self._total_play_time.pop(guild_id, None)
