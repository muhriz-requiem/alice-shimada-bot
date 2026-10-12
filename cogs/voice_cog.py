"""
cogs/voice_cog.py
=================
Discord slash commands for voice channel interaction.
Provides /join, /leave, /voice-status, and handles voice receive
using discord-ext-voice-recv for proper audio reception.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands, voice_recv

from bot.logging_setup import get_logger
import config
from voice import init_pipeline, readiness, voice_pipeline, voice_state
from voice.rvc import is_available as rvc_available, is_loaded as rvc_loaded
from voice.stt import is_loaded as stt_loaded
from voice.state import TurnState

log = get_logger("cogs.voice")

TURN_STATE_EMOJIS = {
    TurnState.IDLE: " ",
    TurnState.SINGLE_SPEAKER: " ",
    TurnState.MULTIPLE_SPEAKERS: " ",
    TurnState.PAUSE: " ",
    TurnState.LULL: "  ",
    TurnState.BOT_RESPONDING: " ",
    TurnState.BARGE_IN: " ",
}


class VoiceCog(commands.Cog, name="Voice"):
    """Voice channel interaction commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def cog_load(self) -> None:
        """Called when the cog is loaded."""
        init_pipeline(self.bot)
        log.info("Voice cog loaded")

    async def cog_unload(self) -> None:
        """Called when the cog is unloaded."""
        for guild_id in list(voice_state.get_connected_guilds()):
            await self._disconnect(guild_id)
        log.info("Voice cog unloaded")

    @app_commands.command(name="join", description="Join your current voice channel")
    async def join(self, interaction: discord.Interaction) -> None:
        """Join the user's current voice channel."""
        await interaction.response.defer()

        if not interaction.user.voice or not interaction.user.voice.channel:
            await interaction.followup.send(
                "You need to be in a voice channel first!", ephemeral=True
            )
            return

        channel = interaction.user.voice.channel
        guild_id = interaction.guild_id

        if voice_state.is_connected(guild_id):
            current_vc = voice_state.get_voice_client(guild_id)
            if current_vc and current_vc.channel.id == channel.id:
                await interaction.followup.send(
                    f"Already connected to {channel.mention}!", ephemeral=True
                )
                return
            else:
                await self._disconnect(guild_id)

        try:
            was_ready = readiness.is_ready()
            if not was_ready:
                readiness.kick()

            vc = await channel.connect(
                cls=voice_recv.VoiceRecvClient,
                self_deaf=False,
                self_mute=not was_ready,
            )
            voice_state.connect(guild_id, channel.id, vc)

            if was_ready:
                sink = voice_pipeline.get_sink(guild_id)
                vc.listen(sink)
                voice_pipeline.start_monitor(guild_id)
                await interaction.followup.send(
                    f"Connected to {channel.mention}! I can now listen and respond."
                )
            else:
                await interaction.followup.send(
                    f"Connected to {channel.mention} (muted while voice warms up — "
                    f"I'll start listening as soon as TTS is ready)."
                )
                asyncio.create_task(
                    self._start_when_ready(guild_id, channel, interaction)
                )
            log.info(f"Joined voice channel: {channel.name} in guild {guild_id}")

        except Exception as e:
            log.error(f"Failed to join voice channel: {e}")
            await interaction.followup.send(
                f"Failed to join voice channel: {e}", ephemeral=True
            )

    @app_commands.command(name="leave", description="Leave the current voice channel")
    async def leave(self, interaction: discord.Interaction) -> None:
        """Leave the current voice channel."""
        await interaction.response.defer()

        guild_id = interaction.guild_id
        if not voice_state.is_connected(guild_id):
            await interaction.followup.send(
                "Not connected to any voice channel!", ephemeral=True
            )
            return

        await self._disconnect(guild_id)
        await interaction.followup.send("Disconnected from voice channel.")

    @app_commands.command(name="voice-status", description="Show voice connection status")
    async def voice_status(self, interaction: discord.Interaction) -> None:
        """Display current voice connection status with conversation state."""
        guild_id = interaction.guild_id

        embed = discord.Embed(
            title="Voice Status",
            color=discord.Color.blue(),
        )

        if voice_state.is_connected(guild_id):
            session = voice_state.get_session(guild_id)
            vc = voice_state.get_voice_client(guild_id)

            embed.add_field(
                name="Channel",
                value=vc.channel.mention if vc else "Unknown",
                inline=True,
            )
            embed.add_field(
                name="Duration",
                value=f"{session.duration:.0f}s" if session else "N/A",
                inline=True,
            )
            embed.add_field(
                name="Messages",
                value=str(session.messages_processed) if session else "0",
                inline=True,
            )

            # Turn state
            if session:
                emoji = TURN_STATE_EMOJIS.get(session.turn_state, "❓")
                embed.add_field(
                    name="Turn State",
                    value=f"{emoji} {session.turn_state.value}",
                    inline=True,
                )
                embed.add_field(
                    name="Active Speakers",
                    value=str(session.active_speaker_count),
                    inline=True,
                )

            # Component status
            is_listening = hasattr(vc, 'is_listening') and vc.is_listening()
            embed.add_field(
                name="Listening",
                value="Yes" if is_listening else "No",
                inline=True,
            )
            embed.add_field(
                name="STT",
                value="Loaded" if stt_loaded() else "Not loaded",
                inline=True,
            )
            embed.add_field(
                name="RVC",
                value=(
                    "Loaded" if rvc_loaded()
                    else ("Loading…" if rvc_available() else "Not available")
                ),
                inline=True,
            )
            embed.add_field(
                name="TTS",
                value=f"{config.VOICE_TTS_ENGINE} ({readiness.state()})",
                inline=True,
            )

            # Pipeline stats
            try:
                stats = voice_pipeline.get_stats(guild_id)
                embed.add_field(
                    name="Conversation",
                    value=(
                        f"Utterances: {stats['conversation']['utterances']}\n"
                        f"Speakers: {stats['conversation']['speakers']}\n"
                        f"Alice silent: {stats['conversation']['alice_silence']:.0f}s"
                    ),
                    inline=False,
                )
                embed.add_field(
                    name="Soft Responses",
                    value=str(stats["response"]["total_soft_responses"]),
                    inline=True,
                )
                embed.add_field(
                    name="Diarization",
                    value=(
                        f"Speakers in last clip: {stats['diarization']['speakers_last']}\n"
                        f"Multi-speaker clips: {stats['diarization']['multi_speaker_clips']}"
                    ),
                    inline=True,
                )
            except Exception:
                pass

        else:
            embed.description = "Not connected to any voice channel."

        await interaction.response.send_message(embed=embed)

    async def _start_when_ready(
        self,
        guild_id: int,
        channel: discord.VoiceChannel,
        interaction: discord.Interaction,
    ) -> None:
        """Wait for a TTS engine to become usable, then start listening and unmute."""
        vc = voice_state.get_voice_client(guild_id)
        if vc and hasattr(vc, "is_listening") and vc.is_listening():
            # Already listening (re-joined during warmup) — nothing to do.
            return

        sink = voice_pipeline.get_sink(guild_id)
        try:
            ok = await readiness.wait_ready(timeout=180.0)
            vc = voice_state.get_voice_client(guild_id)
            if not ok:
                await interaction.followup.send(
                    "TTS still not ready after 180s — leaving the channel. "
                    "/join again once GPT-SoVITS / Edge-TTS is available."
                )
                log.warning(
                    "TTS still not ready after 180s; "
                    "leaving the channel and not listening. "
                    "GPT-SoVITS / Edge-TTS must be available first."
                )
                await self._disconnect(guild_id)
                return
            if not vc or getattr(vc, "channel", None) is None:
                return
            if vc.channel.id != channel.id:
                return

            vc.listen(sink)
            voice_pipeline.start_monitor(guild_id)
            await vc.guild.change_voice_state(channel=vc.channel, self_mute=False)
            log.info(f"Alice started listening (TTS ready: {readiness.state()})")
            try:
                await interaction.followup.send("Voice ready — listening now.")
            except Exception:
                pass
        except Exception as e:
            log.error(f"Start-when-ready failed: {e}", exc_info=True)

    async def _disconnect(self, guild_id: int) -> None:
        """Disconnect from a voice channel."""
        if voice_pipeline:
            voice_pipeline.stop_monitor(guild_id)

        vc = voice_state.get_voice_client(guild_id)
        if vc:
            if hasattr(vc, 'stop_listening'):
                vc.stop_listening()
            await vc.disconnect()

        voice_state.disconnect(guild_id)
        if voice_pipeline:
            voice_pipeline.cleanup(guild_id)


async def setup(bot: commands.Bot) -> None:
    """Setup the voice cog."""
    if not config.VOICE_ENABLED:
        log.info("Voice disabled in config, skipping cog load")
        return

    await bot.add_cog(VoiceCog(bot))
    log.info("Voice cog registered")
