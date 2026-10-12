"""Behavioral checks for the warmup gating change: Alice does not listen until TTS is ready."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from voice import readiness


class FakeVc:
    def __init__(self):
        self.guild = MagicMock()
        self.guild.change_voice_state = AsyncMock()
        self.channel = MagicMock()
        self.channel.id = 999
        self._listening = False
        self.listen = MagicMock()
        self.stop_listening = MagicMock()

    def is_listening(self):
        return self._listening


class FakeInteraction:
    def __init__(self):
        self.followup = MagicMock()
        self.followup.send = AsyncMock()


async def test_start_when_ready_defers_listening_until_ready():
    """listen()/start_monitor()/unmute happen only after TTS is ready."""
    from cogs.voice_cog import VoiceCog

    cog = VoiceCog(MagicMock())
    vc = FakeVc()
    pipeline = MagicMock()

    with patch("cogs.voice_cog.voice_state") as mock_vs, \
            patch("cogs.voice_cog.voice_pipeline", pipeline), \
            patch("voice.readiness.is_ready", return_value=False), \
            patch("voice.readiness.wait_ready", return_value=True):
        mock_vs.get_voice_client.return_value = vc

        await cog._start_when_ready(999, vc.channel, FakeInteraction())

        vc.listen.assert_called_once()
        pipeline.start_monitor.assert_called_once()
        vc.guild.change_voice_state.assert_called_once_with(channel=vc.channel, self_mute=False)


async def test_start_when_ready_returns_immediately_if_already_listening():
    """Re-joined during warmup -> don't start a second listener."""
    from cogs.voice_cog import VoiceCog

    cog = VoiceCog(MagicMock())
    vc = FakeVc()
    vc._listening = True
    pipeline = MagicMock()

    with patch("cogs.voice_cog.voice_state") as mock_vs, \
            patch("cogs.voice_cog.voice_pipeline", pipeline):
        mock_vs.get_voice_client.return_value = vc

        await cog._start_when_ready(999, vc.channel, FakeInteraction())

        vc.listen.assert_not_called()
        pipeline.start_monitor.assert_not_called()


async def test_start_when_ready_disconnects_on_timeout():
    """No TTS after 180s -> warn, leave, and never start listening."""
    from cogs.voice_cog import VoiceCog

    cog = VoiceCog(MagicMock())
    vc = FakeVc()
    pipeline = MagicMock()

    with patch("cogs.voice_cog.voice_state") as mock_vs, \
            patch("cogs.voice_cog.voice_pipeline", pipeline), \
            patch("voice.readiness.wait_ready", return_value=False), \
            patch.object(cog, "_disconnect") as mock_disconnect:
        mock_vs.get_voice_client.return_value = vc

        await cog._start_when_ready(999, vc.channel, FakeInteraction())

        vc.listen.assert_not_called()
        pipeline.start_monitor.assert_not_called()
        mock_disconnect.assert_called_once()
