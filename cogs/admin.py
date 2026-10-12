"""
cogs/admin.py
===============
Owner-only maintenance commands. Access is restricted to Discord user
IDs listed in config.OWNER_IDS (from the OWNER_IDS env var) -- anyone
else gets a clean "not authorized" response via the CheckFailure
handling already wired up in bot/errors.py.
"""

from __future__ import annotations

import asyncio
import time

import discord
from discord import app_commands
from discord.ext import commands

import config
from bot.logging_setup import get_logger
from memory import store as memory_store
from reddit import fetcher, updater
from social import identities as social_identities

log = get_logger("cogs.admin")


def _is_owner(interaction: discord.Interaction) -> bool:
    return str(interaction.user.id) in config.OWNER_IDS


class AdminCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._started_at = time.monotonic()

    @app_commands.command(name="admin-refresh-reddit", description="[Owner] Force a Reddit fetch + rebuild now")
    @app_commands.check(_is_owner)
    async def admin_refresh_reddit(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        await asyncio.to_thread(fetcher.fetch_all)
        stats = await asyncio.to_thread(updater.build_all)

        lines = [f"**{sub}**: {s.get('final_posts', 0)} posts" for sub, s in stats.items()]
        await interaction.followup.send("Reddit refresh complete.\n" + "\n".join(lines), ephemeral=True)

    @app_commands.command(name="admin-reset-memory", description="[Owner] Wipe a user's stored memory")
    @app_commands.describe(user="The user whose memory should be reset")
    @app_commands.check(_is_owner)
    async def admin_reset_memory(self, interaction: discord.Interaction, user: discord.User) -> None:
        user_id = str(user.id)
        await asyncio.to_thread(memory_store.reset_user, user_id)
        self.bot.memory_graph.clear(user_id)
        self.bot.chat_buffer.clear(user_id)
        self.bot.reset_emotion(user_id)
        await interaction.response.send_message(f"Memory reset for {user.mention}.", ephemeral=True)

    @app_commands.command(name="admin-reload-identities", description="[Owner] Reload identities.json without restarting")
    @app_commands.check(_is_owner)
    async def admin_reload_identities(self, interaction: discord.Interaction) -> None:
        social_identities.reload()
        await interaction.response.send_message("Identities reloaded.", ephemeral=True)

    @app_commands.command(name="admin-stats", description="[Owner] Show quick bot stats")
    @app_commands.check(_is_owner)
    async def admin_stats(self, interaction: discord.Interaction) -> None:
        uptime_min = round((time.monotonic() - self._started_at) / 60, 1)
        message_count = await asyncio.to_thread(memory_store.count_messages)

        lines = [
            f"Uptime: {uptime_min} min",
            f"AI backend: {config.AI_BACKEND}",
            f"Character: {self.bot.character.display_name}",
            f"Stored messages: {message_count}",
            f"Guilds: {len(self.bot.guilds)}",
        ]

        for sub, meta in config.REDDIT_SUBREDDITS.items():
            count = 0
            if meta["posts_file"].exists():
                import json
                try:
                    count = len(json.loads(meta["posts_file"].read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    count = -1
            lines.append(f"{meta['display_name']} pool: {count} posts")

        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    @app_commands.command(name="admin-sync", description="[Owner] Re-sync slash commands")
    @app_commands.check(_is_owner)
    async def admin_sync(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        synced = await self.bot.tree.sync()
        await interaction.followup.send(f"Synced {len(synced)} commands.", ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AdminCog(bot))
