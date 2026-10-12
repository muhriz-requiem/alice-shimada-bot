"""
cogs/reddit_cog.py
=====================
Slash commands for pulling a random post from each configured
subreddit. Both commands share one helper so behavior (embed shape,
error handling) stays consistent, and adding a third subreddit later
is: one new short command + one new entry in config.REDDIT_SUBREDDITS.
"""

from __future__ import annotations

import asyncio

import discord
from discord import app_commands
from discord.ext import commands

import config
from bot.logging_setup import get_logger
from reddit import rotation

log = get_logger("cogs.reddit")


class RedditCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    async def _send_random_post(self, interaction: discord.Interaction, subreddit: str) -> None:
        await interaction.response.defer()

        meta = config.REDDIT_SUBREDDITS[subreddit]
        post = await asyncio.to_thread(rotation.get_random_post, subreddit)

        if not post:
            await interaction.followup.send(
                f"No posts available from {meta['display_name']} yet -- "
                "try again after the next refresh, or ask an admin to run "
                "`/admin-refresh-reddit`."
            )
            return

        embed = discord.Embed(title=post["title"], url=post["url"], color=meta["embed_color"])
        if post.get("image"):
            embed.set_image(url=post["image"])
        embed.set_footer(text=f"{meta['display_name']} \u2022 Score: {post['score']}")

        await interaction.followup.send(embed=embed)

    @app_commands.command(name="alicepost", description="Random post from r/AliceShimada")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    async def alicepost(self, interaction: discord.Interaction) -> None:
        await self._send_random_post(interaction, "AliceShimada")

    @app_commands.command(name="guppost", description="Random post from r/GIRLSundPANZER")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    async def guppost(self, interaction: discord.Interaction) -> None:
        await self._send_random_post(interaction, "GIRLSundPANZER")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(RedditCog(bot))
