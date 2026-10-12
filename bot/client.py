"""
bot/client.py
==============
The Discord client itself. Wires together every subsystem (AI backend,
character, memory, emotion, reddit) as attributes on the bot instance,
loads cogs, and sets up a background task that keeps the Reddit post
pools fresh automatically.
"""

from __future__ import annotations

import asyncio
import time

import discord
from discord.ext import commands, tasks

import config
from ai import get_ai_client
from bot.errors import on_app_command_error
from bot.logging_setup import get_logger
from character import GifSelector, load_character
from memory import ChatBuffer, MemoryGraph
from metrics import init_metrics, set_memory_graph_cap, set_uptime
from reddit import fetcher, updater

log = get_logger("bot.client")


class AliceBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix=commands.when_mentioned, intents=intents)

        # --- Shared subsystems, built once at startup ---
        self.ai_client = get_ai_client()
        self.character = load_character(config.CHARACTER_NAME)
        self.memory_graph = MemoryGraph()
        self.chat_buffer = ChatBuffer()
        self.gif_selector = GifSelector(self.character)

        # Per-user emotion state (a shared/global emotion engine would
        # mean every Discord user affects the same mood -- kept
        # per-user here since every other piece of state in this bot
        # already is).
        self._emotion_engines: dict[str, object] = {}

        self.memory_graph.hydrate(self.character.identity_keywords)

        # Initialize metrics
        init_metrics(config.CHARACTER_NAME)
        set_memory_graph_cap(config.MEMORY_GRAPH_CAP_PER_USER)
        self._start_time = time.time()

    def get_emotion_engine(self, user_id: str):
        from emotion import EmotionEngine

        if user_id not in self._emotion_engines:
            self._emotion_engines[user_id] = EmotionEngine(user_id)
        return self._emotion_engines[user_id]

    def reset_emotion(self, user_id: str) -> None:
        self._emotion_engines.pop(user_id, None)

    async def setup_hook(self) -> None:
        await self.load_extension("cogs.chat")
        await self.load_extension("cogs.reddit_cog")
        await self.load_extension("cogs.admin")
        await self.load_extension("cogs.confess")
        await self.load_extension("cogs.alice_gift")

        # Load voice cog if enabled
        if config.VOICE_ENABLED:
            try:
                await self.load_extension("cogs.voice_cog")
            except Exception as e:
                log.error(f"Failed to load voice cog: {e}")

        self.tree.on_error = on_app_command_error

        if config.DEV_GUILD_ID:
            try:
                guild_id = int(config.DEV_GUILD_ID)
            except ValueError:
                log.warning(f"DEV_GUILD_ID='{config.DEV_GUILD_ID}' is not a valid integer; skipping guild sync")
            else:
                guild = discord.Object(id=guild_id)
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                log.info(f"Synced {len(synced)} commands to dev guild {guild_id}.")
        else:
            synced = await self.tree.sync()
            log.info(f"Synced {len(synced)} global commands.")

        self.reddit_refresh_loop.start()

        # Start uptime tracking
        self._uptime_task = asyncio.create_task(self._update_uptime())

    async def _update_uptime(self) -> None:
        """Periodically update uptime metric."""
        while True:
            await asyncio.sleep(10)
            set_uptime(time.time() - self._start_time)

    async def on_ready(self) -> None:
        log.info(f"Logged in as {self.user} (id={self.user.id})")

    async def on_command_error(
        self, ctx: commands.Context, error: commands.CommandError
    ) -> None:
        # This bot has no text (prefix) commands -- everything is a
        # slash command or the mention/DM pipeline in cogs.chat's own
        # on_message listener. But command_prefix=when_mentioned means
        # discord.py's default on_message still tries to parse
        # whatever follows an @mention as a text command name (e.g.
        # "@Alice Shimada Hey Alice!" -> tries to run command "Hey").
        # That's expected and harmless here, so CommandNotFound is
        # swallowed quietly instead of logged as an error. Anything
        # else still gets logged since it could be a real bug.
        if isinstance(error, commands.CommandNotFound):
            return
        log.error(f"Unhandled text command error: {error}", exc_info=error)

    @tasks.loop(hours=config.REDDIT_FETCH_INTERVAL_HOURS)
    async def reddit_refresh_loop(self) -> None:
        log.info("Starting scheduled Reddit refresh...")
        try:
            await asyncio.to_thread(fetcher.fetch_all)
            await asyncio.to_thread(updater.build_all)
            log.info("Scheduled Reddit refresh complete.")
        except Exception as e:  # noqa: BLE001 - never let this kill the loop
            log.error(f"Scheduled Reddit refresh failed: {e}")

    @reddit_refresh_loop.before_loop
    async def _before_reddit_refresh(self) -> None:
        await self.wait_until_ready()
