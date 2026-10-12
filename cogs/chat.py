"""
cogs/chat.py
=============
The core conversation pipeline: /chat slash command, DM handling, and
@mention handling in servers all funnel through the same
`process_message`, so behavior is identical no matter how the
conversation reaches the bot.

Pipeline per message:
  1. Look up the sender's identity (e.g. father-roleplay for a
     specific Discord ID) and relationship.
  2. Update per-user emotion state.
  3. Add the message to long-term memory (graph) and short-term
     history (chat buffer).
  4. Build the prompt (full character prompt for Ollama/Groq, or a
     light identity/relationship note for Shapes, whose dashboard owns
     the personality).
  5. Call the AI backend.
  6. Pick a reaction GIF, if any.
  7. Persist the turn to SQLite + the rolling chat buffer.
"""

from __future__ import annotations

import asyncio
import time

import discord
from discord import app_commands
from discord.ext import commands

import config
from ai.postprocess import FALLBACK_REPLY
from ai.router import is_fast_path
from bot.logging_setup import get_logger
from character import prompt_builder
from emotion.intents import detect_intent
from memory import store as memory_store
from social.identities import get_identity_async

log = get_logger("cogs.chat")

_BUSY_MESSAGE = "Still replying to your last message -- one moment."


def _truncate_for_discord(reply: str) -> str:
    """
    Last-resort safety net before anything reaches Discord:
    - Discord hard-rejects an empty message body (400: "Cannot send an
      empty message") -- every upstream layer (clean_response,
      GroqClient) already guards against returning "", but a different
      AI backend (Shapes, Ollama) could still hand back empty/whitespace,
      so this is the final backstop regardless of which backend ran.
    - Discord also rejects anything over 2000 chars (4000 on boosted
      servers); GROQ_MAX_TOKENS should prevent this, but this stays
      under both limits as a last-resort truncation net.
    Either failure mode, left unhandled, crashes on_message with no
    reply sent to the user at all.
    """
    if not reply or not reply.strip():
        log.warning("Reply was empty before send; using fallback text.")
        return FALLBACK_REPLY

    limit = config.DISCORD_MAX_REPLY_CHARS
    if len(reply) <= limit:
        return reply
    log.warning(f"Reply exceeded {limit} chars ({len(reply)}); truncating before send.")
    return reply[: limit - 1].rstrip() + "…"


class ChatCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._active_users: set[str] = set()

    async def process_message(self, user_id: str, channel_id: str, user_message: str) -> str:
        if user_id in self._active_users:
            return _BUSY_MESSAGE

        self._active_users.add(user_id)
        start = time.monotonic()

        try:
            character = self.bot.character
            identity = await get_identity_async(user_id)

            relationship = await asyncio.to_thread(memory_store.get_relationship, user_id)

            # Auto-assign a relationship from identity data (e.g. the
            # father-roleplay relationship) the first time we see this
            # user, without overriding a relationship that's already
            # been explicitly set to something else.
            auto_rel = identity.get("auto_relationship")
            if auto_rel and relationship == "stranger":
                relationship = auto_rel
                await asyncio.to_thread(memory_store.set_relationship, user_id, relationship)

            fast_path = is_fast_path(user_message)

            emotion_engine = self.bot.get_emotion_engine(user_id)
            await emotion_engine.update_async(user_message, relationship, character.identity_keywords)
            emotion = emotion_engine.get_emotion(user_message)

            self.bot.memory_graph.add(user_id, user_message, character.identity_keywords)
            memory_snippets = [] if fast_path else self.bot.memory_graph.retrieve(
                user_id, config.MEMORY_GRAPH_LIMIT
            )

            history = self.bot.chat_buffer.history(user_id)

            messages = prompt_builder.build_prompt(
                character,
                identity=identity if identity.get("is_authoritative") else None,
                relationship_key=relationship,
                memory_snippets=memory_snippets,
                history=history,
                user_message=user_message,
                fast_path=fast_path,
            )

            reply = await self.bot.ai_client.generate_reply(
                messages, user_id=user_id, channel_id=channel_id
            )

            # --- GIF selection: explicit intent bypasses cooldown, ---
            # --- ambient emotion is the cooldown-gated fallback.    ---
            intent = detect_intent(user_message, character.intents)
            gif = None

            if intent.get("goodnight"):
                gif = self.bot.gif_selector.pick_for_intent("goodnight")
            elif intent.get("agree_special"):
                gif = self.bot.gif_selector.pick_for_intent("agree_special")
            elif intent.get("agree"):
                gif = self.bot.gif_selector.pick_for_intent("agree")
            else:
                gif = await self.bot.gif_selector.pick_for_emotion_async(user_id, emotion)

            # IMPORTANT: the GIF is a Discord-display-only addition, not
            # something Alice "said". It must never be stored in
            # chat_buffer/memory as part of her turn -- if it were, the
            # next prompt's conversation history would show her own past
            # replies ending in a raw URL, and the model would start
            # imitating that pattern by hallucinating URLs of its own
            # (which then get hard-cut by postprocess.py, truncating
            # otherwise-fine replies mid-sentence). Keep the two separate:
            # `reply` (clean, no GIF) is what gets stored as history;
            # `display_reply` (reply + GIF) is what actually gets sent to
            # Discord.
            display_reply = f"{reply}\n{gif}" if gif else reply

            self.bot.chat_buffer.push(user_id, "user", user_message)
            self.bot.chat_buffer.push(user_id, "assistant", reply)

            await asyncio.to_thread(memory_store.add_message, user_id, "user", user_message)
            await asyncio.to_thread(memory_store.add_message, user_id, "assistant", reply)

            log.info(f"Processed message for user={user_id} in {round(time.monotonic() - start, 2)}s")
            return display_reply

        except Exception as e:  # noqa: BLE001 - a pipeline bug must never crash the event loop
            log.error(f"Pipeline error for user={user_id}: {e}", exc_info=True)
            return "Sorry, something went wrong -- please try again."

        finally:
            self._active_users.discard(user_id)

    @app_commands.command(name="chat", description="Talk to the character")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    async def chat_command(self, interaction: discord.Interaction, message: str) -> None:
        await interaction.response.defer(thinking=True)
        reply = await self.process_message(
            user_id=str(interaction.user.id),
            channel_id=str(interaction.channel_id),
            user_message=message,
        )
        try:
            await interaction.followup.send(_truncate_for_discord(reply))
        except discord.HTTPException as e:
            log.error(f"Failed to send /chat reply for user={interaction.user.id}: {e}")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.bot.user in message.mentions

        if not is_dm and not is_mentioned:
            return

        clean_text = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
        if not clean_text:
            return

        async with message.channel.typing():
            reply = await self.process_message(
                user_id=str(message.author.id),
                channel_id=str(message.channel.id),
                user_message=clean_text,
            )
            try:
                await message.reply(_truncate_for_discord(reply))
            except discord.HTTPException as e:
                log.error(f"Failed to send reply to user={message.author.id}: {e}")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(ChatCog(bot))
