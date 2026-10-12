"""
cogs/alice_gift.py
===================
/alice-gift -- give Alice a gift via modal.
Increments per-user gift count, upgrades relationship tier,
and generates an in-character reaction.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

import config
from bot.logging_setup import get_logger
from character import prompt_builder
from memory import store as memory_store
from social import identities as social_identities

log = get_logger("cogs.alice_gift")

_COOLDOWN_SECONDS = 1

# Silent 'h' words that take "an" instead of "a"
_SILENT_H_WORDS = {"hour", "heir", "honest", "honor", "herb", "heirloom", "hereditary"}


def _prepend_article(text: str) -> str:
    """Return text prepended with 'a' or 'an' based on pronunciation."""
    text = text.strip().lower()
    if not text:
        return "a gift"
    first_word = text.split()[0]
    if first_word in _SILENT_H_WORDS:
        return f"an {text}"
    if first_word[0] in "aeiou":
        return f"an {text}"
    return f"a {text}"
_ALICE_GIFT_USER_ID = "alice-gift"
_FALLBACK_REPLY = config.FALLBACK_REPLY

_GIFTS_PATH = Path(__file__).resolve().parent.parent / "character" / "data" / "alice" / "alice_gifts.json"


def _load_gifts() -> dict:
    try:
        with _GIFTS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.error(f"Failed to load alice_gifts.json: {e}")
        return {"categories": {}, "fallback_reaction": "She accepts it quietly."}


_GIFTS_DATA = _load_gifts()


async def _category_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    """Autocomplete for gift categories."""
    categories = _GIFTS_DATA.get("categories", {})
    return [
        app_commands.Choice(name=cat["label"], value=key)
        for key, cat in categories.items()
        if current.lower() in key.lower() or current.lower() in cat["label"].lower()
    ][:25]


class AliceGiftModal(discord.ui.Modal):
    def __init__(self, cog: AliceGiftCog, category_key: str) -> None:
        # Update title to show selected category (max 45 chars)
        categories = _GIFTS_DATA.get("categories", {})
        cat_label = categories.get(category_key, {}).get("label", category_key)
        title = f"Give Alice a Gift ({cat_label})"[:45]

        super().__init__(title=title)
        self._cog = cog
        self._category_key = category_key

        # Dynamic label/placeholder for letter category
        if category_key == "letter":
            item_label = "What does the letter say?"
            item_placeholder = "Write your letter to Alice..."
        else:
            item_label = "What are you giving her?"
            item_placeholder = "Describe the item..."

        # Create TextInputs per-instance to avoid duplicate custom_id
        self.item_description = discord.ui.TextInput(
            label=item_label,
            style=discord.TextStyle.paragraph,
            placeholder=item_placeholder,
            min_length=1,
            max_length=500,
            required=True,
        )

        self.note = discord.ui.TextInput(
            label="Note to Alice (optional)",
            style=discord.TextStyle.short,
            placeholder="A message to go with the gift...",
            max_length=200,
            required=False,
        )

        self.add_item(self.item_description)
        self.add_item(self.note)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            "Your gift has been given to Alice.", ephemeral=True
        )

        user_id = str(interaction.user.id)
        channel_id = str(interaction.channel_id)

        item_desc = self.item_description.value
        note_text = self.note.value or ""

        log.info(
            f"Alice gift submitted by user={user_id} "
            f"in channel={channel_id}: category={self._category_key}, item={item_desc}"
        )

        await self._cog.process_gift(
            user_id=user_id,
            channel_id=channel_id,
            category_key=self._category_key,
            item_description=item_desc,
            note=note_text,
            giver_mention=interaction.user.mention,
        )


class AliceGiftCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    async def process_gift(
        self,
        user_id: str,
        channel_id: str,
        category_key: str,
        item_description: str,
        note: str,
        giver_mention: str,
    ) -> None:
        character = self.bot.character
        identity = social_identities.get_identity(user_id)

        count, new_relationship = await asyncio.to_thread(
            memory_store.increment_gift_count, user_id
        )

        category = _GIFTS_DATA.get("categories", {}).get(category_key, {})
        reaction_hint = category.get("reaction_hint", _GIFTS_DATA.get("fallback_reaction"))

        messages = prompt_builder.build_gift_prompt(
            character,
            identity=identity if identity.get("is_authoritative") else None,
            relationship_key=new_relationship,
            memory_snippets=[],
            history=[],
            category_key=category_key,
            item_description=item_description,
            note=note,
            fast_path=False,
        )

        try:
            reply = await self.bot.ai_client.generate_reply(
                messages, user_id=_ALICE_GIFT_USER_ID, channel_id=channel_id
            )
            log.debug(f"Alice gift raw reply: {reply!r}")
            reply = reply or _FALLBACK_REPLY
        except Exception as e:
            log.error(f"Alice gift reply generation failed: {e}", exc_info=True)
            reply = _FALLBACK_REPLY

        # For letters, ensure the reply includes the muttered reading (safety net)
        if category_key == "letter" and not reply.lstrip().startswith("*"):
            log.warning(f"Letter reply missing muttered reading, prepending default. Reply: {reply!r}")
            reply = f"*reads quietly* \"{item_description[:50]}...\" *looks up, cheeks red* {reply}"

        gif = None
        if random.random() < 0.3:
            gif = self.bot.gif_selector.pick_for_intent("alice_gift")
        if not gif:
            gif = self.bot.gif_selector.pick_for_emotion(user_id, "happy")

        display_reply = f"{reply}\n{gif}" if gif else reply

        # Friendly category name for public message
        if category_key == "letter":
            public_desc = "a letter"
        else:
            public_desc = _prepend_article(item_description)

        try:
            channel = self.bot.get_channel(int(channel_id))
            if channel:
                await channel.send(f"{giver_mention} gave Alice {public_desc}:\n\n{display_reply}")
            else:
                log.warning(f"Channel {channel_id} not found for alice gift reply")
        except discord.HTTPException as e:
            log.error(f"Failed to send alice gift reply to channel={channel_id}: {e}")

    @app_commands.command(
        name="alice-gift", description="Give Alice a gift"
    )
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    @app_commands.checks.cooldown(1, _COOLDOWN_SECONDS, key=lambda i: i.user.id)
    @app_commands.autocomplete(category=_category_autocomplete)
    async def alice_gift(
        self, interaction: discord.Interaction, category: str
    ) -> None:
        await interaction.response.send_modal(AliceGiftModal(self, category))


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AliceGiftCog(bot))