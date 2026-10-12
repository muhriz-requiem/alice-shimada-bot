"""
cogs/confess.py
==================
/aliceconfess -- anonymous confession box. The confession is posted as
its own embed, and Alice's reply follows underneath as a separate
plain message (not inside the embed).

Anonymity is enforced at every layer, not just visually:

1. The confession embed is sent as a plain channel message
   (`interaction.channel.send(...)`), never as the interaction's own
   response. Discord tags an interaction's own response with a visible
   "[user] used /aliceconfess" indicator -- posting the embed that way
   would deanonymize the confessor immediately, regardless of what the
   embed itself contains. The interaction response is used only for a
   private, ephemeral "your confession was sent" acknowledgment that
   only the confessor ever sees.

2. Alice's reply pipeline never receives the confessor's real Discord
   ID -- a fixed synthetic ID is used instead. This matters beyond the
   AI backend itself: it means the confession never touches that
   person's normal memory, relationship, or emotion state, so it can't
   quietly change how Alice treats them later in an ordinary /chat --
   which would otherwise be a backdoor way of deanonymizing them
   through her behavior, even with nothing ever displayed.

3. A purpose-built identity dict (NOT None) is passed to the prompt
   builder, explicitly describing an anonymous confessor. Passing None
   would fall back to prompt_builder's generic "unfamiliar person"
   wording, which says nothing about anonymity specifically and reads
   as a mismatch against the extra_context note below -- giving both
   pieces the same coherent framing keeps the model from treating this
   as just "a stranger" (chatty/curious) instead of "someone who can
   never be identified" (no guessing, no asking).

The only place the real submitter is ever recorded is this process's
own log file, for basic abuse/moderation purposes -- never shown to
Discord, never sent to the AI backend, never linked to the embed. If
you'd rather have zero record of who submitted what, even privately,
just remove the log.info() call in on_submit below.
"""

from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

import config
from bot.logging_setup import get_logger
from character import prompt_builder

log = get_logger("cogs.confess")

_CONFESSION_COLOR = 0x9B59B6
_ANONYMOUS_USER_ID = "anonymous-confession"
_COOLDOWN_SECONDS = 1  # basic spam guard, adjust freely
_FALLBACK_REPLY = config.FALLBACK_REPLY

_ANONYMOUS_IDENTITY = {
    "name": "someone anonymous",
    "role": "an anonymous confessor",
    "description": (
        "This message came through an anonymous confession box. She "
        "cannot know who sent it, cannot guess, and cannot ask -- that's "
        "simply how a confession box works, not something worth "
        "commenting on or being confused by."
    ),
}


class ConfessionModal(discord.ui.Modal, title="Anonymous Confession"):
    confession = discord.ui.TextInput(
        label="What do you want to confess?",
        style=discord.TextStyle.paragraph,
        placeholder="This will be posted anonymously -- no one will know it was you.",
        min_length=1,
        max_length=1000,
        required=True,
    )

    def __init__(self, cog: ConfessCog) -> None:
        super().__init__()
        self._cog = cog

    async def on_submit(self, interaction: discord.Interaction) -> None:
        # The ONLY interaction response tied to the confessor's identity --
        # ephemeral, so only they ever see it.
        await interaction.response.send_message(
            "Your confession has been posted anonymously.", ephemeral=True
        )

        log.info(
            f"Confession submitted by user={interaction.user.id} "
            f"in channel={interaction.channel_id}"
        )

        embed = discord.Embed(
            title="\U0001f48c Anonymous Confession",
            description=self.confession.value,
            color=_CONFESSION_COLOR,
            timestamp=discord.utils.utcnow(),
        )

        # Plain channel message, NOT the interaction response -- see
        # module docstring for why this distinction is what actually
        # keeps the confessor anonymous.
        await interaction.channel.send(embed=embed)

        reply = await self._cog.generate_reply(self.confession.value, str(interaction.channel_id))
        await interaction.channel.send(reply)


class ConfessCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    async def generate_reply(self, confession_text: str, channel_id: str) -> str:
        character = self.bot.character

        messages = prompt_builder.build_prompt(
            character,
            identity=_ANONYMOUS_IDENTITY,
            relationship_key="anonymous",
            memory_snippets=[],
            history=[],
            user_message=confession_text,
            fast_path=False,
            extra_context=(
                "You just received this message anonymously through a confession "
                "box. You genuinely do not know who sent it and have no way to "
                "find out -- don't guess, don't ask who they are, and don't "
                "address them by any name. Just react to what they confessed, "
                "in your own voice, the same shy, quiet way you'd react to "
                "anything else."
            ),
        )

        try:
            reply = await self.bot.ai_client.generate_reply(
                messages, user_id=_ANONYMOUS_USER_ID, channel_id=channel_id
            )
            return reply or _FALLBACK_REPLY
        except Exception as e:  # noqa: BLE001 - a failed reply must never crash the modal flow
            log.error(f"Confession reply generation failed: {e}", exc_info=True)
            return _FALLBACK_REPLY

    @app_commands.command(name="aliceconfess", description="Anonymously confess something to Alice")
    @app_commands.allowed_contexts(guilds=True, dms=False, private_channels=False)
    @app_commands.checks.cooldown(1, _COOLDOWN_SECONDS, key=lambda i: i.user.id)
    async def aliceconfess(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(ConfessionModal(self))


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(ConfessCog(bot))
