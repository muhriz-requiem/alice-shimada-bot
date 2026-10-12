"""
bot/errors.py
==============
Centralized Discord-facing error handling, so one bad interaction (a
timed-out AI call, an unexpected exception in a cog) logs clearly and
tells the user something reasonable instead of the bot silently
failing or crashing.
"""

from __future__ import annotations

import discord
from discord import app_commands

from bot.logging_setup import get_logger

log = get_logger("bot.errors")

_USER_FACING_ERROR = "Something went wrong on my end -- please try again in a moment."


async def on_app_command_error(
    interaction: discord.Interaction,
    error: app_commands.AppCommandError,
) -> None:
    # CommandOnCooldown is actually a *subclass* of CheckFailure in
    # discord.py's app_commands module, so it must be checked first --
    # otherwise the generic CheckFailure branch below catches cooldown
    # hits too and reports the wrong ("no permission") message.
    if isinstance(error, app_commands.CommandOnCooldown):
        message = f"That command is on cooldown -- try again in {error.retry_after:.1f}s."
    elif isinstance(error, app_commands.CheckFailure):
        message = "You don't have permission to use this command."
    else:
        log.error(f"Unhandled app command error in /{interaction.command.name if interaction.command else '?'}: {error}")
        message = _USER_FACING_ERROR

    try:
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
    except discord.HTTPException as e:
        log.error(f"Failed to send error response to user: {e}")
