"""
main.py
==========
Entrypoint. Loads environment variables, sets up logging, validates
configuration (failing fast with a clear message instead of failing
deep inside a Discord event handler), and starts the bot.

Run this directly:  python main.py
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from bot.logging_setup import get_logger, setup_logging

setup_logging()
log = get_logger("main")

import config


async def _start_health_server(bot):
    from health import start_health_server
    await start_health_server(bot, config.HEALTH_HOST, config.HEALTH_PORT)


def main() -> None:
    try:
        config.validate()
    except config.ConfigError as e:
        log.error(str(e))
        sys.exit(1)

    from memory import store as memory_store
    memory_store.init_db()

    from bot.client import AliceBot

    bot = AliceBot()

    # Start health check server in background
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(_start_health_server(bot))

    try:
        bot.run(config.DISCORD_TOKEN, log_handler=None)
    except Exception as e:  # noqa: BLE001
        log.error(f"Bot crashed on startup: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
