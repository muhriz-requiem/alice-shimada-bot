"""
bot/logging_setup.py
=====================
Centralized logging configuration. Logs go to both the console (for
live viewing on a hosting panel's console tab) and a rotating file
under logs/, so history survives even if the console scrollback
doesn't. File size is capped and old files are rotated out
automatically, so logs can't silently fill up a small host's disk.

Supports JSON structured logging for log aggregation systems.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys

import config

try:
    from pythonjsonlogger import jsonlogger
    _HAS_JSON_LOGGER = True
except ImportError:
    _HAS_JSON_LOGGER = False

_CONFIGURED = False


def setup_logging() -> None:
    """Idempotent: safe to call more than once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, config.LOG_LEVEL, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Use JSON formatter if available and LOG_JSON is set
    use_json = _HAS_JSON_LOGGER and getattr(config, "LOG_JSON", False)

    if use_json:
        fmt: logging.Formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=config.LOG_DIR / "bot.log",
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # discord.py's own logging is fairly chatty at INFO; keep it at
    # WARNING unless the whole app is running in DEBUG.
    if level > logging.DEBUG:
        logging.getLogger("discord").setLevel(logging.WARNING)
        logging.getLogger("discord.http").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
