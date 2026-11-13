"""Neutral lip-sync publisher (no desktop import).

The voice pipeline publishes synthesized-sentence events here; the desktop
avatar (MateEngine) subscribes and drives the mouth. Keeping this module
free of any desktop import means voice/ stays usable in headless/Discord
contexts where no avatar exists.
"""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from typing import Any

log_name = "voice.lip_sync"

try:
    from bot.logging_setup import get_logger
    log = get_logger(log_name)
except Exception:  # noqa: BLE001 - keep importable even if logging isn't ready
    import logging
    log = logging.getLogger(log_name)

# Subscriber callbacks receive one event dict:
#   {"text": str, "wav_bytes": bytes, "sr": int, "engine": str,
#    "duration": float}
_subscribers: list[Callable[[dict[str, Any]], None]] = []
_lock = threading.Lock()


def subscribe(cb: Callable[[dict[str, Any]], None]) -> None:
    with _lock:
        if cb not in _subscribers:
            _subscribers.append(cb)


def unsubscribe(cb: Callable[[dict[str, Any]], None]) -> None:
    with _lock, contextlib.suppress(ValueError):
        _subscribers.remove(cb)


def publish(event: dict[str, Any]) -> None:
    """Notify subscribers. Never raises; subscribers must be fast (enqueue only)."""
    with _lock:
        subs = list(_subscribers)
    for cb in subs:
        try:
            cb(event)
        except Exception as exc:  # noqa: BLE001
            log.warning("lip_sync subscriber failed: %s", exc)


def has_subscribers() -> bool:
    with _lock:
        return bool(_subscribers)
