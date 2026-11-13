"""
voice/readiness.py
==================
TTS readiness gate.

Alice stays silent (and self-muted in Discord) until a TTS engine has
been proven usable: either the local GPT-SoVITS server responds, or the
Edge-TTS fallback is available. While not ready, reply/backchannel
emission is dropped instead of reaching the voice channel.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import config
from bot.logging_setup import get_logger

log = get_logger("voice.readiness")

STATE_STARTING = "starting"
STATE_READY = "ready"
STATE_FALLBACK = "fallback"
STATE_UNAVAILABLE = "unavailable"

_POLL_INTERVAL = 30.0
_PROBE_RETRIES = 3
_PROBE_RETRY_DELAY = 3.0

_state = STATE_STARTING
_task: asyncio.Task | None = None
# Optional callbacks (registered via kick(on_ready=...)) that run once
# a TTS engine is confirmed usable. Used to defer heavy preloads (and
# the live-mode backchannel bank build) until the GPU isn't fighting
# the TTS warm-up. Each callback should return quickly (spawn
# background threads/tasks) since it runs on the event loop.
_on_ready_hooks: list[Callable[[], None]] = []

# After falling back to Edge-TTS, keep re-probing GPT-SoVITS this many
# poll cycles (each _POLL_INTERVAL apart) so a slow-starting local
# server can still take over mid-session instead of Alice being stuck
# on the fallback voice permanently.
_UPGRADE_ATTEMPTS = 12


def state() -> str:
    return _state


def is_ready() -> bool:
    return _state in (STATE_READY, STATE_FALLBACK)


def kick(on_ready: Callable[[], None] | None = None) -> None:
    """Start the background readiness loop (idempotent).

    on_ready, when given, is queued and runs once as soon as a TTS
    engine is confirmed usable (ready, or the fallback state settling
    with no upgrade possible). Calling kick again with or without a
    callback never restarts the loop.
    """
    global _task
    if on_ready is not None:
        _on_ready_hooks.append(on_ready)
    if _task is not None and not _task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _task = loop.create_task(_run())


def _fire_on_ready_hook() -> None:
    """Run (once, draining) the deferred on-ready callbacks."""
    global _on_ready_hooks
    hooks, _on_ready_hooks = _on_ready_hooks, []
    if not hooks:
        return
    log.info(f"TTS engine confirmed; running {len(hooks)} deferred on-ready hook(s)")
    for hook in hooks:
        try:
            hook()
        except Exception as e:
            log.warning(f"Readiness on-ready hook failed: {e}")


async def _run() -> None:
    fallback_upgrades = 0
    while True:
        try:
            await _refresh()
        except Exception as e:
            log.warning(f"Readiness check error: {e}")
        if _state == STATE_READY:
            _fire_on_ready_hook()
            break
        if _state == STATE_FALLBACK:
            fallback_upgrades += 1
            if fallback_upgrades >= _UPGRADE_ATTEMPTS:
                log.info(
                    "Upgrade polling window ended; staying on Edge-TTS "
                    "fallback until the next restart"
                )
                _fire_on_ready_hook()
                break
        await asyncio.sleep(_POLL_INTERVAL)


async def wait_ready(timeout: float = 180.0) -> bool:
    """Poll until a TTS engine is usable; returns False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_ready():
            return True
        kick()
        await asyncio.sleep(0.5)
    return is_ready()


async def _refresh() -> None:
    global _state

    if config.VOICE_TTS_ENGINE != "sovits":
        try:
            from voice import tts
            ok = tts.is_available()
        except Exception:
            ok = False
        _set_state(STATE_READY if ok else STATE_UNAVAILABLE)
        return

    from voice import sovits

    try:
        listening = await sovits.ensure_server()
    except Exception as e:
        log.warning(f"ensure_server failed: {e}")
        listening = False

    if listening:
        for attempt in range(_PROBE_RETRIES):
            if await sovits.probe():
                _set_state(STATE_READY)
                log.info("GPT-SoVITS engine ready")
                return
            if attempt < _PROBE_RETRIES - 1:
                await asyncio.sleep(_PROBE_RETRY_DELAY)
        log.warning("GPT-SoVITS API listening but probe synthesis failed")

    try:
        from voice import tts
        edge_ok = tts.is_available()
    except Exception:
        edge_ok = False

    if edge_ok:
        _set_state(STATE_FALLBACK)
        log.warning("GPT-SoVITS unavailable; using Edge-TTS fallback")
    else:
        _set_state(STATE_UNAVAILABLE)
        log.error("No TTS engine available (GPT-SoVITS down, Edge-TTS missing)")


def _set_state(new: str) -> None:
    global _state
    if new != _state:
        log.info(f"TTS readiness: {_state} -> {new}")
        _state = new
