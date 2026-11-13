"""
tests/test_discord_service.py
=============================
Phase 4, in-process Discord service (`desktop/discord_service.py`) --
lifecycle logic without a real token or network:

- start/stop round trip: the fake bot's start() blocks until close(),
  so a passing stop proves the tray's stop signal actually reaches it.
- Bot crash surfaces to the tray (done-callback) and the session
  thread cannot hang on a dead run task.
- Crash state survives stop() (never masked as a clean "off").
"""

from __future__ import annotations

import asyncio
import time

import pytest
from dotenv import load_dotenv

load_dotenv()

import config
from desktop.discord_service import DiscordService


class _FakeBot:
    """AliceBot stand-in: start() blocks until close() (like discord.py)."""

    def __init__(self) -> None:
        self.closed = False

    async def start(self, token: str) -> None:  # noqa: ARG002 - fake
        while not self.closed:
            await asyncio.sleep(0.01)

    async def close(self) -> None:
        self.closed = True


class _BoomBot(_FakeBot):
    """AliceBot stand-in that dies immediately (bad token etc.)."""

    async def start(self, token: str) -> None:  # noqa: ARG002 - fake
        raise RuntimeError("boom: login failed")


async def _fake_health(*_a, **_k) -> None:
    return None


@pytest.fixture(autouse=True)
def _patch_bot_stack(monkeypatch):
    # Never use the real token or validate the whole deployment config.
    monkeypatch.setattr(config, "DISCORD_TOKEN", "x" * 71, raising=False)
    monkeypatch.setattr(config, "validate", lambda: None, raising=False)
    # Import the real modules so the lazy `from X import Y` inside
    # _bot_coro resolves to our fakes, not to discord.py/uvicorn.
    import bot.client  # noqa: F401

    monkeypatch.setattr("bot.client.AliceBot", _FakeBot)
    import health  # noqa: F401

    monkeypatch.setattr("health.start_health_server", _fake_health)
    yield


def _wait_state(svc: DiscordService, state: str, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if svc.state == state:
            return True
        time.sleep(0.05)
    return False


def test_start_stop_roundtrip():
    svc = DiscordService()
    assert svc.start() is True
    # Duplicate start while running/starting is refused.
    assert svc.start() is False
    assert _wait_state(svc, "on"), f"state={svc.state} error={svc.error}"

    svc.stop(timeout=15.0)
    assert svc.state == "off"
    assert svc._thread is None


def test_stop_before_start_is_noop():
    svc = DiscordService()
    svc.stop()  # never started: must not raise or hang
    assert svc.state == "off"


def test_bot_crash_surfaces_and_thread_exits():
    import bot.client as bot_client

    svc = DiscordService()
    monkey_bot = _BoomBot
    # Patch just this test's fixture swap (fixture set _FakeBot).
    bot_client.AliceBot = monkey_bot
    try:
        assert svc.start() is True
        assert _wait_state(svc, "error"), f"state={svc.state} error={svc.error}"
        assert svc.error and "boom" in svc.error
    finally:
        bot_client.AliceBot = _FakeBot
        svc.stop(timeout=10.0)
    assert svc.state == "error"  # crash survives stop(), not masked "off"


def test_stop_reachable_after_late_crash():
    """A crash while blocked on the stop event must not strand the thread."""
    import bot.client as bot_client

    class _LateBoom(_FakeBot):
        async def start(self, token: str) -> None:  # noqa: ARG002
            await asyncio.sleep(0.2)
            raise RuntimeError("late boom")

    bot_client.AliceBot = _LateBoom
    try:
        svc = DiscordService()
        assert svc.start() is True
        # Don't wait for the crash: click stop immediately (like a user).
        t0 = time.time()
        svc.stop(timeout=10.0)
        elapsed = time.time() - t0
        assert elapsed < 12.0, f"stop() hung for {elapsed:.1f}s"
        assert svc.state == "error"
        assert svc.error and "late boom" in svc.error
    finally:
        bot_client.AliceBot = _FakeBot
