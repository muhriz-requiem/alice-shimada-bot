"""
tests/test_sovits_timeout.py
============================
Covers the SoVITS cold-start fixes in voice/sovits.py:

- a request timeout surfaces as a descriptive RuntimeError (the raw
  asyncio.TimeoutError has an empty str(), which logged as a blank
  "SoVITS TTS failed...:" message)
- warm-up/probe calls (_no_retry=True) use the longer
  VOICE_SOVITS_WARMUP_TIMEOUT so cold weight loading on a fresh server
  isn't cut off by the short per-request timeout
"""

from __future__ import annotations

import asyncio

import aiohttp
import pytest

import config
from voice import sovits


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def read(self):
        return b"\x00" * sovits._MIN_AUDIO_BYTES


class _FakeSession:
    """Minimal aiohttp.ClientSession stand-in that records constructor
    kwargs; post() returns a valid response or raises raise_on_post."""

    last_kwargs: dict = {}
    raise_on_post: Exception | None = None

    def __init__(self, *args, **kwargs):
        type(self).last_kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, *args, **kwargs):
        if type(self).raise_on_post is not None:
            raise type(self).raise_on_post
        return _FakeResponse()


@pytest.fixture()
def _healthy(monkeypatch):
    async def _true() -> bool:
        return True

    monkeypatch.setattr(sovits, "health", _true)
    monkeypatch.setattr(aiohttp, "ClientSession", _FakeSession)
    _FakeSession.last_kwargs = {}
    _FakeSession.raise_on_post = None
    yield
    _FakeSession.last_kwargs = {}
    _FakeSession.raise_on_post = None


def test_timeout_error_is_descriptive(_healthy):
    _FakeSession.raise_on_post = asyncio.TimeoutError()
    with pytest.raises(RuntimeError, match="timed out after"):
        asyncio.run(sovits.generate_async("test"))


def test_normal_request_uses_request_timeout(_healthy):
    asyncio.run(sovits.generate_async("test"))
    assert _FakeSession.last_kwargs["timeout"].total == config.VOICE_SOVITS_TIMEOUT


def test_warmup_uses_longer_timeout(_healthy):
    asyncio.run(sovits.generate_async("test", _no_retry=True))
    assert _FakeSession.last_kwargs["timeout"].total == config.VOICE_SOVITS_WARMUP_TIMEOUT
    assert _FakeSession.last_kwargs["timeout"].total > config.VOICE_SOVITS_TIMEOUT
