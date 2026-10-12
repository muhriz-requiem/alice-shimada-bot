"""
tests/test_redis_preflight.py
==============================
Covers the startup stall fix: security software with port-stealth
(Avast et al.) silently DROPS packets to closed loopback ports instead
of refusing them, so redis-py's connect + internal retries hung for
~48s at every bot startup before falling back to in-memory state.

- a fast preflight socket probe bounds the whole check (0.5s local)
- probe failure -> immediate in-memory fallback, redis.Redis never built
- probe success + ping failure -> fallback, client cleared
- availability result is cached after the first check
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
import redis

import state.redis_store as rs


@pytest.fixture(autouse=True)
def _reset_redis_state():
    """Each test starts from the not-yet-checked state."""
    rs._REDIS_CLIENT = None
    rs._REDIS_AVAILABLE = None
    yield
    rs._REDIS_CLIENT = None
    rs._REDIS_AVAILABLE = None


def test_preflight_fails_fast_and_skips_client(monkeypatch, caplog):
    calls = {}

    def fake_create_connection(address, timeout=None, **kwargs):
        calls["address"] = address
        calls["timeout"] = timeout
        raise ConnectionRefusedError(10061, "actively refused")

    monkeypatch.setattr(rs.socket, "create_connection", fake_create_connection)

    def bomb(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("redis.Redis must not be built after failed preflight")

    monkeypatch.setattr(rs.redis, "Redis", bomb)

    import logging
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="state.redis"):
        available = rs.is_redis_available()
    elapsed = time.monotonic() - started

    assert available is False
    assert elapsed < 1.5  # was ~48s before the preflight
    assert calls["address"] == ("localhost", 6379)
    assert calls["timeout"] == rs._PREFLIGHT_TIMEOUT_LOCAL
    assert "Using in-memory fallback" in caplog.text
    assert rs._REDIS_CLIENT is None


def test_preflight_uses_longer_timeout_for_remote_host(monkeypatch):
    monkeypatch.setattr(rs.config, "REDIS_HOST", "redis.example.com")
    seen = {}

    def fake_create_connection(address, timeout=None, **kwargs):
        seen["timeout"] = timeout
        raise OSError("no route")

    monkeypatch.setattr(rs.socket, "create_connection", fake_create_connection)
    assert rs.is_redis_available() is False
    assert seen["timeout"] == rs._PREFLIGHT_TIMEOUT_REMOTE


def test_preflight_success_ping_failure_falls_back(monkeypatch):
    def ok_probe(address, timeout=None, **kwargs):
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(rs.socket, "create_connection", ok_probe)

    class BrokenRedis:
        def __init__(self, **kwargs):
            pass

        def ping(self):
            raise redis.ConnectionError("ping failed")

    monkeypatch.setattr(rs.redis, "Redis", BrokenRedis)

    assert rs.is_redis_available() is False
    assert rs._REDIS_CLIENT is None


def test_healthy_path_caches_availability(monkeypatch):
    probes = []

    def ok_probe(address, timeout=None, **kwargs):
        probes.append(address)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(rs.socket, "create_connection", ok_probe)

    class HealthyRedis:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def ping(self):
            return True

    monkeypatch.setattr(rs.redis, "Redis", HealthyRedis)

    assert rs.is_redis_available() is True
    assert rs.is_redis_available() is True  # cached -- no second probe
    assert len(probes) == 1
    # Internal retries disabled so a hang can never re-stack.
    assert isinstance(rs._REDIS_CLIENT.kwargs["retry"], rs.Retry)


def test_save_without_redis_returns_false_without_error(monkeypatch):
    def fail_probe(address, timeout=None, **kwargs):
        raise ConnectionRefusedError(10061, "actively refused")

    monkeypatch.setattr(rs.socket, "create_connection", fail_probe)
    assert rs.save_emotion_state("user1", {"joy": 0.5}) is False
    assert rs.get_gif_cooldown("user1") is None
    assert rs.cache_identity("user1", {"name": "x"}) is False


def test_health_check_reports_unhealthy_when_preflight_fails(monkeypatch):
    def fail_probe(address, timeout=None, **kwargs):
        raise ConnectionRefusedError(10061, "actively refused")

    monkeypatch.setattr(rs.socket, "create_connection", fail_probe)
    health = rs.check_redis_health()
    assert health["status"] == "unhealthy"
    assert health["connected"] is False
