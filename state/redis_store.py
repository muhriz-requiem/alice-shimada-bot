"""
state/redis_store.py
=====================
Redis-backed shared state for horizontal scaling.

Provides:
- EmotionEngine state per user
- GifSelector cooldown per user
- Identity cache (optional, for invalidation)
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from typing import Any

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

import config
from bot.logging_setup import get_logger

log = get_logger("state.redis")

_REDIS_CLIENT: redis.Redis | None = None
_REDIS_AVAILABLE: bool | None = None  # None = not checked yet, True/False = cached

# Preflight probe timeout: one fast TCP connect to see if anything is
# listening. Security software with "stealth" mode (Avast, etc.)
# silently DROPS packets to closed ports instead of refusing them, so
# redis-py used to burn its internal retry stack (~48s: retries x 5s
# connect timeouts x two resolved addresses) before falling back --
# which stalled every bot startup between "Loaded character" and
# "Hydrated memory graph". A listening server answers on loopback in
# well under this, so the probe bounds the whole startup penalty.
_PREFLIGHT_TIMEOUT_LOCAL = 0.5
_PREFLIGHT_TIMEOUT_REMOTE = 3.0


def _preflight(host: str, port: int) -> None:
    """Raise OSError immediately when nothing accepts connections here."""
    timeout = (
        _PREFLIGHT_TIMEOUT_LOCAL
        if host in ("localhost", "127.0.0.1", "::1")
        else _PREFLIGHT_TIMEOUT_REMOTE
    )
    probe = socket.create_connection((host, port), timeout=timeout)
    probe.close()


def _get_redis() -> redis.Redis:
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        host = getattr(config, "REDIS_HOST", "localhost")
        port = getattr(config, "REDIS_PORT", 6379)
        try:
            _preflight(host, port)
        except OSError as e:
            log.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
            raise redis.ConnectionError(str(e)) from e
        _REDIS_CLIENT = redis.Redis(
            host=host,
            port=port,
            db=getattr(config, "REDIS_DB", 0),
            password=getattr(config, "REDIS_PASSWORD", None) or None,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            # No internal retries: the preflight already proved something
            # is listening, and redis-py's retry stack is what turned a
            # silently-dropped connect into a 48s startup stall.
            retry=Retry(NoBackoff(), 0),
            health_check_interval=0,
        )
        # Test connection once
        try:
            _REDIS_CLIENT.ping()
            log.info("Redis connection established")
        except redis.ConnectionError as e:
            log.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
            _REDIS_CLIENT = None
            raise
    return _REDIS_CLIENT


def is_redis_available() -> bool:
    """Check Redis availability (cached after first check)."""
    global _REDIS_AVAILABLE
    if _REDIS_AVAILABLE is None:
        try:
            _get_redis()
            _REDIS_AVAILABLE = True
        except Exception:
            _REDIS_AVAILABLE = False
    return _REDIS_AVAILABLE


async def is_redis_available_async() -> bool:
    """Async version - runs check in thread pool to avoid blocking event loop."""
    return await asyncio.to_thread(is_redis_available)


def _run_async(redis_func, *args, **kwargs):
    """Run a blocking Redis call in a thread pool."""
    return asyncio.to_thread(redis_func, *args, **kwargs)


# =========================================================
# Emotion Engine State
# =========================================================

_EMOTION_PREFIX = "emotion:"
_EMOTION_TTL = 86400 * 7  # 7 days


def save_emotion_state(user_id: str, state: dict[str, float]) -> bool:
    """Save emotion state to Redis (blocking)."""
    if not is_redis_available():
        return False
    try:
        key = f"{_EMOTION_PREFIX}{user_id}"
        _get_redis().setex(key, _EMOTION_TTL, json.dumps(state))
        return True
    except Exception as e:
        log.error(f"Failed to save emotion state for {user_id}: {e}")
        return False


async def save_emotion_state_async(user_id: str, state: dict[str, float]) -> bool:
    """Save emotion state to Redis (async)."""
    return await _run_async(save_emotion_state, user_id, state)


def load_emotion_state(user_id: str) -> dict[str, float] | None:
    """Load emotion state from Redis (blocking)."""
    if not is_redis_available():
        return None
    try:
        key = f"{_EMOTION_PREFIX}{user_id}"
        data = _get_redis().get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        log.error(f"Failed to load emotion state for {user_id}: {e}")
        return None


async def load_emotion_state_async(user_id: str) -> dict[str, float] | None:
    """Load emotion state from Redis (async)."""
    return await _run_async(load_emotion_state, user_id)


def delete_emotion_state(user_id: str) -> bool:
    """Delete emotion state from Redis (blocking)."""
    if not is_redis_available():
        return False
    try:
        key = f"{_EMOTION_PREFIX}{user_id}"
        _get_redis().delete(key)
        return True
    except Exception as e:
        log.error(f"Failed to delete emotion state for {user_id}: {e}")
        return False


async def delete_emotion_state_async(user_id: str) -> bool:
    """Delete emotion state from Redis (async)."""
    return await _run_async(delete_emotion_state, user_id)


# =========================================================
# GIF Selector Cooldown
# =========================================================

_GIF_PREFIX = "gif_cooldown:"
_GIF_TTL = 86400  # 1 day


def set_gif_cooldown(user_id: str, timestamp: float | None = None) -> bool:
    """Set GIF cooldown timestamp for user (blocking)."""
    if not is_redis_available():
        return False
    try:
        key = f"{_GIF_PREFIX}{user_id}"
        ts = timestamp or time.time()
        _get_redis().setex(key, _GIF_TTL, str(ts))
        return True
    except Exception as e:
        log.error(f"Failed to set GIF cooldown for {user_id}: {e}")
        return False


async def set_gif_cooldown_async(user_id: str, timestamp: float | None = None) -> bool:
    """Set GIF cooldown timestamp for user (async)."""
    return await _run_async(set_gif_cooldown, user_id, timestamp)


def get_gif_cooldown(user_id: str) -> float | None:
    """Get GIF cooldown timestamp for user (blocking)."""
    if not is_redis_available():
        return None
    try:
        key = f"{_GIF_PREFIX}{user_id}"
        data = _get_redis().get(key)
        if data:
            return float(data)
        return None
    except Exception as e:
        log.error(f"Failed to get GIF cooldown for {user_id}: {e}")
        return None


async def get_gif_cooldown_async(user_id: str) -> float | None:
    """Get GIF cooldown timestamp for user (async)."""
    return await _run_async(get_gif_cooldown, user_id)


def check_gif_cooldown(user_id: str, cooldown_seconds: float) -> bool:
    """Check if user is on GIF cooldown (blocking)."""
    last = get_gif_cooldown(user_id)
    if last is None:
        return False
    return (time.time() - last) < cooldown_seconds


async def check_gif_cooldown_async(user_id: str, cooldown_seconds: float) -> bool:
    """Check if user is on GIF cooldown (async)."""
    last = await get_gif_cooldown_async(user_id)
    if last is None:
        return False
    return (time.time() - last) < cooldown_seconds


# =========================================================
# Identity Cache
# =========================================================

_IDENTITY_PREFIX = "identity:"
_IDENTITY_TTL = 3600  # 1 hour


def cache_identity(user_id: str, identity: dict[str, Any]) -> bool:
    """Cache identity in Redis (blocking)."""
    if not is_redis_available():
        return False
    try:
        key = f"{_IDENTITY_PREFIX}{user_id}"
        _get_redis().setex(key, _IDENTITY_TTL, json.dumps(identity))
        return True
    except Exception as e:
        log.error(f"Failed to cache identity for {user_id}: {e}")
        return False


async def cache_identity_async(user_id: str, identity: dict[str, Any]) -> bool:
    """Cache identity in Redis (async)."""
    return await _run_async(cache_identity, user_id, identity)


def get_cached_identity(user_id: str) -> dict[str, Any] | None:
    """Get cached identity from Redis (blocking)."""
    if not is_redis_available():
        return None
    try:
        key = f"{_IDENTITY_PREFIX}{user_id}"
        data = _get_redis().get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        log.error(f"Failed to get cached identity for {user_id}: {e}")
        return None


async def get_cached_identity_async(user_id: str) -> dict[str, Any] | None:
    """Get cached identity from Redis (async)."""
    return await _run_async(get_cached_identity, user_id)


def invalidate_identity_cache(user_id: str) -> bool:
    """Invalidate identity cache for user (blocking)."""
    if not is_redis_available():
        return False
    try:
        key = f"{_IDENTITY_PREFIX}{user_id}"
        _get_redis().delete(key)
        return True
    except Exception as e:
        log.error(f"Failed to invalidate identity cache for {user_id}: {e}")
        return False


async def invalidate_identity_cache_async(user_id: str) -> bool:
    """Invalidate identity cache for user (async)."""
    return await _run_async(invalidate_identity_cache, user_id)


# =========================================================
# Health Check
# =========================================================

def check_redis_health() -> dict[str, Any]:
    """Check Redis connectivity and return status (blocking)."""
    try:
        client = _get_redis()
        info = client.info("memory")
        return {
            "status": "healthy",
            "connected": True,
            "memory_used_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
        }


async def check_redis_health_async() -> dict[str, Any]:
    """Check Redis connectivity and return status (async)."""
    return await _run_async(check_redis_health)
