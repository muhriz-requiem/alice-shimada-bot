"""
social/identities.py
======================
Discord-user identity registry: maps a Discord user ID to how the
character should recognize them (name, role, relationship). This is
what drives things like the father-roleplay behavior for a specific
Discord ID -- it lives here, separate from any one character's data,
because it's about *who the humans are*, not about the character.

This data is intentionally kept outside the AI backend entirely (it's
injected by our own prompt/context code -- see character/prompt_builder.py)
since neither Shapes.inc's dashboard config nor a local Ollama model has
any way to know which Discord account is which person.

Replaces the original identity.py (hardcoded dict + inline lookup) and
the never-used identity_firewall.py wrapper class around it.

Now supports Redis-backed caching for horizontal scaling.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from bot.logging_setup import get_logger
from state import (
    cache_identity,
    cache_identity_async,
    get_cached_identity,
    get_cached_identity_async,
    is_redis_available,
)

log = get_logger("social.identities")

_PATH = Path(__file__).resolve().parent / "data" / "identities.json"

_UNKNOWN = {
    "name": "Unknown",
    "role": "User",
    "relationship": None,
    "auto_relationship": None,
    "description": "No stored identity for this user.",
    "is_authoritative": False,
}


def _load() -> dict:
    if not _PATH.exists():
        log.warning(f"No identities.json found at {_PATH}; all users will be Unknown.")
        return {}
    try:
        with _PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.error(f"Failed to read {_PATH}: {e}")
        return {}


_IDENTITIES = _load()
_REDIS_ENABLED = is_redis_available()


def get_identity(user_id: str) -> dict:
    user_id = str(user_id).strip()

    # Try Redis cache first
    if _REDIS_ENABLED:
        cached = get_cached_identity(user_id)
        if cached is not None:
            return cached

    # Fallback to in-memory
    entry = _IDENTITIES.get(user_id)

    if entry is None:
        return dict(_UNKNOWN)

    result = {
        "name": entry.get("name", "Unknown"),
        "role": entry.get("role", "User"),
        "relationship": entry.get("relationship"),
        "auto_relationship": entry.get("auto_relationship"),
        "description": entry.get("description", "No description."),
        "is_authoritative": True,
    }

    # Cache in Redis
    if _REDIS_ENABLED:
        cache_identity(user_id, result)

    return result


async def get_identity_async(user_id: str) -> dict:
    """Async version of get_identity."""
    user_id = str(user_id).strip()

    # Try Redis cache first
    if _REDIS_ENABLED:
        cached = await get_cached_identity_async(user_id)
        if cached is not None:
            return cached

    # Fallback to in-memory
    entry = _IDENTITIES.get(user_id)

    if entry is None:
        return dict(_UNKNOWN)

    result = {
        "name": entry.get("name", "Unknown"),
        "role": entry.get("role", "User"),
        "relationship": entry.get("relationship"),
        "auto_relationship": entry.get("auto_relationship"),
        "description": entry.get("description", "No description."),
        "is_authoritative": True,
    }

    # Cache in Redis (fire and forget)
    if _REDIS_ENABLED:
        asyncio.create_task(cache_identity_async(user_id, result))

    return result


def reload() -> None:
    """Re-reads identities.json without restarting the bot."""
    global _IDENTITIES
    _IDENTITIES = _load()
    log.info(f"Reloaded {len(_IDENTITIES)} identities.")

    # Invalidate Redis cache
    if _REDIS_ENABLED:
        # Note: In production, you'd want to use SCAN to find all keys
        # For now, we rely on TTL expiration
        pass
