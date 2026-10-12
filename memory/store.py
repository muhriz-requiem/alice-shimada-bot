"""
memory/store.py
=================
SQLite-backed persistent storage: raw message log + per-user
relationship state.

Note on what got cut from the original memory.py: `get_memory`,
`get_memory_detailed`, and `get_relationship_memory` (plus their
`score_memory` helper) were never actually called anywhere in the
original bot -- `process_message` used `MemoryGraph.retrieve()`
instead. Since MemoryGraph lived only in RAM, that meant conversational
memory was silently wiped on every restart despite this database
faithfully logging every message. That's fixed here: `get_recent_messages`
lets MemoryGraph hydrate itself from this database at startup (see
memory/graph.py), so memory now actually survives a restart.

All DB calls here are synchronous by design (SQLite + a single small
file is fast enough that the overhead of async wrapping isn't worth
it) -- callers in async code should wrap calls with `asyncio.to_thread`
so the event loop is never blocked on disk I/O.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager

import config
from bot.logging_setup import get_logger

log = get_logger("memory.store")


@contextmanager
def _connect():
    conn = sqlite3.connect(config.MEMORY_DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                user_id TEXT,
                role TEXT,
                content TEXT,
                timestamp REAL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                user_id TEXT PRIMARY KEY,
                relationship TEXT,
                gift_count INTEGER DEFAULT 0
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id)")
        try:
            c.execute("ALTER TABLE relationships ADD COLUMN gift_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
    log.info(f"Memory database ready at {config.MEMORY_DB_PATH}")


def add_message(user_id: str, role: str, content: str) -> None:
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?)",
                (user_id, role, content, time.time()),
            )
    except sqlite3.Error as e:
        log.error(f"Failed to write message for user={user_id}: {e}")


def get_recent_messages(user_id: str, limit: int = 80) -> list[str]:
    """Used by MemoryGraph.hydrate() to rebuild memory after a restart."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT content FROM messages WHERE user_id=? AND role='user' "
                "ORDER BY timestamp DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [r[0] for r in reversed(rows)]
    except sqlite3.Error as e:
        log.error(f"Failed to read messages for user={user_id}: {e}")
        return []


def get_all_known_user_ids() -> list[str]:
    try:
        with _connect() as conn:
            rows = conn.execute("SELECT DISTINCT user_id FROM messages").fetchall()
        return [r[0] for r in rows]
    except sqlite3.Error as e:
        log.error(f"Failed to list known users: {e}")
        return []


def get_relationship(user_id: str) -> str:
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT relationship FROM relationships WHERE user_id=?",
                (user_id,),
            ).fetchone()
        return row[0] if row else "stranger"
    except sqlite3.Error as e:
        log.error(f"Failed to read relationship for user={user_id}: {e}")
        return "stranger"


def set_relationship(user_id: str, relationship: str) -> None:
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO relationships (user_id, relationship)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET relationship=excluded.relationship
                """,
                (user_id, relationship),
            )
    except sqlite3.Error as e:
        log.error(f"Failed to set relationship for user={user_id}: {e}")


def reset_user(user_id: str) -> None:
    """Admin utility: wipe one user's message history and relationship."""
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM relationships WHERE user_id=?", (user_id,))
        log.info(f"Reset memory for user={user_id}")
    except sqlite3.Error as e:
        log.error(f"Failed to reset user={user_id}: {e}")


def count_messages() -> int:
    try:
        with _connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        return row[0] if row else 0
    except sqlite3.Error:
        return 0


def _relationship_rank(rel: str) -> int:
    order = {
        "stranger": 0,
        "familiar": 1,
        "trusted": 2,
        "mentor-figure": 3,
        "guardian-figure": 4,
        "rival": 0,
        "anonymous": 0,
    }
    return order.get(rel, 0)


def increment_gift_count(user_id: str) -> tuple[int, str]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT gift_count, relationship FROM relationships WHERE user_id=?",
            (user_id,),
        ).fetchone()
        count = (row[0] if row else 0) + 1
        relationship = row[1] if row else "stranger"

        if count <= 2:
            new_rel = "familiar"
        elif count <= 5:
            new_rel = "trusted"
        else:
            new_rel = "trusted"

        if _relationship_rank(new_rel) > _relationship_rank(relationship):
            relationship = new_rel

        conn.execute(
            """INSERT INTO relationships (user_id, relationship, gift_count)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                 relationship=excluded.relationship,
                 gift_count=excluded.gift_count""",
            (user_id, relationship, count),
        )
    return count, relationship
