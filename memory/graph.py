"""
memory/graph.py
=================
In-memory, importance-weighted short/medium-term memory per user.
Ported from the original memory_graph.py essentially unchanged (the
scoring/decay logic was solid) with one addition: `hydrate()`, which
rebuilds each user's nodes from the SQLite message log at startup, so
memory survives a bot restart instead of resetting to empty every time.

The importance-boosting keywords (e.g. "father", "creator") are
generalized to `identity_keywords`, passed in from the active
character's data rather than hardcoded.
"""

from __future__ import annotations

import time
from collections import defaultdict

import config
from bot.logging_setup import get_logger
from memory import store

log = get_logger("memory.graph")


class MemoryNode:
    def __init__(self, text: str, importance: float = 1.0, timestamp: float | None = None):
        self.text = text
        self.importance = importance
        self.timestamp = timestamp or time.time()


class MemoryGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, list[MemoryNode]] = defaultdict(list)

    def _score_importance(self, text: str, identity_keywords: list[str] | None) -> float:
        msg = text.lower()
        importance = 1.0

        if identity_keywords and any(k.lower() in msg for k in identity_keywords):
            importance += 6.0

        if any(x in msg for x in ["sad", "happy", "angry", "frustrated", "cry"]):
            importance += 2.0

        if any(x in msg for x in ["like", "love", "hate", "favorite"]):
            importance += 1.5

        if "?" in msg:
            importance += 0.5

        return importance

    def add(self, user_id: str, text: str, identity_keywords: list[str] | None = None) -> None:
        node = MemoryNode(text=text, importance=self._score_importance(text, identity_keywords))
        self.nodes[user_id].append(node)

        cap = config.MEMORY_GRAPH_CAP_PER_USER
        if len(self.nodes[user_id]) > cap:
            self.nodes[user_id] = self.nodes[user_id][-cap:]

    def _score(self, node: MemoryNode) -> float:
        age = time.time() - node.timestamp
        decay = 1 / (1 + age / 7200)  # 2-hour scale
        return node.importance * decay

    def retrieve(self, user_id: str, limit: int = 10) -> list[str]:
        nodes = self.nodes.get(user_id, [])
        if not nodes:
            return []

        scored = sorted(((self._score(n), n) for n in nodes), key=lambda x: x[0], reverse=True)

        seen = set()
        result = []
        for _, n in scored:
            if n.text in seen:
                continue
            seen.add(n.text)
            result.append(n.text)
            if len(result) >= limit:
                break

        return result

    def get_all(self, user_id: str) -> list[dict]:
        return [
            {"text": n.text, "importance": n.importance, "age_sec": time.time() - n.timestamp}
            for n in self.nodes.get(user_id, [])
        ]

    def clear(self, user_id: str) -> None:
        self.nodes.pop(user_id, None)

    def hydrate(self, identity_keywords: list[str] | None = None) -> None:
        """
        Rebuilds memory nodes from the SQLite message log for every
        user we have history for. Called once at startup so memory
        doesn't reset to empty on every restart.
        """
        user_ids = store.get_all_known_user_ids()
        total = 0

        for user_id in user_ids:
            messages = store.get_recent_messages(user_id, limit=config.MEMORY_GRAPH_CAP_PER_USER)
            for text in messages:
                self.add(user_id, text, identity_keywords)
            total += len(messages)

        log.info(f"Hydrated memory graph: {len(user_ids)} users, {total} messages.")
