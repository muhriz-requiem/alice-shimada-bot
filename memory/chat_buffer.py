"""
memory/chat_buffer.py
=======================
Short rolling in-memory conversation buffer per user (last N turns),
used to give the AI backend immediate conversational continuity on top
of whatever longer-term memory it has. Ephemeral by design -- resets on
restart, same as the original bot.py's chat_buffer dict.
"""

from __future__ import annotations

from collections import defaultdict, deque

import config


class ChatBuffer:
    def __init__(self, maxlen: int | None = None) -> None:
        self._maxlen = maxlen or config.CHAT_HISTORY_LENGTH
        self._buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=self._maxlen))

    def push(self, user_id: str, role: str, content: str) -> None:
        self._buffers[user_id].append({"role": role, "content": content})

    def history(self, user_id: str) -> list[dict]:
        return list(self._buffers.get(user_id, []))

    def clear(self, user_id: str) -> None:
        self._buffers.pop(user_id, None)
