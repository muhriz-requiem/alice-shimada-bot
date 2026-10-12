"""
ai/base.py
==========
Common interface every AI backend implements. Keeping this tiny is what
makes swapping Shapes.inc <-> Ollama <-> anything else in the future a
one-file change instead of a rewrite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class AIClient(ABC):
    @abstractmethod
    async def generate_reply(
        self,
        messages: list[dict],
        *,
        user_id: str,
        channel_id: str,
        max_tokens: int | None = None,
    ) -> str:
        """
        Sends a chat-style message list to the backend and returns the
        assistant's reply text. Must never raise for ordinary failure
        modes (timeouts, connection errors, rate limits) -- implementations
        should catch those, log them, and return a short in-character
        fallback string instead, so a backend hiccup never crashes the
        Discord event handler calling it.

        max_tokens optionally caps the completion (e.g. voice replies
        are 1-3 spoken sentences and pass a small budget); None means
        the backend's configured default.
        """
        raise NotImplementedError
