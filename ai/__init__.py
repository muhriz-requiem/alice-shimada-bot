"""
ai package
==========
Backend-agnostic AI text generation. `get_ai_client()` returns whichever
backend is configured (Shapes.inc cloud API, Groq cloud API, or a local
Ollama server) behind the same small interface, so the rest of the bot
never needs to know which one is in use.
"""

from __future__ import annotations

import config
from ai.base import AIClient
from ai.groq_client import GroqClient
from ai.ollama_client import OllamaClient
from ai.shapes_client import ShapesClient

_client_instance: AIClient | None = None


def get_ai_client() -> AIClient:
    """Returns a singleton AI client chosen by config.AI_BACKEND."""
    global _client_instance

    if _client_instance is not None:
        return _client_instance

    if config.AI_BACKEND == "ollama":
        _client_instance = OllamaClient()
    elif config.AI_BACKEND == "groq":
        _client_instance = GroqClient()
    else:
        _client_instance = ShapesClient()

    return _client_instance


__all__ = ["AIClient", "ShapesClient", "GroqClient", "OllamaClient", "get_ai_client"]
