"""
ai/ollama_client.py
=====================
Optional local-inference backend, talking to a locally running Ollama
server (e.g. `qwen2.5:7b-instruct`). Not used by default -- Shapes.inc
is the default backend so the bot can run on a lightweight cloud host
with no GPU -- but kept available via AI_BACKEND=ollama for offline
development/testing without touching Shapes' rate limit.

If you do run this locally, remember to point Ollama's own model
storage away from your system drive by setting the OLLAMA_MODELS
environment variable (e.g. to D:\\OllamaModels) *before* starting the
Ollama service -- that's a system-level setting for Ollama itself, not
something this bot controls.
"""

from __future__ import annotations

import asyncio
import time

import requests

import config
from ai.base import AIClient
from ai.postprocess import FALLBACK_REPLY, clean_response
from bot.logging_setup import get_logger

log = get_logger("ai.ollama")


class OllamaClient(AIClient):
    def __init__(self) -> None:
        self._url = config.OLLAMA_HOST.rstrip("/") + "/api/chat"
        self._model = config.OLLAMA_CHAT_MODEL

    def _request(
        self, messages: list[dict], max_tokens: int | None = None
    ) -> requests.Response:
        return requests.post(
            self._url,
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": max_tokens or 150,
                    "num_ctx": 2048,
                    "top_p": 0.9,
                },
            },
            timeout=config.OLLAMA_TIMEOUT_SECONDS,
        )

    async def generate_reply(
        self,
        messages: list[dict],
        *,
        user_id: str,
        channel_id: str,
        max_tokens: int | None = None,
    ) -> str:
        start = time.monotonic()

        try:
            response = await asyncio.to_thread(
                self._request, messages, max_tokens
            )
            response.raise_for_status()

            text = response.json().get("message", {}).get("content", "")
            latency = round(time.monotonic() - start, 2)
            log.info(f"Ollama reply ok user={user_id} latency={latency}s")
            return clean_response(text, config.CHARACTER_NAME)

        except requests.exceptions.ConnectionError:
            log.error(
                f"Could not reach Ollama at {self._url}. "
                "Is the Ollama server running?"
            )
            return FALLBACK_REPLY

        except requests.exceptions.Timeout:
            log.error(f"Ollama request timed out for user={user_id}")
            return FALLBACK_REPLY

        except Exception as e:  # noqa: BLE001
            log.error(f"Ollama error: {e}")
            return FALLBACK_REPLY
