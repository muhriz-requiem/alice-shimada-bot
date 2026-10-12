"""
ai/shapes_client.py
====================
AI backend that runs inference on Shapes.inc's cloud API instead of a
local model. This is what lets the bot run on a lightweight host with
no GPU and no local Ollama server -- the only thing the bot itself does
is make a small HTTPS call per message.

Shapes exposes an OpenAI-compatible /chat/completions endpoint:
  base_url = "https://api.shapes.inc/v1/"
  model    = "shapesinc/<your-shape-username>"

Two details matter a lot here:

1. X-User-Id / X-Channel-Id headers. By default Shapes treats every
   caller as the single account that owns the API key, which would
   collapse every Discord user into one shared identity/memory on
   Shapes' side. Setting these headers per-request keeps each Discord
   user's (and each channel's) context separate.

2. Rate limiting. Shapes' default limit is 5-20 requests/minute. This
   client self-throttles to stay under config.SHAPES_RATE_LIMIT_RPM so
   a burst of messages queues briefly instead of erroring out.
"""

from __future__ import annotations

import asyncio
import time

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError

import config
from ai.base import AIClient
from ai.postprocess import FALLBACK_REPLY, clean_response
from bot.logging_setup import get_logger

log = get_logger("ai.shapes")


class ShapesClient(AIClient):
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=config.SHAPES_BASE_URL,
            api_key=config.SHAPES_API_KEY or "missing-key",
            timeout=config.SHAPES_TIMEOUT_SECONDS,
        )
        self._model = config.SHAPES_MODEL

        # Simple self-throttle: enforce a minimum gap between outgoing
        # requests so we stay under Shapes' requests-per-minute limit
        # even when several users message at once.
        self._min_interval = 60.0 / max(config.SHAPES_RATE_LIMIT_RPM, 1)
        self._lock = asyncio.Lock()
        self._last_call = 0.0

        if not config.SHAPES_API_KEY or not config.SHAPES_SHAPE_USERNAME:
            log.warning(
                "Shapes backend selected but SHAPESINC_API_KEY / "
                "SHAPESINC_SHAPE_USERNAME is missing -- requests will fail."
            )

    async def _throttle(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    async def generate_reply(
        self,
        messages: list[dict],
        *,
        user_id: str,
        channel_id: str,
        max_tokens: int | None = None,
    ) -> str:
        last_error: Exception | None = None

        for attempt in range(config.SHAPES_MAX_RETRIES + 1):
            await self._throttle()
            start = time.monotonic()

            try:
                request: dict = {
                    "model": self._model,
                    "messages": messages,
                    "extra_headers": {
                        "X-User-Id": str(user_id),
                        "X-Channel-Id": str(channel_id),
                    },
                }
                if max_tokens is not None:
                    request["max_tokens"] = max_tokens
                response = await self._client.chat.completions.create(**request)

                latency = round(time.monotonic() - start, 2)
                text = response.choices[0].message.content or ""
                log.info(f"Shapes reply ok user={user_id} latency={latency}s")
                return clean_response(text, config.CHARACTER_NAME)

            except RateLimitError as e:
                last_error = e
                backoff = 2 ** attempt
                log.warning(
                    f"Shapes rate limited (attempt {attempt + 1}/"
                    f"{config.SHAPES_MAX_RETRIES + 1}), backing off {backoff}s"
                )
                await asyncio.sleep(backoff)

            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                log.warning(
                    f"Shapes connection issue (attempt {attempt + 1}/"
                    f"{config.SHAPES_MAX_RETRIES + 1}): {e}"
                )
                await asyncio.sleep(1.5 * (attempt + 1))

            except APIError as e:
                last_error = e
                log.error(f"Shapes API error: {e}")
                break

            except Exception as e:  # noqa: BLE001 - never let this crash the caller
                last_error = e
                log.error(f"Unexpected Shapes client error: {e}")
                break

        log.error(f"Shapes request failed for user={user_id}: {last_error}")
        return FALLBACK_REPLY
