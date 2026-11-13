"""
ai/groq_client.py
==================
AI backend that runs inference on Groq's cloud API instead of Shapes.inc
or a local Ollama server. Like Shapes, this is cloud inference -- the
bot itself stays a lightweight host with no GPU required.

Groq exposes an OpenAI-compatible /chat/completions endpoint:
  base_url = "https://api.groq.com/openai/v1"
  model    = e.g. "llama-3.3-70b-versatile"

Unlike Shapes, Groq has no per-user/per-channel identity concept and no
dashboard-configured personality -- it's a plain instruct model. That
means config.INJECT_FULL_CHARACTER_PROMPT should be True when this
backend is selected, so Alice's full lore/identity/style-example block
is built and sent on every request (the same way it already is for the
Ollama backend). config.py handles this by including "groq" in the
same default condition as "ollama".

Groq is fast, but still has per-plan rate limits (requests/minute and
tokens/minute), so this client self-throttles the same way ShapesClient
does: a configurable requests-per-minute gap between calls plus a
rolling 60s tokens-per-minute budget gate. Rate-limit waits that would
outlast the conversation fail fast and return the fallback instead of
queueing replies for minutes (as seen in a 3-user voice session where
429 retries stacked for 40s+ and spammed the channel with fallbacks).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError

import config
from ai.base import AIClient
from ai.postprocess import FALLBACK_REPLY, clean_response
from bot.logging_setup import get_logger

log = get_logger("ai.groq")

# Rolling window for the TPM gate (seconds) -- matches Groq's per-minute
# accounting.
_TPM_WINDOW_SECONDS = 60.0
# Only fire a new request while the window stays under this fraction of
# GROQ_TPM_LIMIT, leaving headroom for the request itself.
_TPM_TARGET_FRACTION = 0.9


class ChatStream:
    """
    Handle over a streaming completion for the live voice path.

    `generate_reply_stream` establishes the HTTP stream (same throttle
    and retry gates as generate_reply) and hands back this object; the
    caller consumes raw text deltas with iter_chunks() while the reply
    is still generating, then applies postprocess cleaning itself.

    State exposed while iterating: ``first_token_at`` (monotonic clock,
    set on the first visible delta), ``finished`` (stream ended),
    ``error`` (set when the connection dropped mid-stream -- deltas up
    to that point are still valid). Usage accounting happens through
    the on_usage callback fed with the final usage chunk
    (stream_options={"include_usage": True}).
    """

    __slots__ = (
        "_chunks", "_on_usage", "_user_id", "_t0",
        "first_token_at", "finished", "error", "coalesce",
    )

    def __init__(self, chunks, *, on_usage=None, user_id: str = "", t0: float = 0.0) -> None:
        self._chunks = chunks
        self._on_usage = on_usage
        self._user_id = user_id
        self._t0 = t0
        self.first_token_at: float | None = None
        self.finished: bool = False
        self.error: Exception | None = None
        self.coalesce: list[str] = []  # raw text, for callers that want it all

    def iter_chunks(self):
        return self._iter()

    async def _iter(self):
        try:
            async for chunk in self._chunks:
                usage = getattr(chunk, "usage", None)
                if usage is not None and self._on_usage is not None:
                    self._on_usage(chunk)
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                text = getattr(delta, "content", None)
                if not text:
                    continue
                if self.first_token_at is None:
                    self.first_token_at = time.monotonic()
                self.coalesce.append(text)
                yield text
        except Exception as e:  # noqa: BLE001 - mid-stream drop isn't fatal
            self.error = e
            log.warning(
                f"Groq stream interrupted after {len(self.coalesce)} "
                f"deltas for user={self._user_id}: {e}"
            )
        finally:
            self.finished = True
            if self.error is None and not self.coalesce:
                log.warning(
                    f"Groq stream produced no visible content for "
                    f"user={self._user_id} (latency="
                    f"{time.monotonic() - self._t0:.2f}s) -- "
                    "reasoning likely consumed the budget"
                )

    async def text(self) -> str:
        """Drain the whole stream and return the raw reply text."""
        parts: list[str] = []
        async for delta in self.iter_chunks():
            parts.append(delta)
        return "".join(parts)




class GroqClient(AIClient):
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=config.GROQ_BASE_URL,
            api_key=config.GROQ_API_KEY or "missing-key",
            timeout=config.GROQ_TIMEOUT_SECONDS,
            # The SDK retries on its own (2 extra attempts with its own
            # backoff) *on top of* this class's retry loop. In production
            # that stacking held 429s for 14-24s per attempt until
            # requests piled up for minutes. All retry policy lives in
            # generate_reply below.
            max_retries=0,
        )
        # Force the SDK's lazy resource import (openai.resources.* plus
        # the jiter native module) here at boot. Left lazy, the first
        # generate_reply pays it synchronously on the event loop -- under
        # Defender/disk contention that took ~55s once and blocked voice
        # heartbeats long enough to drop the voice connection.
        _warm_t0 = time.monotonic()
        self._client.chat  # noqa: B018 - warm the lazy import, discard
        log.info(f"openai SDK warm-up: {time.monotonic() - _warm_t0:.1f}s")
        self._model = config.GROQ_MODEL

        # Simple self-throttle: enforce a minimum gap between outgoing
        # requests so bursts queue instead of erroring against Groq's
        # requests-per-minute limit.
        self._min_interval = 60.0 / max(config.GROQ_RATE_LIMIT_RPM, 1)
        self._lock = asyncio.Lock()
        self._last_call = 0.0

        # Rolling TPM usage: (monotonic timestamp, billable tokens).
        # Groq enforces tokens-per-minute per organisation; overshooting
        # returns 429. Cached prompt-prefix tokens do not count toward
        # Groq's TPM, so only non-cached tokens are recorded here.
        self._tpm_window: deque[tuple[float, int]] = deque()

        if not config.GROQ_API_KEY:
            log.warning(
                "Groq backend selected but GROQ_API_KEY is missing -- "
                "requests will fail."
            )

    async def _throttle(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    def _prune_tpm(self) -> None:
        cutoff = time.monotonic() - _TPM_WINDOW_SECONDS
        while self._tpm_window and self._tpm_window[0][0] < cutoff:
            self._tpm_window.popleft()

    def _record_tpm(self, response) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        total = getattr(usage, "total_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        billable = max(int(total) - int(cached), 0)
        self._tpm_window.append((time.monotonic(), billable))
        log.debug(
            f"Groq tokens recorded: total={total} cached={cached} "
            f"billable={billable}"
        )

    async def _tpm_budget_ok(self) -> bool:
        """True when a request fits inside the rolling TPM window.

        Waits while budget frees up as old usage ages out of the
        window, but returns False instead of blocking longer than the
        configured max wait -- a voice reply that arrives 20s after its
        utterance is worse than no reply at all.
        """
        while True:
            self._prune_tpm()
            used = sum(t for _, t in self._tpm_window)
            if used <= config.GROQ_TPM_LIMIT * _TPM_TARGET_FRACTION:
                return True
            oldest_ts, _ = self._tpm_window[0]
            wait = _TPM_WINDOW_SECONDS - (time.monotonic() - oldest_ts)
            if wait <= 0:
                self._tpm_window.popleft()
                continue
            if wait > config.GROQ_RETRY_MAX_WAIT:
                return False
            await asyncio.sleep(wait)

    @staticmethod
    def _retry_after_seconds(e: RateLimitError) -> float | None:
        """Seconds Groq asked us to wait, from the 429 response headers."""
        try:
            raw = e.response.headers.get("retry-after")
            if raw is None:
                return None
            return max(float(raw), 0.0)
        except (AttributeError, TypeError, ValueError):
            return None

    async def vision_chat(
        self,
        messages: list[dict],
        *,
        user_id: str,
        channel_id: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.5,
    ) -> str:
        """One multimodal (image + text) completion. Used by the screen
        glance feature; shares the RPM throttle and the rolling TPM
        window with text chat so glances can't stampede the free tier.

        Returns the reply text; raises on failure (caller decides).
        Vision models are never gpt-oss -> no reasoning_effort knob, and
        no stop-sequence guard (those tokens belong to the chat model).
        """
        use_model = model or config.SCREEN_LOOK_MODEL
        await self._throttle()
        start = time.monotonic()
        response = await self._client.chat.completions.create(
            model=use_model,
            messages=messages,
            max_tokens=max_tokens or 120,
            temperature=temperature,
        )
        self._record_tpm(response)
        text = (response.choices[0].message.content or "").strip()
        log.info(
            f"Groq vision ok model={use_model} user={user_id} "
            f"({time.monotonic() - start:.1f}s, "
            f"{len(text)} chars)"
        )
        return text

    async def generate_reply(
        self,
        messages: list[dict],
        *,
        user_id: str,
        channel_id: str,
        max_tokens: int | None = None,
    ) -> str:
        last_error: Exception | None = None
        # One reply never spends more than ~2x the max single wait in
        # backoff; beyond that the conversation has moved on.
        wait_budget = config.GROQ_RETRY_MAX_WAIT * 2
        wait_spent = 0.0

        for attempt in range(config.GROQ_MAX_RETRIES + 1):
            if not await self._tpm_budget_ok():
                log.info(
                    f"Groq TPM budget exhausted "
                    f"(limit={config.GROQ_TPM_LIMIT}/min); dropping "
                    f"request for user={user_id}"
                )
                return FALLBACK_REPLY

            await self._throttle()
            start = time.monotonic()

            # gpt-oss spends hidden reasoning tokens before any visible
            # content, and reasoning counts against max_tokens -- a
            # short voice budget could be emptied before a single word
            # was generated (empty content = silent voice / fallback).
            # Groq accepts this knob only on gpt-oss 20B/120B, so skip
            # other models rather than risk a 400.
            effort = config.GROQ_REASONING_EFFORT.strip()
            extra: dict = {}
            if effort and "gpt-oss" in self._model:
                extra["reasoning_effort"] = effort

            try:
                response = await self._client.chat.completions.create(
                    **extra,
                    model=self._model,
                    messages=messages,
                    max_tokens=max_tokens or config.GROQ_MAX_TOKENS,
                    # If the model starts emitting a raw completion
                    # artifact instead of staying in character, stop
                    # generation immediately rather than letting it run
                    # on into leaked training-data text (seen in
                    # practice: "<|endoftext|>" followed by an unrelated
                    # URL and assistant-speak). ai/postprocess.py is the
                    # second line of defense if this ever slips through.
                    stop=["<|endoftext|>", "<|im_end|>"],
                )

                latency = round(time.monotonic() - start, 2)
                choice = response.choices[0]
                text = choice.message.content or ""
                reasoning = getattr(choice.message, "reasoning", None) or ""
                log.debug(f"Groq RAW reply for user={user_id}: {text!r}")
                # Record before any early return: reasoning tokens are
                # billable even when no visible content was produced.
                self._record_tpm(response)
                if not text.strip():
                    log.warning(
                        f"Groq returned empty content for user={user_id} "
                        f"(finish={getattr(choice, 'finish_reason', '?')}, "
                        f"reasoning_chars={len(reasoning)}, "
                        f"max_tokens={max_tokens or config.GROQ_MAX_TOKENS}) "
                        "-- reasoning likely consumed the budget; using fallback"
                    )
                    return FALLBACK_REPLY
                cleaned = clean_response(text, config.CHARACTER_NAME)
                if not cleaned.strip() or cleaned == FALLBACK_REPLY:
                    # clean_response masks empty/unusable text as the
                    # fallback; Discord hard-rejects an empty message
                    # with a 400, so normalize here and let callers
                    # decide (voice stays silent, text chat shows it).
                    log.warning(f"Groq reply unusable after cleaning for user={user_id}; using fallback.")
                    return FALLBACK_REPLY
                log.info(f"Groq reply ok user={user_id} latency={latency}s")
                return cleaned

            except RateLimitError as e:
                last_error = e
                hinted = self._retry_after_seconds(e)
                backoff = hinted if hinted is not None else 2 ** attempt
                if backoff > config.GROQ_RETRY_MAX_WAIT or wait_spent + backoff > wait_budget:
                    log.info(
                        f"Groq rate limited (attempt {attempt + 1}/"
                        f"{config.GROQ_MAX_RETRIES + 1}) with retry-after "
                        f"{backoff:.1f}s > max "
                        f"{config.GROQ_RETRY_MAX_WAIT:.1f}s -- failing fast"
                    )
                    break
                log.warning(
                    f"Groq rate limited (attempt {attempt + 1}/"
                    f"{config.GROQ_MAX_RETRIES + 1}), backing off {backoff}s"
                )
                wait_spent += backoff
                await asyncio.sleep(backoff)

            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                log.warning(
                    f"Groq connection issue (attempt {attempt + 1}/"
                    f"{config.GROQ_MAX_RETRIES + 1}): {e}"
                )
                await asyncio.sleep(1.5 * (attempt + 1))

            except APIError as e:
                last_error = e
                log.error(f"Groq API error: {e}")
                break

            except Exception as e:  # noqa: BLE001 - never let this crash the caller
                last_error = e
                log.error(f"Unexpected Groq client error: {e}")
                break

        log.error(f"Groq request failed for user={user_id}: {last_error}")
        return FALLBACK_REPLY

    async def generate_reply_stream(
        self,
        messages: list[dict],
        *,
        user_id: str,
        channel_id: str,
        max_tokens: int | None = None,
    ) -> ChatStream | None:
        """
        Streaming variant of generate_reply for the live voice path.

        Establishes the completion stream with the same throttle/TPM
        gates and retry/backoff policy as generate_reply (retries cover
        *starting* the stream; a drop mid-stream surfaces through
        ChatStream.error) and returns the ChatStream immediately, so
        the caller can synthesize sentences from early deltas before
        the reply finishes generating.

        Returns None when the stream could not be established within
        the retry budget; the caller falls back to the non-streaming
        generate_reply path. The caller applies postprocess cleaning
        per-sentence (see ai.postprocess.clean_stream_sentence).
        """
        if not await self._tpm_budget_ok():
            log.info(
                f"Groq TPM budget exhausted (streaming) -- dropping "
                f"request for user={user_id}"
            )
            return None

        await self._throttle()
        start = time.monotonic()

        effort = config.GROQ_REASONING_EFFORT.strip()
        extra: dict = {}
        if effort and "gpt-oss" in self._model:
            extra["reasoning_effort"] = effort

        params: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens or config.GROQ_MAX_TOKENS,
            "stop": ["<|endoftext|>", "<|im_end|>"],
            "stream": True,
        }

        include_usage = True
        last_error: Exception | None = None

        for attempt in range(2):
            try:
                call_kwargs = dict(params)
                if include_usage:
                    call_kwargs["stream_options"] = {"include_usage": True}
                stream = await self._client.chat.completions.create(
                    **extra,
                    **call_kwargs,
                )
                log.info(
                    f"Groq stream started user={user_id} "
                    f"(t={time.monotonic() - start:.2f}s)"
                )
                return ChatStream(
                    stream, on_usage=self._record_tpm,
                    user_id=user_id, t0=start,
                )

            except RateLimitError as e:
                last_error = e
                hinted = self._retry_after_seconds(e)
                backoff = hinted if hinted is not None else float(attempt + 1)
                if backoff > config.GROQ_RETRY_MAX_WAIT:
                    log.info(
                        f"Groq stream rate limited with retry-after "
                        f"{backoff:.1f}s > max "
                        f"{config.GROQ_RETRY_MAX_WAIT:.1f}s -- failing fast"
                    )
                    return None
                log.warning(f"Groq stream rate limited, backing off {backoff}s")
                await asyncio.sleep(backoff)

            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                if attempt == 0:
                    log.warning(f"Groq stream connection issue (will retry): {e}")
                    await asyncio.sleep(1.5)
                    continue
                log.error(f"Groq stream connection issue: {e}")
                return None

            except APIError as e:
                last_error = e
                status = getattr(e, "status_code", None)
                if attempt == 0 and status == 400 and "stream_options" in str(e):
                    log.info("Groq rejected stream_options; retrying stream without it")
                    include_usage = False
                    continue
                log.error(f"Groq stream API error: {e}")
                return None

            except Exception as e:  # noqa: BLE001 - never crash the caller
                last_error = e
                log.error(f"Unexpected Groq stream error: {e}")
                return None

        log.error(f"Groq stream failed to start for user={user_id}: {last_error}")
        return None
