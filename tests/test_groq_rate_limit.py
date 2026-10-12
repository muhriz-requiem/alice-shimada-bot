"""
tests/test_groq_rate_limit.py
=============================
Covers the rate-limit hardening added after a live 3-user voice session
where Groq 429 retries stacked for 40s+, the fallback line was spoken
over and over, and concurrent replies raced into vc.play() ("Already
playing audio"):

- rolling TPM gate drops requests instead of queueing them
- retry-after headers honored; long waits fail fast
- SDK-internal retries disabled (no more double-retry stacking)
- per-guild reply serialization + cooldown stamped at reply start
- voice stays silent when the backend returns the fallback
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import httpx
from openai import RateLimitError

import config

GUILD = 111222333
USER = 444555666


def _rl_error(retry_after: str | None) -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, headers=headers, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def _ok_response(text: str = "Hello there.") -> SimpleNamespace:
    usage = SimpleNamespace(
        total_tokens=1000,
        prompt_tokens_details=SimpleNamespace(cached_tokens=700),
    )
    message = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeCreate:
    """Stands in for chat.completions.create with scripted side effects."""

    def __init__(self, side_effects: list) -> None:
        self.calls = 0
        self.kwargs: dict | None = None
        self._side_effects = list(side_effects)

    async def __call__(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        effect = self._side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


def _make_client(monkeypatch, side_effects: list):
    from ai.groq_client import GroqClient

    monkeypatch.setattr(config, "GROQ_RATE_LIMIT_RPM", 6000)  # ~0s gap
    client = GroqClient()
    fake = FakeCreate(side_effects)
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake))
    )
    return client, fake


# ---------------------------------------------------------------------
# Groq client: TPM gate
# ---------------------------------------------------------------------


def test_tpm_gate_allows_empty_window(monkeypatch):
    client, _ = _make_client(monkeypatch, [])
    assert asyncio.run(client._tpm_budget_ok()) is True


def test_tpm_gate_fails_fast_when_budget_spent(monkeypatch):
    monkeypatch.setattr(config, "GROQ_TPM_LIMIT", 1000)
    monkeypatch.setattr(config, "GROQ_RETRY_MAX_WAIT", 0.5)
    client, _ = _make_client(monkeypatch, [])
    client._tpm_window.append((time.monotonic(), 950))  # fresh usage, over 90%
    started = time.monotonic()
    assert asyncio.run(client._tpm_budget_ok()) is False
    # Must not block for the ~60s the entry has left in the window.
    assert time.monotonic() - started < 1.0


def test_tpm_gate_waits_for_budget_to_age_out(monkeypatch):
    monkeypatch.setattr(config, "GROQ_TPM_LIMIT", 1000)
    monkeypatch.setattr(config, "GROQ_RETRY_MAX_WAIT", 2.0)
    client, _ = _make_client(monkeypatch, [])
    client._tpm_window.append((time.monotonic() - 59.7, 950))  # expires ~0.3s
    assert asyncio.run(client._tpm_budget_ok()) is True
    assert len(client._tpm_window) == 0


def test_record_tpm_excludes_cached_tokens(monkeypatch):
    client, _ = _make_client(monkeypatch, [])
    client._record_tpm(_ok_response())
    assert len(client._tpm_window) == 1
    _, tokens = client._tpm_window[0]
    assert tokens == 300  # 1000 total - 700 prompt-cache hits


# ---------------------------------------------------------------------
# Groq client: rate-limit handling
# ---------------------------------------------------------------------


def test_long_retry_after_fails_fast(monkeypatch):
    monkeypatch.setattr(config, "GROQ_RETRY_MAX_WAIT", 3.0)
    client, fake = _make_client(monkeypatch, [_rl_error("22.0"), _ok_response()])
    started = time.monotonic()
    result = asyncio.run(
        client.generate_reply(
            [{"role": "user", "content": "hi"}], user_id="u", channel_id="c"
        )
    )
    elapsed = time.monotonic() - started
    assert result == config.FALLBACK_REPLY
    assert fake.calls == 1  # did not sit on a 22s retry
    assert elapsed < 1.0


def test_short_retry_after_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(config, "GROQ_RETRY_MAX_WAIT", 3.0)
    client, fake = _make_client(monkeypatch, [_rl_error("0"), _ok_response("Hi!")])
    result = asyncio.run(
        client.generate_reply(
            [{"role": "user", "content": "hi"}], user_id="u", channel_id="c"
        )
    )
    assert result == "Hi!"
    assert fake.calls == 2


def test_missing_retry_after_uses_exponential_backoff(monkeypatch):
    monkeypatch.setattr(config, "GROQ_RETRY_MAX_WAIT", 5.0)
    client, fake = _make_client(monkeypatch, [_rl_error(None), _ok_response("Yo")])
    result = asyncio.run(
        client.generate_reply(
            [{"role": "user", "content": "hi"}], user_id="u", channel_id="c"
        )
    )
    assert result == "Yo"
    assert fake.calls == 2


def test_tpm_exhausted_drops_without_calling_api(monkeypatch):
    monkeypatch.setattr(config, "GROQ_TPM_LIMIT", 1000)
    monkeypatch.setattr(config, "GROQ_RETRY_MAX_WAIT", 0.5)
    client, fake = _make_client(monkeypatch, [_ok_response()])
    client._tpm_window.append((time.monotonic(), 950))
    result = asyncio.run(
        client.generate_reply(
            [{"role": "user", "content": "hi"}], user_id="u", channel_id="c"
        )
    )
    assert result == config.FALLBACK_REPLY
    assert fake.calls == 0


def test_max_tokens_passthrough(monkeypatch):
    client, fake = _make_client(monkeypatch, [_ok_response()])
    asyncio.run(
        client.generate_reply(
            [{"role": "user", "content": "hi"}],
            user_id="u",
            channel_id="c",
            max_tokens=250,
        )
    )
    assert fake.kwargs["max_tokens"] == 250

    client2, fake2 = _make_client(monkeypatch, [_ok_response()])
    asyncio.run(
        client2.generate_reply(
            [{"role": "user", "content": "hi"}], user_id="u", channel_id="c"
        )
    )
    assert fake2.kwargs["max_tokens"] == config.GROQ_MAX_TOKENS


def test_sdk_internal_retries_disabled(monkeypatch):
    # The 40s pileups came from the SDK's own retries stacking with
    # this class's loop -- the SDK must be configured not to retry.
    from ai.groq_client import GroqClient

    monkeypatch.setattr(config, "GROQ_RATE_LIMIT_RPM", 6000)
    client = GroqClient()
    assert client._client.max_retries == 0


# ---------------------------------------------------------------------
# Voice pipeline: serialization + silent fallback
# ---------------------------------------------------------------------


class FakeAIClient:
    def __init__(self, reply: str, block: asyncio.Event | None = None) -> None:
        self.calls = 0
        self.reply = reply
        self.block = block
        self.max_tokens: int | None = None

    async def generate_reply(self, messages, *, user_id, channel_id, max_tokens=None):
        self.calls += 1
        self.max_tokens = max_tokens
        if self.block is not None:
            await self.block.wait()
        return self.reply


def _make_pipeline(reply: str, block: asyncio.Event | None = None):
    from voice.pipeline import VoicePipeline

    bot = SimpleNamespace(
        character=object(),
        ai_client=FakeAIClient(reply, block),
        user=SimpleNamespace(id=999888777),
    )
    return VoicePipeline(bot)


def _patch_pipeline_deps(monkeypatch, ready: bool = True) -> None:
    import character.prompt_builder
    import memory.store
    import voice.pipeline
    import voice.readiness

    monkeypatch.setattr(voice.readiness, "is_ready", lambda: ready)
    monkeypatch.setattr(voice.pipeline, "prompt_builder", character.prompt_builder)
    monkeypatch.setattr(
        character.prompt_builder,
        "build_prompt",
        lambda *args, **kwargs: [{"role": "user", "content": "hi"}],
    )

    async def _fake_identity(user_id_str: str) -> dict:
        return {}

    monkeypatch.setattr(voice.pipeline, "get_identity_async", _fake_identity)
    monkeypatch.setattr(memory.store, "get_relationship", lambda user_id_str: None)


def test_fallback_reply_is_silent_in_voice(monkeypatch):
    _patch_pipeline_deps(monkeypatch)
    pipe = _make_pipeline(config.FALLBACK_REPLY)
    asyncio.run(pipe._generate_and_respond(GUILD, USER, "hello", False, "en"))
    assert not pipe._reply_locks[GUILD].locked()
    assert pipe._last_reply.get(GUILD, 0) > 0  # cooldown stamped at start
    assert pipe.bot.ai_client.max_tokens == config.VOICE_MAX_TOKENS
    recent = pipe.conversation.get_recent(GUILD, count=10)
    assert not any(u.is_bot for u in recent)  # fallback never entered history


def test_second_utterance_dropped_while_reply_in_flight(monkeypatch):
    _patch_pipeline_deps(monkeypatch)
    block = asyncio.Event()
    pipe = _make_pipeline(config.FALLBACK_REPLY, block=block)

    async def scenario() -> None:
        task = asyncio.create_task(
            pipe._generate_and_respond(GUILD, USER, "hello", False, "en")
        )
        for _ in range(200):
            if pipe.bot.ai_client.calls:
                break
            await asyncio.sleep(0.01)
        assert pipe.bot.ai_client.calls == 1
        # Stamped while still inside generate_reply -- previously the
        # cooldown was only set after playback finished, so utterances
        # arriving during generation all passed the check.
        assert pipe._last_reply[GUILD] > 0

        # Bypass the cooldown so this call reaches the lock check.
        pipe._last_reply[GUILD] = 0.0
        await pipe._generate_and_respond(GUILD, USER + 1, "again", False, "en")
        assert pipe.bot.ai_client.calls == 1  # dropped, not queued
        assert pipe._last_reply[GUILD] == 0.0  # drop did not re-stamp

        block.set()
        await asyncio.wait_for(task, timeout=5)
        assert not pipe._reply_locks[GUILD].locked()

    asyncio.run(scenario())


def test_cooldown_gates_immediate_second_reply(monkeypatch):
    _patch_pipeline_deps(monkeypatch)
    pipe = _make_pipeline(config.FALLBACK_REPLY)

    async def scenario() -> None:
        await pipe._generate_and_respond(GUILD, USER, "hello", False, "en")
        assert pipe.bot.ai_client.calls == 1
        await pipe._generate_and_respond(GUILD, USER, "hello again", False, "en")
        assert pipe.bot.ai_client.calls == 1  # gated by cooldown

    asyncio.run(scenario())


def test_not_ready_does_not_stamp_cooldown(monkeypatch):
    _patch_pipeline_deps(monkeypatch, ready=False)
    pipe = _make_pipeline(config.FALLBACK_REPLY)
    asyncio.run(pipe._generate_and_respond(GUILD, USER, "hello", False, "en"))
    assert GUILD not in pipe._last_reply
    assert pipe.bot.ai_client.calls == 0
