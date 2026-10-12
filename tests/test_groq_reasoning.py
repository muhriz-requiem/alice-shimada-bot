"""
tests/test_groq_reasoning.py
=============================
Covers the gpt-oss empty-content bug found in a live voice session:
reasoning tokens spend from max_tokens BEFORE any visible content, so
a short voice budget (VOICE_MAX_TOKENS=250 at the default medium
reasoning effort) could be consumed entirely by reasoning. Groq then
returns HTTP 200 with empty content, clean_response masks it as
FALLBACK_REPLY, groq_client logged "reply ok", and the voice pipeline
went silent with nothing useful in the log.

- empty content -> fallback + a warning that says WHY (finish reason,
  reasoning size), never "reply ok"
- TPM still records the billable reasoning tokens on that path
- reasoning_effort is sent only for gpt-oss models (Groq accepts it
  only on gpt-oss 20B/120B) and only when the config knob is set
- non-empty replies pass through unchanged
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import config


def _response(text: str, reasoning: str = "", finish: str = "stop",
              cached: int = 0, total: int = 1000) -> SimpleNamespace:
    usage = SimpleNamespace(
        total_tokens=total,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )
    message = SimpleNamespace(content=text, reasoning=reasoning)
    choice = SimpleNamespace(message=message, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage)


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


def _make_client(monkeypatch, side_effects: list,
                 model: str = "openai/gpt-oss-120b",
                 effort: str = "low"):
    from ai.groq_client import GroqClient

    monkeypatch.setattr(config, "GROQ_RATE_LIMIT_RPM", 6000)  # ~0s gap
    monkeypatch.setattr(config, "GROQ_MODEL", model)
    monkeypatch.setattr(config, "GROQ_REASONING_EFFORT", effort)
    client = GroqClient()
    fake = FakeCreate(side_effects)
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake))
    )
    return client, fake


def _reply(client, **kwargs) -> str:
    return asyncio.run(
        client.generate_reply(
            [{"role": "user", "content": "hi"}],
            user_id="u",
            channel_id="c",
            **kwargs,
        )
    )


# ---------------------------------------------------------------------
# reasoning_effort passthrough
# ---------------------------------------------------------------------


def test_reasoning_effort_sent_for_gpt_oss(monkeypatch):
    client, fake = _make_client(monkeypatch, [_response("Hello!")])
    assert _reply(client) == "Hello!"
    assert fake.kwargs["reasoning_effort"] == "low"


def test_reasoning_effort_skipped_when_config_empty(monkeypatch):
    client, fake = _make_client(monkeypatch, [_response("Hello!")], effort="")
    _reply(client)
    assert "reasoning_effort" not in fake.kwargs


def test_reasoning_effort_skipped_for_non_gpt_oss_model(monkeypatch):
    # Groq only accepts this knob on gpt-oss 20B/120B; other families
    # would 400, so the client must not send it.
    client, fake = _make_client(
        monkeypatch, [_response("Hello!")], model="llama-3.3-70b-versatile"
    )
    _reply(client)
    assert "reasoning_effort" not in fake.kwargs


# ---------------------------------------------------------------------
# empty content (reasoning ate the budget)
# ---------------------------------------------------------------------


def test_empty_content_returns_fallback_with_warning(monkeypatch, caplog):
    client, _ = _make_client(
        monkeypatch, [_response("", reasoning="x" * 500, finish="length")]
    )
    with caplog.at_level(logging.INFO, logger="ai.groq"):
        result = _reply(client, max_tokens=250)
    assert result == config.FALLBACK_REPLY
    assert "empty content" in caplog.text
    assert "finish=length" in caplog.text
    # The misleading "reply ok" must not fire for a fallback return.
    assert "reply ok" not in caplog.text


def test_empty_content_still_records_tpm(monkeypatch):
    client, _ = _make_client(
        monkeypatch, [_response("", finish="length", total=700)]
    )
    _reply(client)
    # Reasoning tokens are billable -- the gate must see them even
    # though no visible content was produced.
    assert len(client._tpm_window) == 1
    _, tokens = client._tpm_window[0]
    assert tokens == 700


def test_filtered_to_fallback_logs_unusable_not_ok(monkeypatch, caplog):
    # A bare URL hard-cuts to empty inside clean_response, which
    # masks the empty text as the fallback.
    client, _ = _make_client(monkeypatch, [_response("https://example.com/x")])
    with caplog.at_level(logging.INFO, logger="ai.groq"):
        result = _reply(client)
    assert result == config.FALLBACK_REPLY
    assert "unusable after cleaning" in caplog.text
    assert "reply ok" not in caplog.text


# ---------------------------------------------------------------------
# happy path unchanged
# ---------------------------------------------------------------------


def test_normal_reply_passes_through(monkeypatch):
    client, fake = _make_client(monkeypatch, [_response("Hi there.")])
    assert _reply(client) == "Hi there."
    assert fake.kwargs["max_tokens"] == config.GROQ_MAX_TOKENS
    assert fake.kwargs["reasoning_effort"] == "low"
