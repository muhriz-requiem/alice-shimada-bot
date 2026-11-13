"""
Tests for ai.postprocess.clean_stream_sentence / find_hard_cut (the
per-sentence cleaning used by the streaming voice path) and for
ai.groq_client.ChatStream (the streaming completion handle).
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from ai.postprocess import clean_stream_sentence, clean_response, find_hard_cut
from ai.groq_client import ChatStream


class TestCleanStreamSentence:
    def test_collapses_word_repeats(self):
        assert clean_stream_sentence("I I am here.") == "I am here."

    def test_strips_stammer(self):
        assert clean_stream_sentence("F-Father, look.") == "Father, look."

    def test_normalizes_unicode_ellipsis(self):
        assert clean_stream_sentence("hmm… ok") == "hmm... ok"

    def test_leak_phrase_drops_whole_sentence(self):
        assert clean_stream_sentence("as an AI I cannot do that") == ""

    def test_character_leak_marker(self):
        assert clean_stream_sentence("we are alice of the school", "Alice") == ""

    def test_empty_input(self):
        assert clean_stream_sentence("") == ""
        assert clean_stream_sentence("   ") == ""


class TestFindHardCut:
    def test_special_token_location(self):
        assert find_hard_cut("hello <|im_end|> junk") == 6

    def test_url_location(self):
        assert find_hard_cut("check https://example.com out") == 6

    def test_clean_text(self):
        assert find_hard_cut("perfectly fine sentence.") is None

    def test_clean_response_unchanged(self):
        # batch path still behaves the same using the shared helper
        assert clean_response("Hello. I I am...", "Alice") == "Hello. I am..."
        assert clean_response("aaa <|x|> leak", "Alice") == "aaa"


def _chunk(text: str, usage=None):
    if text is None:
        return SimpleNamespace(usage=usage, choices=None)
    return SimpleNamespace(
        usage=usage,
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))],
    )


class TestChatStream:
    def test_accumulates_deltas_and_first_token(self):
        async def gen():
            yield _chunk("Hello")
            yield _chunk(" ")
            yield _chunk("there.")
            yield _chunk(None, usage=SimpleNamespace(total_tokens=10))

        stream = ChatStream(gen())
        gathered = asyncio.run(stream.text())
        assert gathered == "Hello there."
        assert stream.first_token_at is not None
        assert stream.finished
        assert not stream.error

    def test_midstream_error_keeps_partial_text(self):
        async def gen():
            yield _chunk("Holder of ")
            raise RuntimeError("connection dropped")

        stream = ChatStream(gen())
        gathered = asyncio.run(stream.text())
        assert gathered == "Holder of "
        assert stream.finished
        assert isinstance(stream.error, RuntimeError)

    def test_usage_callback_fires(self):
        seen: list = []

        async def gen():
            yield _chunk("x", usage=SimpleNamespace(total_tokens=7))
            yield _chunk("y")

        stream = ChatStream(gen(), on_usage=lambda c: seen.append(c))
        asyncio.run(stream.text())
        assert len(seen) == 1

    def test_empty_stream_finished(self):
        async def gen():
            yield _chunk(None)

        stream = ChatStream(gen())
        gathered = asyncio.run(stream.text())
        assert gathered == ""
        assert stream.finished
        assert stream.first_token_at is None

    def test_first_token_monotonic_reference(self):
        async def gen():
            yield _chunk("")  # blank deltas skipped
            yield _chunk("real")

        stream = ChatStream(gen())
        asyncio.run(stream.text())
        assert stream.first_token_at <= time.monotonic()