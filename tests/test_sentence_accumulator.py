"""
Tests for voice.streaming_tts.SentenceAccumulator: the incremental
sentence splitter used by the streaming Groq reply path. Contract:
feed() + flush() must equal split_sentences() on the concatenated text
(same boundaries as the batch path), and an undecidable trailing
boundary must NOT split early.
"""

from __future__ import annotations

from voice.streaming_tts import SentenceAccumulator, split_sentences

SAMPLES = [
    "Hello there. How are you today? I am fine!",
    "I'm doing alright... I think.",
    "Father. ...I think so.",
    "Hello... ... ... World. ...More here.",
    "こんにちは。今日はいい天気です",
    "そうだね。うん。わかった",
    "Wait... really? Yes.",
    "One. Two. Three.",
]


class TestParityWithBatch:
    def test_feed_plus_flush_equals_split(self):
        for text in SAMPLES:
            acc = SentenceAccumulator()
            streamed: list[str] = []
            for i in range(len(text)):
                streamed.extend(acc.feed(text[i]))
            streamed.extend(acc.flush())
            assert streamed == split_sentences(text), f"mismatch for {text!r}"

    def test_whole_text_in_one_feed(self):
        for text in SAMPLES:
            acc = SentenceAccumulator()
            got = acc.feed(text) + acc.flush()
            assert got == split_sentences(text), f"mismatch for {text!r}"

    def test_big_delta_chunks_match(self):
        for text in SAMPLES:
            acc = SentenceAccumulator()
            streamed: list[str] = []
            for i in range(0, len(text), 3):
                streamed.extend(acc.feed(text[i:i + 3]))
            streamed.extend(acc.flush())
            assert streamed == split_sentences(text)


class TestIncrementalTiming:
    def test_sentence_available_as_soon_as_boundary_known(self):
        acc = SentenceAccumulator()
        assert acc.feed("Hello there") == []
        out = acc.feed(". How are you")
        assert out == ["Hello there."]
        # The rest is still buffered, not lost
        assert acc.flush() == ["How are you"]

    def test_trailing_ja_period_waits_for_next_char(self):
        acc = SentenceAccumulator()
        # "。" could begin a 。。。 run: no split until one more character.
        assert acc.feed("こんにちは。") == []
        assert acc.feed("今") == ["こんにちは。"]

    def test_trailing_ascii_period_no_split_without_ws(self):
        acc = SentenceAccumulator()
        assert acc.feed("alright.") == []
        out = acc.feed(" I")
        assert out == ["alright."]

    def test_pure_punctuation_chunks_dropped(self):
        acc = SentenceAccumulator()
        out = acc.feed("Hello... ... ... World. ")
        # The bare "..." runs between whitespace collapse into the same
        # chunk and it contains words, so it is kept; a punct-only chunk
        # between sentence boundaries must not be emitted as its own TTS
        # request.
        assert all(c.strip() for c in out)
        assert not any(
            c.replace(".", "").replace(" ", "") == "" for c in out
        )

    def test_flush_emits_remainder(self):
        acc = SentenceAccumulator()
        assert acc.feed("complete sentence.") == []
        # End of stream: the undecided trailing 。 belongs to the final
        # chunk, exactly like split_sentences keeps it attached.
        assert acc.flush() == ["complete sentence."]

    def test_buffer_drained_after_flush(self):
        acc = SentenceAccumulator()
        first = acc.feed("Hello. more")
        assert first == ["Hello."]
        assert acc.flush() == ["more"]


class TestEllipsisRuns:
    def test_ellipsis_run_splits_after_run_not_inside(self):
        acc = SentenceAccumulator()
        out = acc.feed("I'm doing alright... I think.")
        # The '...' run stays intact; boundary falls at the whitespace
        # after it. The second sentence still lacks a decidable trailing
        # boundary at this point (batch parity: flush finishes it).
        assert out == ["I'm doing alright..."]
        assert acc.flush() == ["I think."]

    def test_no_split_inside_the_run(self):
        acc = SentenceAccumulator()
        out = acc.feed("alright..")  # still inside the run
        assert out == []
        out = acc.feed(". I")
        assert out[0] == "alright..."

    def test_leading_ellipsis_chunk_attaches_forward(self):
        acc = SentenceAccumulator()
        out = acc.feed("Father. ...I think. so")
        split = split_sentences("Father. ...I think. so")
        assert out == split[: len(out)]
        assert out[0] == "Father."
