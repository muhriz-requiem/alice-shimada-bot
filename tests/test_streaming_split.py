"""
Tests for voice.streaming_tts sentence splitting.
"""

from voice.streaming_tts import _CHUNK_PURE_PUNCT, split_sentences


class TestSplitSentences:
    def test_basic_english_split(self):
        chunks = split_sentences("Hello there. How are you today? I am fine!")
        assert chunks == ["Hello there.", "How are you today?", "I am fine!"]

    def test_ellipsis_keeps_words_attached(self):
        chunks = split_sentences("I'm doing alright... I think.")
        assert chunks == ["I'm doing alright...", "I think."]

    def test_leading_ellipsis_not_dot_chunks(self):
        chunks = split_sentences("Father. ...I think so.")
        assert chunks == ["Father.", "...I think so."]

    def test_no_bare_punctuation_chunks(self):
        text = "Hello... ... ... World. ...More here."
        chunks = split_sentences(text)
        assert not any(_CHUNK_PURE_PUNCT.fullmatch(c) for c in chunks)
        assert "World." in chunks

    def test_japanese_no_space_after_period(self):
        chunks = split_sentences("こんにちは。今日はいい天気です")
        assert chunks == ["こんにちは。", "今日はいい天気です"]

    def test_empty_inputs(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_single_sentence(self):
        assert split_sentences("Just one line") == ["Just one line"]


class TestPurePunctPattern:
    def test_matches_punctuation_only(self):
        assert _CHUNK_PURE_PUNCT.fullmatch("...")
        assert _CHUNK_PURE_PUNCT.fullmatch(". . !")
        assert _CHUNK_PURE_PUNCT.fullmatch("。。。")

    def test_rejects_words(self):
        assert not _CHUNK_PURE_PUNCT.fullmatch("Hello...")
        assert not _CHUNK_PURE_PUNCT.fullmatch("...Hi")
        assert not _CHUNK_PURE_PUNCT.fullmatch("T-34")
