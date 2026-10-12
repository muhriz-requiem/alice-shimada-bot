"""
Tests for ai.router module.
"""

from ai.router import is_fast_path


class TestIsFastPath:
    def test_empty_string(self):
        assert is_fast_path("") is True
        assert is_fast_path(None) is True
        assert is_fast_path("   ") is True

    def test_greetings(self):
        greetings = ["hi", "hey", "hello", "yo", "sup", "hiya", "heya"]
        for g in greetings:
            assert is_fast_path(g) is True, f"Failed for: {g}"
            assert is_fast_path(f"  {g}  ") is True

    def test_short_acknowledgments(self):
        acks = ["ok", "okay", "k", "kk", "lol", "lmao", "haha"]
        for a in acks:
            assert is_fast_path(a) is True

    def test_yes_no(self):
        responses = ["yes", "no", "yeah", "yep", "nah", "nope"]
        for r in responses:
            assert is_fast_path(r) is True

    def test_thanks(self):
        thanks = ["thanks", "thank you", "ty", "np"]
        for t in thanks:
            assert is_fast_path(t) is True

    def test_good_morning_night(self):
        times = ["good morning", "good night", "gn", "gm"]
        for t in times:
            assert is_fast_path(t) is True

    def test_bye(self):
        byes = ["bye", "goodbye", "cya", "see ya"]
        for b in byes:
            assert is_fast_path(b) is True

    def test_short_no_question(self):
        assert is_fast_path("How are you") is True  # 3 words, no ?
        assert is_fast_path("I am here") is True
        assert is_fast_path("Just checking") is True

    def test_question_mark_not_fast(self):
        assert is_fast_path("How are you?") is False
        assert is_fast_path("What?") is False

    def test_longer_messages_not_fast(self):
        assert is_fast_path("This is a longer message with more words") is False
        assert is_fast_path("Hello how are you doing today?") is False

    def test_punctuation_stripped(self):
        assert is_fast_path("Hello!") is True
        # "Hey?" strips to "hey" which is a greeting -> fast path
        assert is_fast_path("Hey?") is True
        assert is_fast_path("OK.") is True
