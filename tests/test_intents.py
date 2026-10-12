"""
Tests for emotion.intents module.
"""

from emotion.intents import detect_intent


class TestDetectIntent:
    def test_detects_goodnight(self):
        intents = {
            "goodnight": ["goodnight", "gn", "sleep well"],
            "agree": ["agree", "yes"],
        }
        result = detect_intent("Goodnight everyone", intents)
        assert result["goodnight"] is True
        assert result["agree"] is False

    def test_detects_gn_abbreviation(self):
        intents = {"goodnight": ["gn", "good night"]}
        result = detect_intent("gn", intents)
        assert result["goodnight"] is True

    def test_detects_agree(self):
        intents = {"agree": ["agree", "you're right", "true"]}
        result = detect_intent("I agree with you", intents)
        assert result["agree"] is True

    def test_detects_agree_special(self):
        intents = {
            "agree": ["agree", "yes"],
            "agree_special": ["i think so too", "probably"],
        }
        result = detect_intent("I think so too", intents)
        assert result["agree_special"] is True

    def test_case_insensitive(self):
        intents = {"goodnight": ["GOODNIGHT"]}
        result = detect_intent("goodnight", intents)
        assert result["goodnight"] is True

    def test_partial_word_match(self):
        intents = {"test": ["test"]}
        result = detect_intent("testing this", intents)
        assert result["test"] is True

    def test_empty_message(self):
        intents = {"goodnight": ["goodnight"]}
        result = detect_intent("", intents)
        assert result["goodnight"] is False

    def test_empty_intents_config(self):
        result = detect_intent("anything", {})
        assert result == {}

    def test_multiple_intents_detected(self):
        intents = {
            "goodnight": ["goodnight"],
            "agree": ["agree"],
        }
        result = detect_intent("goodnight and I agree", intents)
        assert result["goodnight"] is True
        assert result["agree"] is True
