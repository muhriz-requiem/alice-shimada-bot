"""
Tests for character.memory_shaping module.
"""

from character.memory_shaping import (
    classify_memory,
    format_memory,
    sanitize_memory,
)


class TestClassifyMemory:
    def test_fact_keywords(self):
        assert classify_memory("My name is Alice") == "FACT"
        assert classify_memory("I am a student") == "FACT"
        assert classify_memory("I like cats") == "FACT"
        assert classify_memory("My favorite color is blue") == "FACT"

    def test_emotion_keywords(self):
        assert classify_memory("I feel happy") == "EMOTION"
        assert classify_memory("I'm frustrated") == "EMOTION"
        assert classify_memory("This makes me sad") == "EMOTION"
        assert classify_memory("I trust you") == "EMOTION"

    def test_identity_keywords(self):
        keywords = ["the maintainer", "father", "chiyo", "mother"]
        assert classify_memory("My father is here", keywords) == "IDENTITY"
        assert classify_memory("the maintainer is my dad", keywords) == "IDENTITY"
        assert classify_memory("Chiyo is my mother", keywords) == "IDENTITY"

    def test_general_fallback(self):
        assert classify_memory("Hello there") == "GENERAL"
        assert classify_memory("What time is it?") == "GENERAL"
        assert classify_memory("The weather is nice") == "GENERAL"

    def test_question_bonus(self):
        # Question mark adds GENERAL + small bonus but stays GENERAL
        result = classify_memory("What is your name?")
        assert result == "GENERAL"


class TestSanitizeMemory:
    def test_removes_blocked_phrases(self):
        snippets = [
            "User is a student at academy",
            "User likes cats",
            "Roleplay scenario here",
        ]
        blocked = ["student at", "academy", "roleplay"]
        result = sanitize_memory(snippets, blocked)
        assert len(result) == 1
        assert result[0] == "User likes cats"

    def test_case_insensitive_blocking(self):
        snippets = ["STUDENT AT school", "Normal memory"]
        blocked = ["student at"]
        result = sanitize_memory(snippets, blocked)
        assert len(result) == 1

    def test_limits_to_last_12(self):
        snippets = [f"Memory {i}" for i in range(20)]
        blocked = []
        result = sanitize_memory(snippets, blocked)
        assert len(result) == 12
        assert result[0] == "Memory 8"

    def test_empty_input(self):
        assert sanitize_memory([], ["blocked"]) == []
        assert sanitize_memory(None, ["blocked"]) == []

    def test_empty_blocked_phrases(self):
        snippets = ["Memory 1", "Memory 2"]
        result = sanitize_memory(snippets, [])
        assert result == snippets


class TestFormatMemory:
    def test_formats_with_classification(self):
        snippets = ["My name is Alice", "I feel happy", "Hello there"]
        result = format_memory(snippets, ["the maintainer"])
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0].startswith("[FACT]")
        assert lines[1].startswith("[EMOTION]")
        assert lines[2].startswith("[GENERAL]")

    def test_limits_to_last_10(self):
        snippets = [f"Memory {i}" for i in range(15)]
        result = format_memory(snippets)
        lines = result.split("\n")
        assert len(lines) == 10

    def test_empty_returns_none(self):
        assert format_memory([]) == "None"
        assert format_memory(None) == "None"

    def test_identity_keywords_passed_through(self):
        snippets = ["the maintainer is here"]
        result = format_memory(snippets, ["the maintainer"])
        assert "[IDENTITY]" in result
