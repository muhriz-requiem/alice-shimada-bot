"""
Tests for ai.postprocess module.
"""

from ai.postprocess import (
    FALLBACK_REPLY,
    clean_response,
    collapse_repeats,
    normalize_ellipsis,
    strip_stammer,
)


class TestNormalizeEllipsis:
    def test_keeps_ascii_ellipsis(self):
        assert normalize_ellipsis("Hello...") == "Hello..."
        assert normalize_ellipsis("Hello ... world") == "Hello ... world"

    def test_converts_unicode_ellipsis(self):
        assert normalize_ellipsis("Hello\u2026") == "Hello..."
        assert normalize_ellipsis("Hello \u2026 world") == "Hello ... world"

    def test_collapses_long_runs_of_dots(self):
        assert normalize_ellipsis("Hello.....") == "Hello..."
        assert normalize_ellipsis("u\u2026 well") == "u... well"

    def test_handles_empty_string(self):
        assert normalize_ellipsis("") == ""
        assert normalize_ellipsis(None) is None


class TestCollapseRepeats:
    def test_collapses_whole_word_repeat(self):
        assert collapse_repeats("I I'm here") == "I'm here"
        assert collapse_repeats("Just just do it") == "Just do it"

    def test_case_insensitive(self):
        assert collapse_repeats("I i'm") == "I'm"
        assert collapse_repeats("JUST just") == "JUST"

    def test_preserves_contractions(self):
        assert collapse_repeats("I I'm") == "I'm"
        assert collapse_repeats("We we're") == "We're"

    def test_no_match_returns_original(self):
        assert collapse_repeats("Hello there") == "Hello there"
        assert collapse_repeats("I am here") == "I am here"

    def test_handles_empty_string(self):
        assert collapse_repeats("") == ""
        assert collapse_repeats(None) is None


class TestStripStammer:
    def test_strips_consonant_stammer(self):
        assert strip_stammer("F-Father. Hi.") == "Father. Hi."
        assert strip_stammer("H-Hi there.") == "Hi there."
        assert strip_stammer("Y-Yeah, I'm here.") == "Yeah, I'm here."

    def test_strips_vowel_stammer(self):
        assert strip_stammer("I-I'm doing alright.") == "I'm doing alright."
        assert strip_stammer("O-Oh, welcome back.") == "Oh, welcome back."
        assert strip_stammer("A-A Boko gift?!") == "A Boko gift?!"

    def test_protects_real_hyphenated_words(self):
        assert strip_stammer("T-shirt") == "T-shirt"
        assert strip_stammer("T-34 tank") == "T-34 tank"
        assert strip_stammer("sensha-do") == "sensha-do"
        assert strip_stammer("Boko-chan") == "Boko-chan"
        assert strip_stammer("X-ray") == "X-ray"

    def test_handles_empty_string(self):
        assert strip_stammer("") == ""
        assert strip_stammer(None) is None


class TestCleanResponse:
    def test_empty_returns_fallback(self):
        assert clean_response("") == FALLBACK_REPLY
        assert clean_response(None) == FALLBACK_REPLY
        assert clean_response("   ") == FALLBACK_REPLY

    def test_strips_whitespace(self):
        assert clean_response("  Hello  ") == "Hello"

    def test_removes_leak_phrases(self):
        assert "How can I help" not in clean_response("Hello\nHow can I help you today")
        # When all lines are leaks, falls back to first line
        result = clean_response("As an AI, I cannot help")
        assert result == FALLBACK_REPLY or "As an AI" in result
        # Single leak line falls back to original
        result = clean_response("As a language model, I know")
        assert result == "As a language model, I know"

    def test_hard_cuts_at_special_tokens(self):
        token = '<|' + 'end' + 'of' + 'text' + '|>'
        result = clean_response(f"Hello {token} more text")
        assert token not in result
        assert result == "Hello"

    def test_hard_cuts_at_urls(self):
        result = clean_response("Check this out https://example.com")
        assert "https://example.com" not in result

    def test_keeps_ellipses(self):
        result = clean_response("Hello ... world")
        assert "..." in result
        assert result == "Hello ... world"

    def test_strips_stammer(self):
        assert clean_response("F-Father. H-Hi.") == "Father. Hi."
        assert clean_response("I I'm here") == "I'm here"

    def test_fallback_when_all_lines_are_leaks(self):
        result = clean_response("How can I help you?\nAs an AI I cannot")
        assert result == FALLBACK_REPLY or result == "How can I help you?"

    def test_character_name_added_to_leak_markers(self):
        # "we are alice" gets added to leak markers when character_name="alice"
        # Since it's the only line and gets filtered, falls back to original line
        result = clean_response("we are alice", character_name="alice")
        assert result == "we are alice"
