"""
tests/test_voice_stt_hallucination.py
=====================================
Sanity checks for the STT hallucination filter (_is_stt_hallucination),
which sits between transcription and the AI so bad transcripts never seed
a response loop.
"""

from __future__ import annotations

import pytest
import voice.pipeline as pipeline


class TestIsSttHallucination:
    """_is_stt_hallucination rejects loops/prompt-echo; keeps real speech."""

    @pytest.fixture(autouse=True)
    def set_defaults(self, monkeypatch):
        monkeypatch.setattr(
            pipeline.config, "VOICE_MENTION_KEYWORDS", ["alice", "alice123"]
        )

    @pytest.mark.parametrize(
        "text,dur,expected",
        [
            # --- Rule 1: known hallucination sentences ---
            # Exact loop of the classic prompt phrase = reject.
            ("Alice, how are you doing? Alice, how are you doing?", 1.5, True),
            # Saturated single occurrence on short audio = prompt echo.
            ("Hello Alice. Hi Alice. Alice, how are you doing?", 1.8, True),
            # Saturated single occurrence on short audio = prompt echo.
            ("What are you doing, Alice? What are you doing, Alice?", 1.5, True),
            # Two different hallucination sentences = reject.
            ("Hello Alice. Hi Alice.", 1.0, True),
            # Two sentence variants = reject.
            ("Alice, how are you doing? What are you doing, Alice?", 2.0, True),
            # --- Rule 2: self-address loop ---
            # Keyword+comma+short phrase twice = reject.
            ("Alice? Alice? Alice,", 1.5, True),
            ("Alice, how are you doing? Alice,", 1.5, True),
            # --- Rule 3: silence impostors ---
            ("uh um mm hmm", 5.0, True),
            ("you", 5.0, True),
            ("um", 5.0, True),
            # --- Keep these (real speech) ---
            # One-off clean direct address = allowed even on short audio.
            ("Alice, how are you doing?", 2.5, False),
            ("alice, what's up?", 3.0, False),
            # Normal speech with keyword mentioned naturally.
            ("hey alice, what's up with the game last night", 4.0, False),
            ("hey, what's up with the game last night", 4.0, False),
            ("alice, can you pass me the salt?", 3.0, False),
            # Question with keyword but not a clean address.
            ("did alice say something about the game?", 3.0, False),
            # Longer coherent sentence saturating a phrase but real.
            (
                "alice, how are you doing? "
                "i'm doing great, thanks for asking alice",
                4.5, False,
            ),
            # Legitimate greetings / clean one-off direct addresses pass.
            ("hi alice", 1.0, False),
            ("Hello Alice.", 1.5, False),
            ("Hi Alice.", 1.0, False),
            ("hello alice, what's up?", 2.5, False),
            ("alice, how are you doing today?", 3.0, False),
        ],
    )
    def test_hallucination_filter(
        self, text: str, dur: float, expected: bool
    ) -> None:
        assert pipeline._is_stt_hallucination(text, dur) == expected

    def test_custom_hallucination_phrases(self, monkeypatch):
        monkeypatch.setattr(
            pipeline.config,
            "VOICE_STT_HALLUCINATION_PHRASES",
            "Custom Alice Phrase.|Another one.",
        )
        assert pipeline._is_stt_hallucination(
            "Custom Alice Phrase. Custom Alice Phrase.", 1.0
        )
        assert not pipeline._is_stt_hallucination(
            "alice, what's up?", 3.0
        )

    def test_empty_text_is_rejected(self):
        assert pipeline._is_stt_hallucination("", 2.0)
        assert pipeline._is_stt_hallucination("   ", 2.0)
