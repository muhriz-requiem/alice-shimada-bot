"""Tests for the shared-call governor in voice/response.py.

Scenario basis: on 2026-10-25 the owner was in a call with a friend and
said "you're not that sweet" *to the friend*; the owner-priority rule
made Alice reply anyway. The new rule: in a shared call (another human
present in the voice channel, regardless of who spoke when) she only
speaks when addressed by name; a private call keeps the owner-priority
behavior. The engine receives others_present as a plain bool -- the
pipeline computes it from channel members, so no discord objects here.
"""

from __future__ import annotations

import pytest

import config
from voice.response import (
    ResponseDecision,
    ResponseDecisionEngine,
    ResponsePriority,
)
from voice.turn_detector import TurnState

INCIDENT_TEXT = "you're not that sweet"  # said to the friend, not to Alice


class FakeConversation:
    """Only get_alice_silence_duration is consulted by the engine."""

    def __init__(self, alice_silence: float = 30.0):
        self.alice_silence = alice_silence

    def get_alice_silence_duration(self, guild_id: int) -> float:
        return self.alice_silence


class FakeTurnDetector:
    """Fixed turn state per instance."""

    def __init__(self, state: TurnState = TurnState.SINGLE_SPEAKER):
        self.state = state

    def get_state(self, guild_id: int) -> TurnState:
        return self.state


def make_engine(turn_state: TurnState = TurnState.SINGLE_SPEAKER,
                alice_silence: float = 30.0) -> ResponseDecisionEngine:
    return ResponseDecisionEngine(
        conversation=FakeConversation(alice_silence),
        turn_detector=FakeTurnDetector(turn_state),
    )


@pytest.fixture()
def smart_config(monkeypatch):
    """Pin the knobs the engine reads so tests don't inherit .env values."""
    monkeypatch.setattr(config, "VOICE_RESPONSE_MODE", "smart", raising=False)
    monkeypatch.setattr(config, "VOICE_OWNER_PRIORITY", True, raising=False)
    monkeypatch.setattr(config, "VOICE_SHARED_CALL_GOVERNOR", True, raising=False)
    monkeypatch.setattr(config, "VOICE_REPLY_COOLDOWN", 0.0, raising=False)
    monkeypatch.setattr(config, "VOICE_MENTION_KEYWORDS", ["alice"], raising=False)
    monkeypatch.setattr(config, "VOICE_NAME_MENTION_RESPONSE", "soft", raising=False)
    monkeypatch.setattr(config, "VOICE_LULL_JOIN_PROBABILITY", 0.3, raising=False)
    monkeypatch.setattr(config, "VOICE_BACKCHANNEL_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "VOICE_BARGE_IN_ENABLED", True, raising=False)
    return monkeypatch


def _tie_random(monkeypatch, value: float) -> None:
    """Make random.random() deterministic for the lull/backchannel paths."""
    monkeypatch.setattr("voice.response.random.random", lambda: value)


# --- Private call behavior (unchanged) -------------------------------------


def test_solo_owner_unaddressed_responds(smart_config):
    """A 1:1 call: anything the owner says is assumed to be for her."""
    engine = make_engine()
    decision, priority = engine.should_respond(
        1, 111, INCIDENT_TEXT, is_owner=True, others_present=False,
    )
    assert decision is ResponseDecision.RESPOND
    assert priority is ResponsePriority.CRITICAL


def test_solo_nonowner_unaddressed_default_skip(smart_config):
    """Non-owner, private call, nothing in context -> unchanged default."""
    engine = make_engine()
    decision, priority = engine.should_respond(
        1, 222, "i-i was in here", is_owner=False, others_present=False,
    )
    assert decision is ResponseDecision.SKIP
    assert priority is ResponsePriority.MINIMAL


# --- Shared call: the 2026-10-25 incident ----------------------------------


def test_shared_owner_unaddressed_skips(smart_config):
    """Incident: owner's remark to the friend must not trigger a reply."""
    engine = make_engine()
    decision, priority = engine.should_respond(
        1, 111, INCIDENT_TEXT, is_owner=True, others_present=True,
    )
    assert decision is ResponseDecision.SKIP
    assert priority is ResponsePriority.MINIMAL


def test_shared_owner_unaddressed_skips_even_in_lull(smart_config, monkeypatch):
    """Owner-side lull joins (the 70% chance) must also be suppressed.
    Same utterance after a quiet stretch while the friend is present."""
    _tie_random(monkeypatch, 0.01)  # would always respond without the gate
    engine = make_engine(TurnState.LULL)
    decision, _ = engine.should_respond(
        1, 111, INCIDENT_TEXT, is_owner=True, others_present=True,
    )
    assert decision is ResponseDecision.SKIP


def test_shared_owner_midpause_skips(smart_config):
    """Long quiet stretch mid-chat with the friend: still skip, not the
    old owner-priority reply."""
    engine = make_engine(TurnState.PAUSE)
    decision, _ = engine.should_respond(
        1, 111, "okay, never", is_owner=True, others_present=True,
    )
    assert decision is ResponseDecision.SKIP


def test_shared_nonowner_unaddressed_skips(smart_config):
    """Friend chatter without her name stays silent (always-mode previously
    answered this with 'Oh... hi....')."""
    engine = make_engine()
    decision, _ = engine.should_respond(
        1, 222, "i-i was in here", is_owner=False, others_present=True,
    )
    assert decision is ResponseDecision.SKIP


def test_shared_backchannel_suppressed(smart_config, monkeypatch):
    """A brief acknowledgment aimed at nobody in particular is still her
    butting in when someone else is present."""
    _tie_random(monkeypatch, 0.01)  # would backchannel the question solo
    engine = make_engine(TurnState.PAUSE)
    decision, _ = engine.should_respond(
        1, 222, "when is the meetup again?", is_owner=False, others_present=False,
    )
    assert decision is ResponseDecision.BACKCHANNEL
    decision, _ = engine.should_respond(
        1, 222, "when is the meetup again?", is_owner=False, others_present=True,
    )
    assert decision is ResponseDecision.SKIP


# --- Shared call: the ways in ----------------------------------------------


def test_shared_owner_addressed_responds(smart_config):
    engine = make_engine()
    decision, priority = engine.should_respond(
        1, 111, "alice, what do you think", is_owner=True, others_present=True,
    )
    assert decision is ResponseDecision.RESPOND
    assert priority is ResponsePriority.CRITICAL


def test_shared_nonowner_addressed_responds(smart_config):
    engine = make_engine()
    decision, priority = engine.should_respond(
        1, 222, "hey alice, come play too", is_owner=False, others_present=True,
    )
    assert decision is ResponseDecision.RESPOND
    assert priority is ResponsePriority.HIGH


def test_shared_third_person_soft_response_kept(smart_config):
    """S2 mentions of her name ("like alice ...") still get the soft
    clarification even in shared calls."""
    engine = make_engine()
    decision, _ = engine.should_respond(
        1, 222, "she's basically like alice honestly", is_owner=False,
        others_present=True,
    )
    assert decision is ResponseDecision.RESPOND_SOFT


def test_shared_bargein_still_allowed(smart_config):
    """Interrupting her reply is addressing her -- no name needed."""
    engine = make_engine(TurnState.BOT_RESPONDING)
    decision, priority = engine.should_respond(
        1, 111, "shut up for a sec", is_owner=True, others_present=True,
    )
    assert decision is ResponseDecision.RESPOND
    assert priority is ResponsePriority.CRITICAL


# --- Config interaction ------------------------------------------------------


def test_governor_disabled_restores_old_behavior(smart_config):
    """VOICE_SHARED_CALL_GOVERNOR=false turns the clock back: owner's
    everything gets a reply even with others present."""
    smart_config.setattr(config, "VOICE_SHARED_CALL_GOVERNOR", False)
    engine = make_engine()
    decision, priority = engine.should_respond(
        1, 111, INCIDENT_TEXT, is_owner=True, others_present=True,
    )
    assert decision is ResponseDecision.RESPOND
    assert priority is ResponsePriority.CRITICAL
