"""
tests/test_desktop_app.py
=========================
Phase 2, tray entry (`desktop/app.py`): pure-logic pieces that do not
need a real tray, a display or audio devices.

- Single-instance mutex (second acquire blocked, works after release)
- Runtime STT engine flip (`set_stt_engine`)
- CompanionSession mute/status behavior before any boot
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import config
from desktop.app import (
    acquire_single_instance,
    release_single_instance,
    set_stt_engine,
)
from desktop.voice_app import CompanionSession

# --- single instance --------------------------------------------------

def _external_instance_holds_mutex() -> bool:
    """True when a LIVE companion (tray) already owns the mutex -- the
    single-instance semantics can't be tested in-process then."""
    handle = acquire_single_instance()
    if handle is None:
        return True
    release_single_instance(handle)
    return False


def test_single_instance_blocks_second_acquire():
    if _external_instance_holds_mutex():
        pytest.skip("live desktop companion holds the single-instance mutex")
    handle = acquire_single_instance()
    assert handle is not None
    try:
        second = acquire_single_instance()
        assert second is None
    finally:
        if handle is not None:
            release_single_instance(handle)


def test_single_instance_after_release():
    if _external_instance_holds_mutex():
        pytest.skip("live desktop companion holds the single-instance mutex")
    handle = acquire_single_instance()
    assert handle is not None
    release_single_instance(handle)
    # The mutex OBJECT is gone (closed): the next launch can acquire it.
    handle2 = acquire_single_instance()
    assert handle2 is not None
    release_single_instance(handle2)


# --- STT engine runtime switch ----------------------------------------

def test_set_stt_engine_flips_config():
    initial = config.VOICE_STT_HOSTED
    try:
        # Flip away from the current value.
        target = "local" if initial else "hosted"
        assert set_stt_engine(target) is True
        assert config.VOICE_STT_HOSTED is (not initial)

        # Same value again: no-op, reports no change.
        assert set_stt_engine(target) is False
        assert config.VOICE_STT_HOSTED is (not initial)
    finally:
        config.VOICE_STT_HOSTED = initial


def test_set_stt_engine_accepts_aliases():
    initial = config.VOICE_STT_HOSTED
    try:
        for alias, expected in (("groq", True), ("cloud", True),
                                ("cpu", False), ("offline", False)):
            set_stt_engine(alias)
            assert config.VOICE_STT_HOSTED is expected
    finally:
        config.VOICE_STT_HOSTED = initial


def test_set_stt_engine_rejects_unknown():
    with pytest.raises(ValueError):
        set_stt_engine("moonshine-jazz")


# --- CompanionSession (no devices) -------------------------------------

def test_session_mute_and_status_before_start():
    session = CompanionSession()
    assert session.status() == {"state": "off", "error": None, "muted": False}

    session.set_muted(True)
    assert session.muted is True            # queued for boot; and...
    assert session._booted is None          # ...no mic yet, safely ignored
    assert session.status()["muted"] is True

    session.set_muted(False)
    assert session.status()["muted"] is False


def test_session_stop_before_start_is_noop():
    session = CompanionSession()
    session.stop()  # never started: must not raise
    assert session.status()["state"] == "off"
