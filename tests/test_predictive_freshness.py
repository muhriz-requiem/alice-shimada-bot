"""
Tests for the live-mode latencies around check_speech_complete:
the adaptive silence-threshold override (private vs shared calls) and
the AudioBuffer revision counter used by predictive STT.
"""

from __future__ import annotations


import numpy as np

import config
from voice.sink import AudioBuffer, AliceVoiceSink


GUILD = 1
USER = 7


def _sink_with_buffer(duration: float, silence: float) -> AliceVoiceSink:
    """A sink whose buffer fallback reports fixed durations."""
    sink = AliceVoiceSink(guild_id=GUILD)
    sink._vads.pop(USER, None)  # force the buffer-based fallback path

    class _FakeBuffer:
        def __init__(self):
            self.revision = 0

        @staticmethod
        def get_duration() -> float:
            return duration

        @property
        def silence_duration(self) -> float:
            return silence

    sink.buffers[USER] = _FakeBuffer()
    return sink


class TestAdaptiveThreshold:
    def test_override_param_used(self):
        sink = _sink_with_buffer(duration=1.0, silence=0.6)
        # Default (0.8): not complete yet.
        assert not sink.check_speech_complete(USER)
        # Private-call override (0.5): the same silence now counts.
        assert sink.check_speech_complete(USER, silence_threshold=0.5)

    def test_none_uses_configured_threshold(self):
        sink = _sink_with_buffer(duration=1.0, silence=0.6)
        assert (
            sink.check_speech_complete(USER, silence_threshold=None)
            == (0.6 >= config.VOICE_SILENCE_THRESHOLD)
        )

    def test_short_speech_always_rejected(self):
        sink = _sink_with_buffer(duration=0.1, silence=0.9)
        assert not sink.check_speech_complete(USER, silence_threshold=0.5)

    def test_max_duration_completes_even_mid_speech(self):
        sink = AliceVoiceSink(guild_id=GUILD)
        sink._vads.pop(USER, None)
        sink._max_speech = 0.05
        # Separate rolling-trim semantics from the gate: a huge buffer
        # limit so the over-max audio stays instead of being trimmed.
        sink.buffers[USER] = AudioBuffer(max_duration=999.0)
        sink.buffers[USER].add(
            np.zeros(int(sink.buffers[USER].sample_rate * 0.7), dtype=np.float32)
        )
        assert sink.check_speech_complete(USER, silence_threshold=0.5)


class TestAudioBufferRevision:
    def test_revision_increments_and_survives_clear(self):
        buf = AudioBuffer()
        r0 = buf.revision
        buf.add(np.zeros(100, dtype=np.float32))
        assert buf.revision == r0 + 1
        buf.clear()
        assert buf.revision == r0 + 1  # never reset by clear
        buf.add(np.zeros(100, dtype=np.float32))
        assert buf.revision == r0 + 2

    def test_snapshot_returns_revision(self):
        buf = AudioBuffer()
        audio = np.arange(16, dtype=np.float32) / 32768.0
        buf.add(audio)
        snap, rev = buf.get_snapshot()
        assert np.allclose(snap, audio)
        assert rev == buf.revision
        # snapshot must not consume the buffer
        snap2, _ = buf.get_snapshot()
        assert np.allclose(snap2, audio)

    def test_observation_and_revision(self):
        sink = AliceVoiceSink(guild_id=GUILD)
        sink._vads.pop(USER, None)
        sink.buffers[USER] = AudioBuffer()
        sink.buffers[USER].add(np.zeros(10, dtype=np.float32))
        assert sink.observation(USER) is None  # below min_speech
        sink.buffers[USER].add(
            np.zeros(int(sink._min_speech * sink.buffers[USER].sample_rate) + 1000, dtype=np.float32)
        )
        obs = sink.observation(USER)
        assert obs is not None
        assert obs["revision"] == 2
        assert sink.get_revision(USER) == obs["revision"]
        assert sink.snapshot_audio(USER) is not None
