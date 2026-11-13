"""
Regression test: streamed (first) sentences must still reach lip-sync.

2026-11-11 field bug: the SoVITS streaming branch of
streaming_tts._synthesize_and_add returned before the lip-sync publish,
so the avatar mouth never moved for streamed sentences -- on the desktop
that meant every single-sentence reply (the most common case) played
with a static mouth, and lip-sync subscribers saw nothing at all.
"""
import asyncio
import io
import wave

import numpy as np
import pytest

import config
import voice.lip_sync as lip_sync
import voice.sovits as sovits
import voice.streaming_tts as streaming_tts
import voice.tts as tts


class _FakeSource:
    def __init__(self) -> None:
        self.added: list = []

    def add_pcm(self, chunk, source_sr) -> None:
        self.added.append((chunk, source_sr))

    def add_wav_bytes(self, wav) -> None:
        self.added.append(("wav", wav))


@pytest.fixture
def events():
    got: list = []
    lip_sync.subscribe(got.append)
    yield got
    lip_sync.unsubscribe(got.append)


def _collect(events) -> list:
    return list(events)


def test_streamed_sentence_publishes_to_lip_sync(monkeypatch, events):
    monkeypatch.setattr(config, "VOICE_LIP_SYNC", True)
    monkeypatch.setattr(config, "VOICE_TTS_ENGINE", "sovits")
    monkeypatch.setattr(config, "VOICE_SOVITS_STREAMING", True)

    async def fake_stream(text, text_lang=None):
        # Two streamed PCM chunks @ 32 kHz, mono float32.
        yield np.zeros(3200, dtype=np.float32)
        yield np.zeros(3200, dtype=np.float32)

    monkeypatch.setattr(sovits, "generate_stream", fake_stream)

    src = asyncio.run(
        streaming_tts._synthesize_and_add(
            "Hi Father.", 0, _FakeSource(), "neutral", use_sovits_stream=True
        )
    )
    got = _collect(events)
    assert src > 0
    assert len(got) == 1, f"streamed sentence must publish once, got {got}"
    ev = got[0]
    assert ev["text"] == "Hi Father."
    assert ev["engine"] == "sovits"
    assert ev["sr"] == 32000
    assert ev["duration"] > 0
    # The wav_bytes must be a decodable WAV (the desktop decodes it).
    with wave.open(io.BytesIO(ev["wav_bytes"]), "rb") as wf:
        assert wf.getnframes() == 6400


def test_legacy_sentence_publishes_to_lip_sync(monkeypatch, events):
    monkeypatch.setattr(config, "VOICE_LIP_SYNC", True)
    monkeypatch.setattr(config, "VOICE_TTS_ENGINE", "sovits")

    wav = tts.array_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)

    async def fake_tts(text):
        return (wav, 24000, "sovits")

    monkeypatch.setattr(tts, "generate_tts", fake_tts)

    asyncio.run(
        streaming_tts._synthesize_and_add(
            "Sure, what's on your mind?", 0, _FakeSource(), "neutral"
        )
    )
    got = _collect(events)
    assert len(got) == 1
    assert got[0]["text"] == "Sure, what's on your mind?"
    assert got[0]["duration"] == pytest.approx(1.0, abs=0.05)


def test_publish_disabled_when_voice_lip_sync_off(monkeypatch, events):
    monkeypatch.setattr(config, "VOICE_LIP_SYNC", False)
    monkeypatch.setattr(config, "VOICE_TTS_ENGINE", "sovits")

    wav = tts.array_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)

    async def fake_tts(text):
        return (wav, 24000, "sovits")

    monkeypatch.setattr(tts, "generate_tts", fake_tts)
    asyncio.run(
        streaming_tts._synthesize_and_add(
            "Quiet mouth.", 0, _FakeSource(), "neutral"
        )
    )
    assert _collect(events) == []
