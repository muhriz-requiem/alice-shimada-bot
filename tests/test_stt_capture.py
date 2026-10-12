"""
tests/test_stt_capture.py
=========================
Covers the STT eval capture hook (VOICE_STT_CAPTURE): raw voice clips +
JSON sidecars land in STT_EVAL_DIR so manage.py eval-stt can score
candidate engines on the user's own accented speech.

- capture writes a readable WAV + sidecar with the fields the eval needs
- the disk cap is respected
- the hook in process_audio fires only when enabled
- capture never breaks transcription (errors are swallowed)
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
import scipy.io.wavfile as wavfile

import config
import voice.pipeline as pipeline
from voice.pipeline import VoicePipeline, _capture_clip


@pytest.fixture()
def eval_dir(tmp_path, monkeypatch):
    d = tmp_path / "clips"
    d.mkdir()
    monkeypatch.setattr(config, "STT_EVAL_DIR", str(d))
    monkeypatch.setattr(config, "VOICE_STT_CAPTURE_LIMIT", 5)
    return d


def _sine(seconds=1.0, sr=48000, freq=440.0, amp=0.5):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_capture_writes_wav_and_sidecar(eval_dir):
    audio = _sine()
    result = {"text": "hello there", "language": "en", "segments": []}
    _capture_clip(123, 456, audio, 48000, result)

    wavs = list(eval_dir.glob("clip_*.wav"))
    sides = list(eval_dir.glob("clip_*.json"))
    assert len(wavs) == 1
    assert len(sides) == 1

    sr, data = wavfile.read(str(wavs[0]))
    assert sr == 48000
    assert data.dtype == np.int16
    assert len(data) == 48000  # 1s

    side = json.loads(sides[0].read_text(encoding="utf-8"))
    assert side["guild_id"] == 123
    assert side["user_id"] == 456
    assert side["sample_rate"] == 48000
    assert side["dur_s"] == 1.0
    assert side["text"] == "hello there"
    assert side["language"] == "en"
    assert side["engine"] == config.VOICE_STT_ENGINE
    assert side["noise_floor_rms"] >= 0.0


def test_capture_keeps_empty_text_clips(eval_dir):
    # Accent-caused silence (empty transcript) is a key failure case —
    # the clip must still be saved for the eval.
    _capture_clip(1, 1, _sine(), 48000, {"text": "", "language": "en"})
    sides = list(eval_dir.glob("clip_*.json"))
    assert len(sides) == 1
    assert json.loads(sides[0].read_text(encoding="utf-8"))["text"] == ""


def test_capture_normalizes_peak_over_one(eval_dir):
    _capture_clip(1, 1, _sine(amp=2.0), 48000, {"text": "", "language": "en"})
    sr, data = wavfile.read(str(next(eval_dir.glob("clip_*.wav"))))
    assert data.max() <= 32767


def test_capture_respects_limit(eval_dir):
    for i in range(5):
        (eval_dir / f"clip_pre_{i}.wav").write_bytes(b"x")
    _capture_clip(1, 1, _sine(), 48000, {"text": "", "language": "en"})
    assert len(list(eval_dir.glob("clip_*.wav"))) == 5  # only the pre-made ones


def test_capture_swallows_errors(eval_dir, tmp_path, monkeypatch, caplog):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    monkeypatch.setattr(config, "STT_EVAL_DIR", str(blocker / "clips"))

    import logging

    with caplog.at_level(logging.WARNING, logger="voice.pipeline"):
        _capture_clip(1, 1, _sine(), 48000, {"text": "", "language": "en"})

    assert "STT eval capture failed" in caplog.text
    assert list(eval_dir.glob("clip_*.wav")) == []


def test_process_audio_captures_when_enabled(eval_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(config, "VOICE_STT_CAPTURE", True)
    monkeypatch.setattr(pipeline, "_capture_clip", lambda *a: calls.append(a))

    def fake_transcribe(audio, sr, language=None):
        return {"text": "hi alice", "language": "en", "segments": []}

    monkeypatch.setattr(pipeline.stt, "transcribe", fake_transcribe)

    vp = VoicePipeline(bot=None)
    text, lang = asyncio.run(vp.process_audio(123, 456, _sine(), 48000))

    assert (text, lang) == ("hi alice", "en")
    assert len(calls) == 1
    guild_id, user_id, audio, sample_rate, result = calls[0]
    assert (guild_id, user_id, sample_rate) == (123, 456, 48000)
    assert result["text"] == "hi alice"


def test_process_audio_skips_capture_when_disabled(eval_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(config, "VOICE_STT_CAPTURE", False)
    monkeypatch.setattr(pipeline, "_capture_clip", lambda *a: calls.append(a))

    def fake_transcribe(audio, sr, language=None):
        return {"text": "hi alice", "language": "en", "segments": []}

    monkeypatch.setattr(pipeline.stt, "transcribe", fake_transcribe)

    vp = VoicePipeline(bot=None)
    asyncio.run(vp.process_audio(123, 456, _sine(), 48000))

    assert calls == []
