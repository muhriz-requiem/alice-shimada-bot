"""
tests/test_preprocess.py
========================
Covers voice/preprocess.py: the shared preprocessing path used by both
the live pipeline and manage.py eval-stt.

- loudnorm boosts quiet audio to a target RMS (and caps gain / clips)
- denoise returns same-shape audio (no-op when noisereduce missing)
- prepare() returns 16k mono float32 for every mode
"""

from __future__ import annotations

import numpy as np
import pytest

from voice import preprocess


def _sine(seconds=1.0, sr=48000, freq=440.0, amp=0.5):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_loudnorm_boosts_quiet_audio():
    quiet = _sine(amp=0.02)
    out = preprocess.loudnorm(quiet)
    rms = np.sqrt(np.mean(out * out))
    assert abs(rms - preprocess._TARGET_RMS) < 0.02


def test_loudnorm_caps_gain_on_silence():
    silence = np.zeros(4800, dtype=np.float32)
    out = preprocess.loudnorm(silence)
    assert np.allclose(out, 0.0)


def test_loudnorm_clips_to_unit_peak():
    loud = _sine(amp=0.9)
    out = preprocess.loudnorm(loud, target_rms=0.5)
    assert np.max(np.abs(out)) <= 1.0


def test_denoise_same_shape():
    audio = _sine()
    out = preprocess.denoise(audio, 48000)
    assert out.shape == audio.shape
    assert out.dtype == np.float32


@pytest.mark.parametrize("mode", ["off", "loudnorm", "denoise", "both"])
def test_prepare_returns_16k_mono(mode):
    audio = _sine(seconds=1.0, sr=48000)
    out = preprocess.prepare(audio, 48000, mode)
    assert out.ndim == 1
    assert len(out) == 16000
    assert out.dtype == np.float32


def test_prepare_stereo_downmixes():
    sr = 48000
    t = np.arange(sr) / sr
    # codebase convention: (samples, channels)
    stereo = np.stack(
        [np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)], axis=1
    ).astype(np.float32)
    out = preprocess.prepare(stereo, sr, "off")
    assert out.ndim == 1
    assert len(out) == 16000


def test_prepare_already_16k():
    audio = _sine(seconds=0.5, sr=16000)
    out = preprocess.prepare(audio, 16000, "loudnorm")
    assert len(out) == 8000
