"""
voice/preprocess.py
===================
Audio preprocessing shared by the live pipeline and the offline STT eval
(manage.py eval-stt). One code path so eval numbers match production.

Modes (VOICE_STT_PREPROCESS / eval matrix):
- "off":     resample only
- "loudnorm": RMS gain for quiet mics (current code only normalizes when
  peak > 1.0, so quiet input gets no gain)
- "denoise":  noisereduce spectral gating (helps noisy/unclear mics)
- "both":    denoise + loudnorm
"""

from __future__ import annotations

import numpy as np

_TARGET_RMS = 0.1
_MAX_GAIN = 10.0


def noise_floor_rms(audio: np.ndarray, sample_rate: int, frame_ms: int = 20) -> float:
    """RMS of the quietest 10% of frames — a cheap noise-floor estimate.

    Computed on the raw input at capture time so the eval report can
    roughly separate accent errors from noise errors per clip.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if not audio.size:
        return 0.0
    frame = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = max(1, len(audio) // frame)
    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    k = max(1, int(np.ceil(n_frames * 0.1)))
    quietest = np.partition(rms, k - 1)[:k]
    return float(quietest.mean())


def loudnorm(audio: np.ndarray, target_rms: float = _TARGET_RMS) -> np.ndarray:
    """Scale audio to a target RMS so quiet mics aren't under-amplified.

    The live pipeline only normalizes when peak > 1.0, so low-gain input
    reaches the STT quiet. Gain is capped to avoid exploding pure noise.
    """
    rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
    if rms < 1e-6:
        return audio
    gain = min(target_rms / rms, _MAX_GAIN)
    out = audio * gain
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return out


def denoise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Spectral-gate denoise via noisereduce (no-op if unavailable)."""
    try:
        from noisereduce import reduce_noise

        return reduce_noise(y=audio, sr=sample_rate, prop_decrease=0.8)
    except Exception:
        return audio


def _resample(audio: np.ndarray, sample_rate: int, target_sr: int = 16000) -> np.ndarray:
    if sample_rate == target_sr:
        return audio
    import math
    import scipy.signal
    g = math.gcd(int(sample_rate), int(target_sr))
    return scipy.signal.resample_poly(
        audio, target_sr // g, sample_rate // g
    ).astype(np.float32)


def prepare(audio: np.ndarray, sample_rate: int, mode: str = "off") -> np.ndarray:
    """Apply preprocessing and return 16k mono float32 audio.

    Called by both the live pipeline and manage.py eval-stt so eval
    numbers match production exactly.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio.astype(np.float32))
    if mode in ("denoise", "both"):
        audio = denoise(audio, sample_rate)
    if mode in ("loudnorm", "both"):
        audio = loudnorm(audio)
    return _resample(audio, sample_rate, 16000)
