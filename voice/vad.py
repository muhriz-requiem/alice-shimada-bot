"""
voice/vad.py
=============
Voice Activity Detection using Silero VAD.
Replaces the simple RMS-based VAD with a proper neural VAD
for more accurate speech/silence detection.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from bot.logging_setup import get_logger
import config

log = get_logger("voice.vad")

# Silero VAD model (loaded lazily)
_model = None
_model_lock = None


def _load_model():
    """Load the Silero VAD model lazily."""
    global _model, _model_lock
    if _model is not None:
        return _model

    try:
        import torch
        from silero_vad import load_silero_vad, get_speech_timestamps

        # Load model (downloads on first use)
        _model = load_silero_vad(onnx=True)
        log.info("Silero VAD model loaded")
        return _model
    except Exception as e:
        log.warning(f"Failed to load Silero VAD: {e}")
        return None


def preload() -> None:
    """Preload the VAD model at startup so the first packet doesn't block."""
    _load_model()


def is_model_loaded() -> bool:
    return _model is not None


class SileroVAD:
    """
    Wrapper around Silero VAD for voice activity detection.
    
    Supports:
    - Chunk-based detection (16kHz, 512 samples = 32ms)
    - Rolling window for faster state transitions
    - Energy-based pre-filtering to skip silence
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self._model = None
        self._last_speech_at: float = 0
        self._is_speaking: bool = False
        self._silence_start: float = 0
        self._speech_start: float = 0

    def _ensure_model(self):
        if self._model is None:
            # Never block the audio hot path: if global model isn't loaded
            # yet, leave self._model None so is_speech() uses RMS fallback.
            # Preloading happens via voice.vad.preload() at pipeline init.
            global _model
            if _model is not None:
                self._model = _model
            # Do NOT call _load_model() here — it can take 100s+ in-bot
            # and would freeze the PacketRouter thread mid-packet.

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Check if an audio chunk contains speech.

        Args:
            audio_chunk: Float32 audio at 16kHz (or resampled internally)

        Returns:
            True if speech detected
        """
        self._ensure_model()
        if self._model is None:
            # Fallback to RMS if Silero not available
            return self._rms_fallback(audio_chunk)

        try:
            import torch

            # Ensure correct shape for Silero: (1, num_samples)
            if audio_chunk.ndim == 1:
                audio_tensor = torch.from_numpy(audio_chunk).float()
            else:
                audio_tensor = torch.from_numpy(audio_chunk.flatten()).float()

            # Silero expects exactly 512 samples at 16kHz
            if len(audio_tensor) != 512:
                # Pad or truncate to 512
                if len(audio_tensor) < 512:
                    audio_tensor = torch.nn.functional.pad(audio_tensor, (0, 512 - len(audio_tensor)))
                else:
                    audio_tensor = audio_tensor[:512]

            # Get speech probability
            speech_prob = self._model(audio_tensor, self.sample_rate).item()

            now = time.time()
            has_speech = speech_prob >= self.threshold

            if has_speech:
                self._last_speech_at = now
                if not self._is_speaking:
                    self._speech_start = now
                    self._is_speaking = True
                self._silence_start = 0
            else:
                if self._is_speaking and self._silence_start == 0:
                    self._silence_start = now

            return has_speech

        except Exception as e:
            log.debug(f"Silero VAD error: {e}, using RMS fallback")
            return self._rms_fallback(audio_chunk)

    def _rms_fallback(self, audio_chunk: np.ndarray) -> bool:
        """Simple RMS-based fallback if Silero not available."""
        rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2)))
        threshold = config.VOICE_VAD_THRESHOLD / 1000.0
        return rms > threshold

    @property
    def is_speaking(self) -> bool:
        """Whether continuous speech is in progress."""
        if not self._is_speaking:
            return False
        if self._silence_start > 0:
            silence_dur = time.time() - self._silence_start
            return silence_dur < config.VOICE_SILENCE_THRESHOLD
        return True

    @property
    def silence_duration(self) -> float:
        """Duration of current silence period."""
        if self._silence_start > 0:
            return time.time() - self._silence_start
        if self._last_speech_at > 0:
            return time.time() - self._last_speech_at
        return float("inf")

    @property
    def speech_duration(self) -> float:
        """Duration of current speech period."""
        if not self._is_speaking:
            return 0
        return time.time() - self._speech_start

    def reset(self) -> None:
        """Reset VAD state."""
        self._is_speaking = False
        self._silence_start = 0
        self._speech_start = 0
        self._last_speech_at = 0
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass

    @property
    def is_available(self) -> bool:
        """Check if Silero VAD is loaded."""
        self._ensure_model()
        return self._model is not None
