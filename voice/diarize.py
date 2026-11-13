"""
voice/diarize.py
=================
Speaker diarization using pyannote. Identifies who is speaking
when in multi-speaker audio.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from bot.logging_setup import get_logger
import config

log = get_logger("voice.diarize")

_pipeline = None
_load_failed = False
_load_lock = threading.Lock()


def _get_pipeline():
    """Lazy-load the pyannote diarization pipeline.

    Guarded by a module lock: several utterances can call diarize()
    concurrently, and two simultaneous loads double the VRAM spike.
    """
    global _pipeline, _load_failed
    with _load_lock:
        return _load_pipeline()


def _load_pipeline():
    global _pipeline, _load_failed
    if _pipeline is not None:
        return _pipeline
    if _load_failed:
        # A previous load attempt failed. Returning None lets diarize()
        # fall back to single-speaker output instead of retrying (and
        # re-spamming the log) on every incoming clip.
        return None

    if not config.VOICE_DIARIZATION:
        log.debug("Diarization disabled in config")
        return None

    if not config.VOICE_HF_TOKEN:
        log.warning("VOICE_HF_TOKEN not set, diarization unavailable")
        return None

    try:
        from pyannote.audio import Pipeline

        log.info("Loading pyannote diarization pipeline...")
        try:
            # pyannote.audio >= 4.0 renamed use_auth_token -> token.
            _pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=config.VOICE_HF_TOKEN,
            )
        except TypeError:
            # pyannote.audio < 4.0 only accepts use_auth_token.
            _pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=config.VOICE_HF_TOKEN,
            )

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _pipeline.to(device)
        _load_failed = False
        log.info(f"Diarization pipeline loaded on {device}")

    except Exception as e:
        log.error(f"Failed to load diarization pipeline: {e}")
        _pipeline = None
        _load_failed = True
        return None

    return _pipeline


def diarize(
    audio: np.ndarray,
    sample_rate: int = 16000,
    num_speakers: Optional[int] = None,
) -> list[dict]:
    """
    Perform speaker diarization on audio.

    Args:
        audio: Float32 audio samples.
        sample_rate: Sample rate.
        num_speakers: Hint for number of speakers (None = auto).

    Returns:
        List of speaker segments with keys:
            - speaker: Speaker label (e.g., "SPEAKER_00").
            - start: Start time in seconds.
            - end: End time in seconds.
    """
    pipeline = _get_pipeline()
    if pipeline is None:
        # Fallback: treat entire audio as one speaker
        duration = len(audio) / sample_rate
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": duration}]

    import torch
    import torchaudio

    # Convert to torch tensor
    waveform = torch.from_numpy(audio).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Run diarization
    try:
        diarization = pipeline(
            {"waveform": waveform, "sample_rate": 16000},
            num_speakers=num_speakers,
        )

        # pyannote.audio 4.x returns a DiarizeOutput dataclass whose
        # .speaker_diarization field holds the classic Annotation object.
        # pyannote.audio 3.x returned the Annotation directly.
        annotation = getattr(diarization, "speaker_diarization", diarization)

        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
            })

        log.debug(f"Diarization found {len(segments)} segments, "
                  f"{len(set(s['speaker'] for s in segments))} speakers")
        return segments

    except Exception as e:
        log.error(f"Diarization failed: {e}")
        duration = len(audio) / sample_rate
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": duration}]


def find_alice_speaker(
    segments: list[dict],
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> str:
    """
    Heuristic to identify which speaker is Alice.
    Uses audio characteristics (pitch, volume) to guess.
    """
    if not segments:
        return "SPEAKER_00"

    # For now, return the speaker with the most airtime
    # (Alice is usually the most active speaker in her voice channel)
    speaker_durations = {}
    for seg in segments:
        spk = seg["speaker"]
        duration = seg["end"] - seg["start"]
        speaker_durations[spk] = speaker_durations.get(spk, 0) + duration

    if speaker_durations:
        return max(speaker_durations, key=speaker_durations.get)
    return segments[0]["speaker"]


def is_available() -> bool:
    """Check if diarization is available."""
    if not config.VOICE_DIARIZATION:
        return False
    return _pipeline is not None


def warmup() -> None:
    """Pre-load the diarization pipeline in the background.

    Called at bot startup so the first spoken utterance of a session
    doesn't pay the ~15s model load inline between transcription and
    the response decision. Safe to call repeatedly; failures are
    logged once and degrade to the single-speaker fallback.
    """
    if config.VOICE_DIARIZE_MODE == "none" or not config.VOICE_HF_TOKEN:
        return
    try:
        _get_pipeline()
    except Exception as e:
        log.warning(f"Diarization warmup failed: {e}")


def unload() -> None:
    """Unload the diarization pipeline."""
    global _pipeline, _load_failed
    _pipeline = None
    _load_failed = False
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    log.info("Diarization pipeline unloaded.")
