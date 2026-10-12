"""
voice/diarize.py
=================
Speaker diarization using pyannote. Identifies who is speaking
when in multi-speaker audio.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from bot.logging_setup import get_logger
import config

log = get_logger("voice.diarize")

_pipeline = None


def _get_pipeline():
    """Lazy-load the pyannote diarization pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not config.VOICE_DIARIZATION:
        log.debug("Diarization disabled in config")
        return None

    if not config.VOICE_HF_TOKEN:
        log.warning("VOICE_HF_TOKEN not set, diarization unavailable")
        return None

    try:
        from pyannote.audio import Pipeline

        log.info("Loading pyannote diarization pipeline...")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=config.VOICE_HF_TOKEN,
        )

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _pipeline.to(device)
        log.info(f"Diarization pipeline loaded on {device}")

    except Exception as e:
        log.error(f"Failed to load diarization pipeline: {e}")
        _pipeline = None
        raise

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

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
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


def unload() -> None:
    """Unload the diarization pipeline."""
    global _pipeline
    _pipeline = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    log.info("Diarization pipeline unloaded.")
