"""
voice/tts.py
============
Text-to-speech dispatch. Routes to GPT-SoVITS (config.VOICE_TTS_ENGINE
= "sovits") or Edge-TTS with character-appropriate voice, rate, and
pitch settings; Edge-TTS acts as the silent fallback engine. Any
stammer fragment (F-Father) is stripped at entry so TTS never reads a
lone letter as a letter name; ellipses pass through as natural pauses.
"""

from __future__ import annotations

import asyncio
import io
import re
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from bot.logging_setup import get_logger
import config

log = get_logger("voice.tts")

# Mirrors ai.postprocess._STAMMER_PATTERN (kept local to avoid
# importing the ai package from the voice layer).
_STAMMER_RE = re.compile(r"(?<![A-Za-z])([A-Za-z])-([A-Z])")


def normalize_for_tts(text: str) -> str:
    """
    Strip any stammer fragments before synthesis: "F-Father" -> "Father",
    "I-I'm" -> "I'm".

    Stammers are a text-only affectation now (her tone in call already
    carries the shyness); if one does reach the TTS layer, the lone
    letter would be spoken as its letter name ("eff Father"), so the
    fragment is simply dropped. Requiring a capital letter after the
    hyphen protects legit hyphenated words (T-shirt, T-34, Boko-chan
    is safe as its first segment is multi-letter). Ellipses are left
    untouched -- engines render them as a quiet pause.
    """
    return _STAMMER_RE.sub(r"\2", text)


async def generate_tts(
    text: str,
    voice: Optional[str] = None,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
) -> tuple[bytes, int, str]:
    """
    Generate TTS audio and report which engine produced it.

    Routes on config.VOICE_TTS_ENGINE. When the engine is "sovits" and
    the GPT-SoVITS API fails, silently falls back to Edge-TTS.

    Returns:
        Tuple of (wav_bytes, sample_rate, engine_used).
    """
    normalized = normalize_for_tts(text)
    if normalized != text:
        log.debug(f"Stammer strip: {text!r} -> {normalized!r}")
        text = normalized

    if config.VOICE_TTS_ENGINE == "sovits":
        try:
            from voice import sovits

            wav_bytes, sr = await sovits.generate_async(text)
            log.debug(f"Synthesized via sovits ({len(wav_bytes)} bytes)")
            return wav_bytes, sr, "sovits"
        except Exception as e:
            log.warning(f"SoVITS TTS failed, falling back to edge: {e}")

    wav_bytes, sr = await _edge_generate_async(text, voice, rate, pitch)
    return wav_bytes, sr, "edge"


async def generate_async(
    text: str,
    voice: Optional[str] = None,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
) -> tuple[bytes, int]:
    """
    Generate TTS audio asynchronously.

    Args:
        text: Text to speak.
        voice: Edge-TTS voice name (default: config.VOICE_TTS_VOICE).
        rate: SSML rate adjustment (default: config.VOICE_TTS_RATE).
        pitch: SSML pitch adjustment (default: config.VOICE_TTS_PITCH).

    Returns:
        Tuple of (wav_bytes, sample_rate).
    """
    wav_bytes, sr, _engine = await generate_tts(text, voice, rate, pitch)
    return wav_bytes, sr


async def _edge_generate_async(
    text: str,
    voice: Optional[str] = None,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
) -> tuple[bytes, int]:
    """Edge-TTS generation (original behavior)."""
    import edge_tts

    voice = voice or config.VOICE_TTS_VOICE
    rate = rate or config.VOICE_TTS_RATE
    pitch = pitch or config.VOICE_TTS_PITCH

    log.debug(f"Generating TTS: voice={voice}, rate={rate}, pitch={pitch}")
    log.debug(f"Text: {text[:100]}...")

    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)

    audio_data = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.write(chunk["data"])

    audio_bytes = audio_data.getvalue()
    if not audio_bytes:
        log.error("Edge-TTS returned empty audio")
        raise ValueError("TTS returned empty audio")

    # Edge-TTS outputs MP3; convert to WAV for processing
    wav_bytes = await _mp3_to_wav(audio_bytes)

    log.debug(f"Generated {len(wav_bytes)} bytes of WAV audio")
    return wav_bytes, 24000


def generate_sync(
    text: str,
    voice: Optional[str] = None,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
) -> tuple[bytes, int]:
    """Synchronous wrapper for generate_async."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            generate_async(text, voice, rate, pitch)
        )
    finally:
        loop.close()


async def _mp3_to_wav(mp3_bytes: bytes) -> bytes:
    """Convert MP3 bytes to WAV bytes using ffmpeg."""
    from voice.utils import find_ffmpeg
    ffmpeg = find_ffmpeg()
    proc = await asyncio.create_subprocess_exec(
        ffmpeg, "-i", "pipe:0",
        "-ar", "24000", "-ac", "1", "-f", "wav",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate(input=mp3_bytes)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg MP3->WAV conversion failed")
    return stdout


def wav_bytes_to_array(wav_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """Convert WAV bytes to float32 numpy array."""
    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sr != target_sr:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * target_sr / sr))

    return audio


def array_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert float32 numpy array to WAV bytes."""
    import wave

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return buf.getvalue()


def is_available() -> bool:
    """Check if Edge-TTS is available."""
    try:
        import edge_tts
        return True
    except ImportError:
        return False
