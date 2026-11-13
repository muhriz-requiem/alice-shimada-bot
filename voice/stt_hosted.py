"""
voice/stt_hosted.py
===================
Hosted Whisper transcription through Groq's OpenAI-compatible
/audio/transcriptions endpoint. Behind VOICE_STT_HOSTED (default: off) --
the local faster-whisper engine stays the default because it is free,
GPU-local, and never sends speech off the machine.

Why it exists here: the live-mode pipeline wants every revelation of
latency it can get. A hosted engine decodes nothing locally (no GPU
queueing behind TTS or diarization) and the network round-trip is often
faster than a local whisper-medium decode; the module returns the same
{"text", "language", ...} shape as voice.stt.transcribe so the pipeline
can swap engines without changes.

Failure contract: transcribe() returns None on ANY problem (missing
key, HTTP error, timeout, unparsable body) -- the caller then falls
back to the local engine. This module never raises.
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from typing import Optional

import numpy as np

from bot.logging_setup import get_logger
import config

log = get_logger("voice.stt_hosted")

# Per-null-result cooldown: when the endpoint is missing/down there is
# no point hammering it on every utterance. Transcriptions are tried at
# most once per this interval; inside the window the caller goes
# straight to the local engine.
_RETRY_COOLDOWN_SECONDS = 60.0

_last_failure: float = 0.0


def is_configured() -> bool:
    """True when hosted STT is enabled and an API key exists."""
    return bool(config.VOICE_STT_HOSTED and config.GROQ_API_KEY)


def _cooldown_active() -> bool:
    return (time.monotonic() - _last_failure) < _RETRY_COOLDOWN_SECONDS


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono audio as a 16-bit PCM WAV in memory."""
    pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = np.max(np.abs(pcm)) if pcm.size else 0.0
    if peak > 1.0:
        pcm = pcm / peak
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes((pcm * 32767.0).astype(np.int16).tobytes())
    return buf.getvalue()


async def transcribe(
    audio: np.ndarray,
    sample_rate: int = 16000,
    language: Optional[str] = None,
) -> Optional[dict]:
    """
    Transcribe via Groq's hosted Whisper. Same return shape as
    voice.stt.transcribe; None on any failure (caller falls back).

    language: None follows config.VOICE_STT_LANGUAGE ("auto" lets the
    API detect, "en"/"ja" pass an explicit language code -- note the
    API wants "en" style codes just like the local engine).
    """
    global _last_failure

    if not is_configured():
        return None
    if audio is None or getattr(audio, "size", 0) == 0:
        return None
    if _cooldown_active():
        return None

    lang = (language or config.VOICE_STT_LANGUAGE or "auto").lower()
    url = config.GROQ_BASE_URL.rstrip("/") + "/audio/transcriptions"
    model = config.VOICE_STT_HOSTED_MODEL

    wav = await asyncio.to_thread(_wav_bytes, audio, sample_rate)
    suffix = ".wav"
    t0 = time.monotonic()

    try:
        import aiohttp

        form = aiohttp.FormData()
        # aiohttp needs (bytes, filename, content_type) to send raw data.
        form.add_field(
            "file",
            wav,
            filename=f"utterance{suffix}",
            content_type="audio/wav",
        )
        form.add_field("model", model)
        form.add_field("response_format", "verbose_json")
        form.add_field("temperature", "0.0")
        if lang and lang != "auto":
            form.add_field("language", lang)

        timeout = aiohttp.ClientTimeout(total=10.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                data=form,
                headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            ) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:200]
                    _last_failure = time.monotonic()
                    log.warning(
                        f"Hosted STT HTTP {resp.status}: {body}"
                    )
                    return None
                data = await resp.json(content_type=None)

        text = (data.get("text") or "").strip()
        if not text:
            # A valid empty transcript: keep it -- empty means silence,
            # and the pipeline's own empty-check handles skipping.
            log.info(f"Hosted STT: empty transcript ({time.monotonic()-t0:.2f}s)")
            return {"text": "", "language": lang if lang != "auto" else "en"}

        detected = data.get("language") or (lang if lang != "auto" else "en")
        result = {
            "text": text,
            "language": detected,
            "language_probability": 1.0,
            "segments": data.get("segments") or [],
            "engine": "groq-hosted",
        }
        log.info(
            f"Hosted STT ok ({time.monotonic()-t0:.2f}s): "
            f"'{text[:60]}' (lang={detected})"
        )
        return result

    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - hosted is always best-effort
        _last_failure = time.monotonic()
        log.warning(f"Hosted STT failed ({e.__class__.__name__}: {e})")
        return None
