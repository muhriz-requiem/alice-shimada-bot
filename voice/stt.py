"""
voice/stt.py
============
Speech-to-text with two engines behind config.VOICE_STT_ENGINE:

- "moonshine" (default): Moonshine models via transformers. Fixed
  language only (VOICE_STT_LANGUAGE=en/ja) -- no language detection
  at all, so nothing is spent on it -- and no 30s zero-padded window,
  which makes it fast and accurate on short VAD-gated utterances.
- "whisper": faster-whisper, with auto-detection when the language
  resolves to "auto". Also the silent fallback if moonshine fails.

Both engines return the same result dict shape.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from bot.logging_setup import get_logger
import config

# Models are already cached locally; skip all Hugging Face Hub traffic
# on load (HF_TOKEN nag, repo-rename 307 redirects, optional-file 404
# HEADs -- ~5s of round-trips per startup) and let STT work offline.
# setdefault so an explicit HF_HUB_OFFLINE=0 in .env (e.g. to download
# a new model once) still wins -- the value must be in the environment
# before huggingface_hub is first imported below.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

log = get_logger("voice.stt")

_model = None  # faster-whisper WhisperModel (lazy)
_moonshine_cache: dict[str, tuple] = {}  # lang -> (model, processor)

# Same word repeated this many times in a row = decode loop, discard.
_REPEAT_LIMIT = 6

# Whisper beam size (overridable by manage.py eval-stt --beam).
_WHISPER_BEAM = 3


def _engine() -> str:
    return (getattr(config, "VOICE_STT_ENGINE", "moonshine") or "moonshine").strip().lower()


def _resolved_language(language: Optional[str] = None) -> Optional[str]:
    """
    Resolve the language for this transcription.

    Returns a 2-letter code ("en"/"ja") for fixed-language mode (no
    detection happens), or None for auto-detect ("auto"/empty/invalid).
    """
    lang = language or config.VOICE_STT_LANGUAGE or None
    if not lang:
        return None
    lang = lang.strip().lower()
    if lang in ("auto", "any"):
        return None
    if len(lang) == 2 and lang.isalpha():
        return lang
    log.debug(f"Ignoring invalid STT language code: {lang!r}")
    return None


def _whisper_model():
    """Lazy-load the Whisper model (heavy, only load once)."""
    global _model
    if _model is not None:
        return _model

    try:
        from faster_whisper import WhisperModel

        log.info(
            f"Loading Whisper model: {config.VOICE_STT_MODEL} "
            f"(device={config.VOICE_STT_DEVICE}, compute={config.VOICE_STT_COMPUTE_TYPE})"
        )
        _model = WhisperModel(
            config.VOICE_STT_MODEL,
            device=config.VOICE_STT_DEVICE,
            compute_type=config.VOICE_STT_COMPUTE_TYPE,
        )
        log.info("Whisper model loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load Whisper model: {e}")
        raise

    return _model


def _moonshine_model_id(lang: str) -> str:
    if lang == "ja":
        return config.VOICE_STT_MOONSHINE_JA
    return config.VOICE_STT_MOONSHINE_EN


def _get_moonshine(lang: str):
    """Lazy-load a fixed-language Moonshine model (cached per language)."""
    if lang in _moonshine_cache:
        return _moonshine_cache[lang]

    import torch
    from transformers import AutoConfig, AutoProcessor

    model_id = _moonshine_model_id(lang)
    device = config.VOICE_STT_DEVICE if config.VOICE_STT_DEVICE in ("cuda", "cpu") else "cpu"
    # Moonshine runs fp16 on cuda; compute_type only configures whisper.
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Streaming checkpoints use a different architecture class; pick by the
    # model config's declared architectures so both load through one path.
    cfg = AutoConfig.from_pretrained(model_id)
    arch = getattr(cfg, "architectures", [None])[0] or ""
    if "Streaming" in arch:
        from transformers import MoonshineStreamingForConditionalGeneration as ModelCls
    else:
        from transformers import MoonshineForConditionalGeneration as ModelCls

    log.info(f"Loading Moonshine model: {model_id} (lang={lang}, device={device}, dtype={dtype})")
    model = ModelCls.from_pretrained(model_id)
    model = model.to(device).to(dtype).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    _moonshine_cache[lang] = (model, processor)
    log.info(f"Moonshine model loaded ({model_id}).")
    return _moonshine_cache[lang]


def _resample(audio: np.ndarray, sample_rate: int, target_sr: int = 16000) -> np.ndarray:
    if sample_rate == target_sr:
        return audio
    import math
    import scipy.signal
    g = math.gcd(int(sample_rate), int(target_sr))
    return scipy.signal.resample_poly(
        audio, target_sr // g, sample_rate // g
    ).astype(np.float32)


def _is_repetitive(text: str) -> bool:
    """Detect a decode loop: the same word repeated back-to-back."""
    words = [w.strip(".,!?;:'\"").lower() for w in text.split() if w.strip(".,!?;:'\"")]
    run = 1
    for i in range(1, len(words)):
        if words[i] and words[i] == words[i - 1]:
            run += 1
            if run >= _REPEAT_LIMIT:
                return True
        else:
            run = 1
    return False


def _transcribe_moonshine(
    audio: np.ndarray,
    sample_rate: int,
    lang: str,
) -> dict:
    import torch

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio.astype(np.float32))
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    audio = _resample(audio, sample_rate)

    dur = len(audio) / 16000.0
    if not audio.size or dur < 0.1:
        return {"text": "", "language": lang, "language_probability": 1.0, "segments": []}

    model, processor = _get_moonshine(lang)
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    input_values = inputs.input_values.to(device=model.device, dtype=model.dtype)

    # Hallucination-loop cap from the model card: ~6.5 output tokens per
    # audio second (+ small margin so dense speech isn't cut off).
    max_length = max(8, int(dur * 6.5) + 8)

    gen_kwargs = {}
    if "attention_mask" in inputs:
        gen_kwargs["attention_mask"] = inputs.attention_mask.to(device=model.device)

    t0 = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(input_values, max_length=max_length, do_sample=False, **gen_kwargs)
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    elapsed = (time.monotonic() - t0) * 1000

    if text and _is_repetitive(text):
        log.debug(f"Discarded repetitive Moonshine output: {text[:80]!r}")
        text = ""

    log.debug(
        f"Moonshine ({lang}) transcribed {dur:.1f}s audio in {elapsed:.0f}ms "
        f"-> {text[:80]!r}"
    )

    return {
        "text": text,
        "language": lang,
        "language_probability": 1.0,
        "segments": [{"start": 0.0, "end": dur, "text": text}] if text else [],
    }


def transcribe(
    audio: np.ndarray,
    sample_rate: int = 16000,
    language: Optional[str] = None,
) -> dict:
    """
    Transcribe audio numpy array to text.

    Args:
        audio: Float32 audio samples (1D array).
        sample_rate: Sample rate of the audio (default 16000).
        language: Optional language override ("en"/"ja"/"auto").
            None = use config.VOICE_STT_LANGUAGE.

    Returns:
        dict with keys:
            - text: Full transcribed text.
            - language: Language code used ("en"/"ja"/detected).
            - language_probability: 1.0 for fixed-language engines.
            - segments: List of segment dicts with start, end, text.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    lang = _resolved_language(language)
    engine = _engine()

    if engine == "moonshine" and lang:
        try:
            return _transcribe_moonshine(audio, sample_rate, lang)
        except Exception as e:
            log.warning(f"Moonshine STT failed, falling back to whisper: {e}")

    return _transcribe_whisper(audio, sample_rate, language)


def _transcribe_whisper(
    audio: np.ndarray,
    sample_rate: int,
    language: Optional[str] = None,
) -> dict:
    """faster-whisper transcription (auto-detects when language is 'auto')."""
    model = _whisper_model()

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    lang = _resolved_language(language)

    # Our pipeline already gates on Silero VAD, so Whisper's VAD filter
    # is redundant and can wipe short utterances entirely.
    audio_dur = len(audio) / sample_rate if sample_rate else 0.0
    use_vad_filter = audio_dur >= 2.0

    # Hotwords: bias decoding toward the character name so "Alice" isn't
    # misheard as e.g. "Ellison".
    #
    # NOTE: a full-sentence initial_prompt was removed. Whisper re-emits
    # that text verbatim when audio is weak/overlapping (exactly the busy-
    # voice-channel case), which is what produced "Alice, how are you doing?
    # Alice, how are you doing?" hallucinations. Short word-level hints via
    # hotwords do not carry that echo risk.
    #
    # Only ASCII mention keywords become hotwords. Multi-language aliases
    # (e.g. アリス) must not bias the STT: when Whisper detects the name, a
    # non-ASCII hotword makes it auto-insert that token into the
    # transcript (e.g. "Alice アリス Alice"). The mention keyword list
    # itself is untouched — response/conversation logic uses it as before.
    mention_kws = [
        kw.strip() for kw in getattr(config, "VOICE_MENTION_KEYWORDS", [])
        if kw and kw.strip()
    ]
    # De-dupe case-insensitively so ["alice", "Alice"] collapses to one
    # token; Whisper's hotwords are case-insensitive anyway.
    ascii_kws = [kw for kw in mention_kws if all(ord(c) < 128 for c in kw)]
    seen = set()
    hotword_parts = []
    for kw in ascii_kws + ["Alice", "alice"]:
        low = kw.lower()
        if low not in seen:
            seen.add(low)
            hotword_parts.append(kw)
    hotwords = " ".join(hotword_parts)

    segments_raw, info = model.transcribe(
        audio,
        language=lang,
        beam_size=_WHISPER_BEAM,
        hotwords=hotwords,
        condition_on_previous_text=False,
        vad_filter=use_vad_filter,
        vad_parameters=dict(
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
            speech_pad_ms=100,
        ) if use_vad_filter else None,
    )

    segments = []
    full_text_parts = []
    for seg in segments_raw:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })
        full_text_parts.append(seg.text.strip())

    result = {
        "text": " ".join(full_text_parts),
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": segments,
    }

    log.debug(
        f"Transcribed {len(audio)/sample_rate:.1f}s audio -> "
        f"'{result['text'][:80]}...' (lang={result['language']}, "
        f"prob={result['language_probability']:.2f})"
    )

    return result


def transcribe_file(
    file_path: str | Path,
    language: Optional[str] = None,
) -> dict:
    """Transcribe an audio file from disk."""
    lang = _resolved_language(language)

    if _engine() == "moonshine" and lang:
        try:
            import scipy.io.wavfile as wavfile
            sr, data = wavfile.read(str(file_path))
            audio = data.astype(np.float32)
            if data.dtype == np.int16:
                audio = audio / 32768.0
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return _transcribe_moonshine(audio, sr, lang)
        except Exception as e:
            log.warning(f"Moonshine file STT failed, falling back to whisper: {e}")

    model = _whisper_model()

    segments_raw, info = model.transcribe(
        str(file_path),
        language=lang,
        beam_size=3,
        hotwords="Alice alice",
        condition_on_previous_text=False,
        vad_filter=True,
    )

    segments = []
    full_text_parts = []
    for seg in segments_raw:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })
        full_text_parts.append(seg.text.strip())

    return {
        "text": " ".join(full_text_parts),
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": segments,
    }


def is_loaded() -> bool:
    """Check if the active engine's model is loaded."""
    if _engine() == "moonshine":
        lang = _resolved_language(None)
        return bool(lang) and lang in _moonshine_cache
    return _model is not None


def preload() -> None:
    """Preload the active engine's model at startup (blocking; call in a thread)."""
    try:
        if _engine() == "moonshine":
            lang = _resolved_language(None)
            if lang:
                _get_moonshine(lang)
                return
        _whisper_model()
    except Exception as e:
        log.error(f"STT preload failed: {e}")


def unload() -> None:
    """Unload all STT models to free GPU memory."""
    global _model
    if _model is not None:
        _model = None
        log.info("Whisper model unloaded.")
    if _moonshine_cache:
        _moonshine_cache.clear()
        log.info("Moonshine models unloaded.")
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
