"""
voice/rvc.py
=============
RVC voice conversion. Converts TTS audio to sound like Alice using
the trained RVC model via D:\\RVC offline inference (infer.vc.modules.VC).
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from bot.logging_setup import get_logger
import config

log = get_logger("voice.rvc")

_RVC_ROOT = Path(r"D:\RVC")
_model = None  # infer.vc.modules.VC instance once loaded
_load_lock = threading.Lock()
_convert_lock = threading.Lock()
_load_failed = False
_env_ready = False


class _rcwd:
    """Temporarily chdir to D:\\RVC (i18n loads ./i18n/locale/...)."""

    def __enter__(self):
        self._prev = os.getcwd()
        os.chdir(_RVC_ROOT)
        return self

    def __exit__(self, *exc):
        os.chdir(self._prev)
        return False


def _resolve_path(raw: str) -> Optional[Path]:
    """Resolve a possibly-relative path against the bot project root."""
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = config.BASE_DIR / p
    return p.resolve()


def _model_path() -> Optional[Path]:
    return _resolve_path(config.VOICE_RVC_MODEL_PATH)


def _index_path() -> Optional[Path]:
    return _resolve_path(config.VOICE_RVC_INDEX_PATH)


def _setup_env() -> None:
    """Put D:\\RVC on sys.path and set path env vars RVC expects."""
    global _env_ready
    if _env_ready:
        return
    if not _RVC_ROOT.is_dir():
        raise FileNotFoundError(f"RVC repo not found: {_RVC_ROOT}")
    root = str(_RVC_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ["weight_root"] = str(_model_path().parent)
    os.environ["index_root"] = str(_RVC_ROOT / "logs")
    os.environ["outside_index_root"] = str(_RVC_ROOT / "assets" / "indices")
    os.environ["rmvpe_root"] = str(_RVC_ROOT / "assets" / "rmvpe")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # RVC load_audio shells out to ffmpeg/ffprobe; ensure they are findable.
    ffmpeg_bin = Path(r"C:\ffmpeg\ffmpeg-9.0.2-essentials_build\bin")
    if ffmpeg_bin.is_dir():
        path = os.environ.get("PATH", "")
        if str(ffmpeg_bin) not in path.split(os.pathsep):
            os.environ["PATH"] = str(ffmpeg_bin) + os.pathsep + path
    _env_ready = True


def _create_config():
    """Build D:\\RVC Config without letting it parse the bot's argv."""
    old_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        from configs.config import Config
        return Config()
    finally:
        sys.argv = old_argv


def _get_model():
    """Lazy-load the RVC VC engine. Thread-safe. Returns VC or None."""
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None

    with _load_lock:
        if _model is not None:
            return _model
        if not config.VOICE_RVC_ENABLED:
            log.debug("RVC disabled in config")
            return None

        mp = _model_path()
        if mp is None or not mp.exists():
            log.warning(f"RVC model not found: {config.VOICE_RVC_MODEL_PATH}")
            _load_failed = True
            return None

        try:
            t0 = __import__("time").perf_counter()
            _setup_env()
            log.info(f"Loading RVC model: {mp.name}")

            with _rcwd():
                cfg = _create_config()
                if config.VOICE_RVC_DEVICE.lower() == "cpu":
                    cfg.device = "cpu"
                    cfg.is_half = False

                from infer.vc.modules import VC

                vc = VC(cfg)
                # get_vc loads net_g from weight_root/<filename>
                vc.get_vc(mp.name)

            if vc.net_g is None or vc.pipeline is None:
                raise RuntimeError("VC.get_vc did not load model weights")

            _model = vc
            dt = __import__("time").perf_counter() - t0
            log.info(
                f"RVC model loaded successfully on {cfg.device} "
                f"(is_half={cfg.is_half}) in {dt:.1f}s"
            )
        except Exception as e:
            log.error(f"Failed to load RVC model: {e}", exc_info=True)
            _model = None
            _load_failed = True
            return None

        return _model


def preload() -> None:
    """Preload RVC engine + HuBERT + RMVPE (blocking; call in a thread)."""
    try:
        model = _get_model()
        if model is None:
            return

        with _rcwd():
            # HuBERT: content encoder used on every convert
            if getattr(model, "hubert_model", None) is None:
                from infer.vc.utils import load_hubert

                t0 = __import__("time").perf_counter()
                model.hubert_model = load_hubert(model.config)
                log.info(
                    f"RVC HuBERT preloaded in "
                    f"{__import__('time').perf_counter() - t0:.1f}s"
                )

            # RMVPE: f0 estimator (loaded lazily inside pipeline otherwise)
            pipe = getattr(model, "pipeline", None)
            if pipe is not None and not hasattr(pipe, "model_rmvpe"):
                from infer.rmvpe import RMVPE

                t0 = __import__("time").perf_counter()
                pipe.model_rmvpe = RMVPE(
                    "%s/rmvpe.pt" % os.environ["rmvpe_root"],
                    is_half=pipe.is_half,
                    device=pipe.device,
                )
                log.info(
                    f"RVC RMVPE preloaded in "
                    f"{__import__('time').perf_counter() - t0:.1f}s"
                )

            # Dummy convert: absorb first-call CUDA/cuDNN warmup (~4s)
            # so the first live sentence doesn't pay it.
            t0 = __import__("time").perf_counter()
            sr = 16000
            t = np.arange(sr, dtype=np.float32) / sr
            warm = 0.2 * np.sin(2 * np.pi * 220 * t)
            convert(warm, sr)
            log.info(
                f"RVC warm convert done in "
                f"{__import__('time').perf_counter() - t0:.1f}s"
            )
    except Exception as e:
        log.error(f"RVC preload failed: {e}", exc_info=True)


def convert(
    audio: np.ndarray,
    sample_rate: int = 16000,
    pitch_shift: Optional[int] = None,
) -> np.ndarray:
    """
    Convert audio to Alice's voice using RVC.

    Args:
        audio: Float32 audio samples.
        sample_rate: Sample rate of input audio.
        pitch_shift: Semitone shift (default: config.VOICE_RVC_PITCH).

    Returns:
        Converted float32 audio samples (same length/rate as input).
    """
    model = _get_model()
    if model is None:
        log.debug("RVC not available, returning original audio")
        return audio

    pitch = pitch_shift if pitch_shift is not None else config.VOICE_RVC_PITCH
    index_rate = config.VOICE_RVC_INDEX_RATE

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak * 0.95

    # RVC offline path expects a 16 kHz mono WAV on disk
    if sample_rate != 16000:
        import scipy.signal
        audio_16k = scipy.signal.resample(
            audio, int(len(audio) * 16000 / sample_rate)
        ).astype(np.float32)
    else:
        audio_16k = audio

    import soundfile as sf

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="alice_rvc_")
        os.close(fd)
        sf.write(tmp_path, audio_16k, 16000, subtype="PCM_16")

        idx = _index_path()
        idx_str = str(idx) if idx is not None and idx.exists() and index_rate > 0 else ""
        rate = index_rate if idx_str else 0.0

        # VC is not re-entrant; also chdir so any relative RVC paths resolve.
        with _convert_lock, _rcwd():
            status, result = model.vc_single(
                0,                 # speaker id
                tmp_path,
                int(pitch),
                "rmvpe",
                idx_str,
                float(rate),
                0,                 # resample_sr: keep model tgt_sr
                1.0,               # rms_mix_rate
                0.33,              # protect
            )

        if not result or result[0] is None or result[1] is None:
            log.error(f"RVC conversion failed: {status}")
            return audio

        out_sr, output = result
        output = np.asarray(output, dtype=np.float32).reshape(-1)

        # Resample model output (typically 40 kHz) back to caller rate
        if out_sr != sample_rate and len(output) > 0:
            import scipy.signal
            output = scipy.signal.resample(
                output, int(len(output) * sample_rate / out_sr)
            ).astype(np.float32)

        # Keep levels Discord-safe (RVC often overshoots past full scale)
        out_peak = float(np.max(np.abs(output))) if len(output) else 0.0
        if out_peak > 0.95:
            output = output * (0.95 / out_peak)

        log.debug(
            f"RVC conversion: {len(audio_16k)/16000:.1f}s -> "
            f"{len(output)/max(sample_rate,1):.1f}s @ {sample_rate}Hz "
            f"(model {out_sr}Hz, peak {out_peak:.2f})"
        )
        return output

    except Exception as e:
        log.error(f"RVC conversion failed: {e}", exc_info=True)
        return audio
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def soften_human(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Post-process RVC output: noise gate, lowpass, soft clip.
    (Aspiration-noise injection removed — it hissed in silent gaps.)
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    # 1. Noise gate: silence very quiet parts
    threshold = 0.01
    gate = np.where(np.abs(audio) > threshold, 1.0, 0.0)
    gate = np.convolve(gate, np.ones(512) / 512, mode="same")
    audio = audio * gate

    # 2. Lowpass filter at 9kHz to remove RVC HF artifacts
    try:
        import scipy.signal
        nyq = sample_rate / 2
        if nyq > 9000:
            b, a = scipy.signal.butter(6, 9000 / nyq, btype="low")
            audio = scipy.signal.lfilter(b, a, audio)
    except ImportError:
        pass

    # 3. Soft clip (tanh saturation)
    audio = np.tanh(audio * 1.2) / np.tanh(1.2)

    return audio


def is_available() -> bool:
    """
    True if RVC should run for TTS.

    Deliberately does NOT require the model to already be in memory —
    that deadlocked convert() forever. Availability = enabled + model file
    present (+ not permanently failed). convert() lazy-loads on demand.
    """
    if not config.VOICE_RVC_ENABLED:
        return False
    if _load_failed:
        return False
    if _model is not None:
        return True
    mp = _model_path()
    return mp is not None and mp.exists()


def is_loaded() -> bool:
    """True only when the VC engine is fully in memory."""
    return _model is not None


def unload() -> None:
    """Unload the RVC model."""
    global _model, _load_failed, _env_ready
    with _load_lock:
        _model = None
        _load_failed = False
        _env_ready = False
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    log.info("RVC model unloaded.")
