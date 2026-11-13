"""desktop/prewarm.py
Warm the voice stack at app launch instead of at the first utterance.

Discord gets all of this in voice/pipeline.init_pipeline() on on_ready;
the desktop companion historically skipped every preload, so the first
spoken turn paid for:
- Silero VAD model load -- silently degraded to the RMS fallback
  (the VAD hot path never lazy-loads by design),
- the full faster-whisper CUDA/model load (~15-35s),
- a cold GPT-SoVITS server boot (up to the whole
  VOICE_SOVITS_STARTUP_TIMEOUT budget, blocking INSIDE the first reply),
- the phoneme worker cold start (handled at tray start by
  LipSyncSubscriber instead).

Order matters, learned the hard way (2026-10-25 overnight stall):
whisper warm-up running while the GPT-SoVITS server was cold-booting
starved it into a full-boot timeout. The sequence below is strictly:

    1. VAD preload          (CPU; also warms the torch import/DLLs)
    2. GPT-SoVITS bring-up  (ensure_server + probe; only when the
                             sovits engine is selected)
    3. STT preload          (GPU; starts only AFTER the server is up)

Disable with DESKTOP_PREWARM=0 (edge-engine users lose nothing but
warmer STT).
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Optional

from bot.logging_setup import get_logger

log = get_logger("desktop.prewarm")


def _enabled() -> bool:
    return os.getenv(
        "DESKTOP_PREWARM", "1"
    ).strip().lower() not in ("0", "false", "no", "off")


_started = False
_lock = threading.Lock()


def start_prewarm() -> bool:
    """Kick the prewarm sequence in a daemon thread. Return False if
    already started, or disabled by env (`start` is idempotent)."""
    global _started
    with _lock:
        if _started:
            return False
        _started = True
    if not _enabled():
        log.info("Desktop prewarm disabled (DESKTOP_PREWARM)")
        return False
    threading.Thread(
        target=_run, daemon=True, name="desktop-prewarm"
    ).start()
    return True


def _run() -> None:
    from desktop.boot_timeline import boot_timeline
    import config

    timeline = boot_timeline
    timeline.mark("prewarm_start")

    # --- 1. VAD (CPU; warms torch import so later GPU work pays less) ----
    _preload_vad(timeline)

    # --- 2. GPT-SoVITS server --------------------------------------------
    sovits_ok: Optional[bool] = None
    if config.VOICE_TTS_ENGINE == "sovits":
        sovits_ok = _boot_sovits(timeline)
    else:
        log.info(
            "SoVITS prewarm skipped (VOICE_TTS_ENGINE=%s)",
            config.VOICE_TTS_ENGINE,
        )

    # --- 3. STT (GPU; only after the TTS server has settled) --------------
    # Hosted STT decodes nothing locally: skip the local model load so the
    # GPU keeps the headroom SoVITS needs. The fallback engine lazy-loads
    # on demand if the hosted call ever fails.
    if config.VOICE_STT_HOSTED:
        timeline.mark("stt_ready")
        log.info("STT preload skipped (VOICE_STT_HOSTED=true; local engine stays lazy)")
    elif sovits_ok is not False:  # None = sovits not selected; skip-gate only
        _preload_stt(timeline)


def _preload_vad(timeline) -> None:
    try:
        from voice import vad
        if not vad.is_model_loaded():
            t0 = time.perf_counter()
            vad.preload()
            timeline.mark("vad_ready")
            log.info(
                "VAD preloaded in %.1fs", time.perf_counter() - t0
            )
        else:
            timeline.mark("vad_ready")
    except Exception as exc:  # noqa: BLE001
        log.warning("VAD preload failed (RMS fallback stays): %s", exc)


def _check_lazy_bert_patch() -> None:
    """Warn when the external GPT-SoVITS lost the lazy-BERT VRAM patch.

    A GPT-SoVITS update overwrites TTS.py and the ~1.35 GB Chinese BERT
    starts loading eagerly again (4 GB cards: synthesis headroom and the
    avatar engine starve). Re-run desktop/sovits_patch/apply.py.
    """
    try:
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).resolve().parent / "sovits_patch" / "apply.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            log.info("GPT-SoVITS lazy-BERT patch present")
        else:
            log.warning(
                "GPT-SoVITS lazy-BERT patch MISSING (%s) -- ~1.35 GB VRAM "
                "will be wasted on a BERT Alice's en/ja speech never uses. "
                "Run: venv\\Scripts\\python.exe desktop\\sovits_patch\\apply.py",
                (proc.stdout or "").strip().splitlines()[:1],
            )
    except Exception as exc:  # noqa: BLE001 - advisory only
        log.debug("lazy-bert patch check failed: %s", exc)


def _boot_sovits(timeline) -> bool:
    """ensure_server + probe in this thread's own asyncio loop.

    Desktop semantics stay unchanged: build_pipeline() forces TTS
    readiness either way, so a failed prewarm simply degenerates to the
    old on-demand behavior (edge fallback / first-turn server boot) --
    nothing here can lose Alice's voice.
    """
    _check_lazy_bert_patch()
    try:
        from voice import readiness, sovits
        readiness._state = readiness.STATE_READY

        async def _bring_up() -> bool:
            if not await sovits.ensure_server():
                return False
            return await sovits.probe()

        ok = asyncio.run(_bring_up())
        timeline.mark("sovits_up" if ok else "sovits_failed")
        log.info("SoVITS prewarm: %s", "ready" if ok else "unavailable")
        return ok
    except Exception as exc:  # noqa: BLE001
        log.warning("SoVITS prewarm failed: %s", exc)
        timeline.mark("sovits_failed")
        return False


def _preload_stt(timeline) -> None:
    try:
        from voice import stt
        if not stt.is_loaded():
            t0 = time.perf_counter()
            stt.preload()
            timeline.mark("stt_ready")
            log.info(
                "STT preloaded in %.1fs", time.perf_counter() - t0
            )
        else:
            timeline.mark("stt_ready")
    except Exception as exc:  # noqa: BLE001
        log.warning("STT preload failed: %s", exc)
