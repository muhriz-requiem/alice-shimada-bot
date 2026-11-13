"""Desktop subscriber: drives MateEngine mouth from voice lip-sync events.

Subscribes to voice.lip_sync (neutral publisher) and, for each synthesized
sentence, computes a weighted viseme timeline and plays it via
MateBridge.speak_timeline() on a background worker thread.

Kept out of voice/ so the voice pipeline has no desktop dependency.
"""
from __future__ import annotations

import contextlib
import io
import queue
import threading
import time
import wave
from typing import Any

import numpy as np

from bot.logging_setup import get_logger

log = get_logger("desktop.lip_sync_subscriber")

try:
    import config
except Exception:  # noqa: BLE001
    config = None


def _is_japanese(text: str) -> bool:
    return any(
        "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in text
    )


def _decode_wav(wav_bytes: bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    mono = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        mono = mono.reshape(-1, ch).mean(axis=1)
    return mono, sr


class LipSyncSubscriber:
    def __init__(self):
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._next_start = 0.0
        self._bridge = None
        self._worker = None  # PhonemeWorker (English)
        self._engine_manager = None

    def set_engine_manager(self, em) -> None:
        self._engine_manager = em

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Register with the neutral publisher
        try:
            from voice import lip_sync
            lip_sync.subscribe(self._on_event)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not subscribe to lip_sync: %s", exc)
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="lip-sync")
        self._thread.start()
        log.info("lip-sync subscriber started")

    def stop(self) -> None:
        self._running = False
        try:
            from voice import lip_sync
            lip_sync.unsubscribe(self._on_event)
        except Exception:  # noqa: BLE001
            pass
        # sentinel to unblock the worker
        with contextlib.suppress(Exception):
            self._queue.put_nowait(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _on_event(self, event: dict[str, Any]) -> None:
        """Fast: just enqueue. Never block the voice pipeline."""
        if not getattr(config, "VOICE_LIP_SYNC", True) if config else True:
            return
        with contextlib.suppress(Exception):  # never block the pipeline
            self._queue.put_nowait(event)

    def _engine_ready(self) -> bool:
        em = self._engine_manager
        if em is None:
            return True  # unknown -> try anyway
        try:
            return em.status == "ready"
        except Exception:  # noqa: BLE001
            return True

    def _worker_loop(self) -> None:
        import os
        import sys
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if here not in sys.path:
            sys.path.insert(0, here)
        # Lazy imports (desktop-side modules)
        try:
            from desktop import mouth
        except Exception:  # noqa: BLE001
            try:
                import mouth
            except Exception as exc:  # noqa: BLE001
                log.warning("mouth module unavailable: %s", exc)
                mouth = None
        try:
            from desktop.mate_bridge import MateBridge
        except Exception:  # noqa: BLE001
            try:
                import os
                import sys
                here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if here not in sys.path:
                    sys.path.insert(0, here)
                from desktop.mate_bridge import MateBridge
            except Exception as exc:  # noqa: BLE001
                log.warning("MateBridge unavailable: %s", exc)
                MateBridge = None

        # English phoneme worker (persistent, cold ~7 s)
        pw = None
        try:
            from desktop.phonemes import PhonemeWorker
            pw = PhonemeWorker()
            pw.phonemize("test", lang="en")  # warm up
        except Exception as exc:  # noqa: BLE001
            log.warning("PhonemeWorker unavailable: %s", exc)
            pw = None

        while self._running:
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if event is None:
                break
            if not self._running:
                break
            try:
                self._handle(event, mouth, MateBridge, pw)
            except Exception as exc:  # noqa: BLE001
                log.warning("lip-sync handle failed: %s", exc)

    def _handle(self, event, mouth, MateBridge, pw) -> None:
        if mouth is None or MateBridge is None:
            return
        if not self._engine_ready():
            return
        text = (event.get("text") or "").strip()
        wav_bytes = event.get("wav_bytes")
        sr = event.get("sr") or 0
        if not text or not wav_bytes:
            return
        try:
            mono, audio_sr = _decode_wav(wav_bytes)
        except Exception as exc:  # noqa: BLE001
            log.debug("wav decode failed: %s", exc)
            return
        if audio_sr:
            sr = audio_sr
        if sr <= 0 or len(mono) == 0:
            return
        duration = len(mono) / sr
        if duration <= 0.05:
            return

        # Build timeline
        try:
            from desktop.mouth import find_pauses
            pauses = find_pauses(mono, sr)
        except Exception:  # noqa: BLE001
            try:
                from mouth import find_pauses
                pauses = find_pauses(mono, sr)
            except Exception:  # noqa: BLE001
                pauses = []

        events = None
        if _is_japanese(text):
            try:
                events = mouth.kana_timeline(text, duration, pauses)
            except Exception as exc:  # noqa: BLE001
                log.debug("kana_timeline failed: %s", exc)
                events = None
        else:
            phones = None
            if pw is not None:
                try:
                    phones = pw.phonemize(text, lang="en")
                except Exception as exc:  # noqa: BLE001
                    log.debug("phonemize failed: %s", exc)
                    phones = None
            if phones:
                try:
                    events = mouth.build_timeline(phones, duration, pauses)
                except Exception as exc:  # noqa: BLE001
                    log.debug("build_timeline failed: %s", exc)
                    events = None

        if not events:
            return

        # Schedule: don't overlap with previous sentence's animation
        now = time.perf_counter()
        if now < self._next_start:
            time.sleep(self._next_start - now)
        self._next_start = time.perf_counter() + duration

        # Play
        try:
            br = MateBridge()
            br.speak_timeline(events)
        except Exception as exc:  # noqa: BLE001
            log.debug("speak_timeline failed: %s", exc)
