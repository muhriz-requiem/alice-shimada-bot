"""
desktop/dsp.py
==============
WebRTC Audio Processing Module (APM) for the desktop companion mic path.

The Discord client processes every mic stream with WebRTC's audio DSP
before the bot ever sees a sample: acoustic echo cancellation, noise
suppression, gain control. The desktop companion historically captured
RAW PortAudio audio -- every hardware weakness landed directly in the
VAD/STT path (speaker echo re-triggering the mic, faint levels, noise).

This module restores parity using the `aec-audio-processing` package
(python-webrtc APM bindings):

- AEC:  Alice's own TTS playback is fed as the reverse (far-end) stream
        from SdOutput's device callback, so her voice coming out of the
        speakers is subtracted from the mic instead of being heard as
        the user (echo) or barging into herself.
- NS:   stationary noise (fans, hum) suppressed before Silero VAD and
        Whisper see the samples.

The processed stream is 16 kHz mono (APM contract); blocks are handed
back upsampled to the 48 kHz stereo s16le shape MicInput/sink expect
(nearest-hold upsampling -- VAD's [::3] decimation and the STT resample
both recover the identical 16 kHz content).

Failure contract: from_config() returns None when the package is
missing or the module cannot initialize -- callers then run the legacy
raw path (echo gate + AGC only). Nothing here raises into the hot path.
"""

from __future__ import annotations

import contextlib
import queue
import threading

import numpy as np

from bot.logging_setup import get_logger

log = get_logger("desktop.dsp")

try:
    from aec_audio_processing import AudioProcessor

    _APM_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - depends on host packages
    AudioProcessor = None
    _APM_AVAILABLE = False
    _IMPORT_ERROR = _exc

_MIC_RATE = 48000
_APM_RATE = 16000
# 10 ms of 16 kHz mono s16le -- APM's only accepted frame size.
_APM_FRAME = 320  # bytes (160 samples * 2)


def available() -> bool:
    """True when the APM bindings import on this machine."""
    return _APM_AVAILABLE


class MicApm:
    """Echo-cancel + noise-suppress one desktop mic stream.

    near:  raw mic blocks (48 kHz stereo s16le bytes, 20 ms) -> processed
           blocks in the same shape.
    far:   the device output stream (what actually reaches the speakers)
           fed as the reverse stream, so the AEC can subtract Alice.

    Both PortAudio callback threads call in concurrently -- every APM
    entry point is serialized by a lock (the C module is not
    thread-safe), and each call is microseconds of C++, so neither
    callback risks an underrun.
    """

    def __init__(self, delay_ms: int = 60) -> None:
        if not _APM_AVAILABLE:
            raise RuntimeError("aec-audio-processing not importable")
        self._lock = threading.Lock()
        self._apm = AudioProcessor(
            enable_aec=True,
            enable_ns=True,
            # Deterministic gain is MicInput's AGC job (config-driven);
            # APM's statistical AGC on top would fight it.
            enable_agc=False,
            enable_vad=False,
        )
        self._apm.set_stream_format(_APM_RATE, 1, _APM_RATE, 1)
        self._apm.set_reverse_stream_format(_APM_RATE, 1)
        self._delay_ms = int(delay_ms)
        # Far-end accumulation: device callbacks arrive in arbitrary
        # block sizes; APM consumes strict 10 ms frames.
        self._far_buf = np.zeros(0, dtype=np.int16)
        self._near_frames = 0
        # Far-end work must NEVER run on the output device callback
        # thread: the APM lock is contended by the mic feeder (~50x/s),
        # and blocking the callback misses its deadline -> device
        # underruns (crackling output). Queue instead; a feeder thread
        # drains it on its own time.
        self._far_q: queue.SimpleQueue = queue.SimpleQueue()
        threading.Thread(
            target=self._far_loop, daemon=True, name="apm-far-feeder"
        ).start()
        log.info("Mic APM initialized (AEC+NS, delay=%dms)", self._delay_ms)

    def _far_loop(self) -> None:
        """Drain queued played-audio blocks into the APM (own thread)."""
        while True:
            block = self._far_q.get()
            if block is None:
                return
            with contextlib.suppress(Exception):  # far feed is best-effort
                self._feed_far(block)

    def process_far(self, mono_f32: np.ndarray) -> None:
        """Queue played-out audio (mono float32 48 kHz) as the far end.

        Called from the OUTPUT device callback: enqueue only, never
        process here.
        """
        if mono_f32 is None or len(mono_f32) == 0:
            return
        # Pure-silence blocks (pre-roll hold, gaps) carry no reference
        # value -- skip the enqueue churn.
        if not np.any(mono_f32):
            return
        self._far_q.put(np.asarray(mono_f32, dtype=np.float32))

    def _feed_far(self, mono_f32: np.ndarray) -> None:
        pcm16 = np.clip(
            np.asarray(mono_f32, dtype=np.float32) * 32768.0,
            -32768.0, 32767.0,
        ).astype(np.int16)
        with self._lock:
            self._far_buf = np.concatenate(
                (self._far_buf, self._decimate48(pcm16))
            )
            while len(self._far_buf) >= _APM_FRAME // 2:
                frame = self._far_buf[: _APM_FRAME // 2].tobytes()
                self._far_buf = self._far_buf[_APM_FRAME // 2:]
                with contextlib.suppress(Exception):  # far feed is best-effort
                    self._apm.process_reverse_stream(frame)

    # --- near end (the mic) --------------------------------------------------

    def process_near(self, pcm_bytes: bytes) -> bytes:
        """Process one mic block (48 kHz stereo s16le) -> same shape, cleaned."""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        mono16 = self._decimate(samples)
        out = np.zeros(len(mono16), dtype=np.int16)
        with self._lock:
            for i in range(0, len(mono16) - 159, 160):
                try:
                    processed = self._apm.process_stream(
                        mono16[i:i + 160].tobytes()
                    )
                    self._apm.set_stream_delay(self._delay_ms)
                    out[i:i + 160] = np.frombuffer(processed, dtype=np.int16)
                except Exception:  # noqa: BLE001
                    out[i:i + 160] = mono16[i:i + 160]
        self._near_frames += 1
        # 16 kHz mono -> 48 kHz stereo s16le (nearest hold; see module doc).
        up = np.repeat(out, 3)
        stereo = np.empty(len(up) * 2, dtype=np.int16)
        stereo[0::2] = up
        stereo[1::2] = up
        return stereo.tobytes()

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _decimate48(mono16: np.ndarray) -> np.ndarray:
        """48 kHz mono int16 -> 16 kHz mono int16 (3-sample mean)."""
        f32 = mono16.astype(np.float32)
        n3 = (len(f32) // 3) * 3
        if n3 == 0:
            return np.zeros(0, dtype=np.int16)
        return f32[:n3].reshape(-1, 3).mean(axis=1).astype(np.int16)

    @classmethod
    def _decimate(cls, samples: np.ndarray) -> np.ndarray:
        """48 kHz (stereo-interleaved or mono) int16 -> 16 kHz mono int16."""
        if samples.ndim == 1 and len(samples) % 2 == 0:
            # Stereo-interleaved input: fold to mono first.
            samples = samples.reshape(-1, 2).mean(axis=1)
        return cls._decimate48(samples.astype(np.int16))


def from_config(delay_ms: int = 60) -> MicApm | None:
    """MicApm or None (package missing / init failed) -- never raises."""
    if not _APM_AVAILABLE:
        log.info(
            "Mic APM unavailable (aec-audio-processing not installed: %s); "
            "raw mic path stays (echo gate + AGC only)",
            _IMPORT_ERROR,
        )
        return None
    try:
        return MicApm(delay_ms=delay_ms)
    except Exception as exc:  # noqa: BLE001
        log.warning("Mic APM init failed (%s); raw mic path stays", exc)
        return None
