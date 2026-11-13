"""
desktop/local_client.py
=======================
Duck-typed stand-in for discord.VoiceClient that plays Alice's TTS to
the local speakers instead of a Discord voice channel.

The whole playback stack (VoiceMixer, the streaming TTS path, banked
acks) only needs a small voice-client contract:

- vc.play(source, after=...)    start playing a discord.AudioSource
- vc.is_playing()               True while audio is being produced
- vc.stop()                     barge-in / abort playback

Every source the pipeline hands to play() (voice.streaming_source
.StreamingAudioSource and discord.FFmpegPCMAudio) yields 20ms frames of
16-bit 48kHz interleaved stereo PCM and b"" at end of stream. The local
player converts each frame to mono float32 and pushes it into a
sounddevice OutputStream, which paces the audio in real time.

after() fires exactly once, from the asyncio loop thread (like
discord.py does), once playback truly finished through the device -- so
VoiceMixer's completion waits and the pipeline's drain loops behave
identically to the Discord path.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from bot.logging_setup import get_logger

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - only hit on machines without audio I/O
    sd = None

log = get_logger("desktop.local_client")

_SAMPLE_RATE = 48000  # discord sources always emit 48kHz stereo
FRAME_SIZE = 960      # mono samples per 20ms frame (48000 * 0.02)
_READ_RATE_HZ = 50.0  # sources emit 20ms frames; consume at real time

# Sentinel telling the device "source data complete" (EOF marker).
_END = object()


def _client_exception(msg: str) -> Exception:
    from discord.errors import ClientException

    return ClientException(msg)


def _stereo_to_mono(frame: bytes) -> Optional[np.ndarray]:
    """Convert one 20ms interleaved s16le stereo frame to mono float32.

    Returns None for empty/odd-sized frames (treated as EOF by the
    player loop) -- sources may also hand back None at stream end.
    """
    try:
        samples = np.frombuffer(frame, dtype=np.int16)
    except (ValueError, TypeError):
        return None
    if samples.size < 2:
        return None
    mono = samples.reshape(-1, 2).mean(axis=1).astype(np.float32) / 32768.0
    return mono


class SdOutput:
    """A real sounddevice.OutputStream fed via a thread-safe queue.

    The PortAudio callback paces playback in real time and consumes
    frames at whatever block size the device asks for -- queued frames
    accumulate in an internal buffer, so nothing is dropped and no CPU
    loop sleeps are needed. The queue running dry plays silence (a
    short synthesis stall won't glitch the output).
    """

    def __init__(
        self,
        rms_callback: Optional[Callable[[float], None]] = None,
        device: Optional[int] = None,
        far_end_callback: Optional[Callable[[np.ndarray], None]] = None,
        preroll_ms: int = 250,
    ) -> None:
        self.playing = True
        self._q: "queue.SimpleQueue" = queue.SimpleQueue()
        self._buf = np.zeros(0, dtype=np.float32)
        self._end_seen = False
        self._drained = threading.Event()
        self._rms_callback = rms_callback
        # Echo-canceller reference: every sample the device actually
        # plays (including underrun silence) goes back to the APM as the
        # far end, so subtracted echo matches what the mic can hear.
        self._far_end_callback = far_end_callback
        # Pre-roll: hold playback start until this much audio (samples)
        # is buffered. Streaming TTS chunks arrive with GPU jitter; a
        # cushion absorbs it so the first sentence plays smoothly
        # instead of stuttering. Costs `preroll_ms` of added TTFA.
        self._preroll_samples = max(0, int(preroll_ms * _SAMPLE_RATE / 1000))
        self._drain_started = self._preroll_samples == 0
        self._stream = sd.OutputStream(
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            device=device,
        )
        self._stream.start()

    # --- PortAudio thread ------------------------------------------------
    def _audio_callback(self, outdata, frames, t, status) -> None:
        # While holding for the pre-roll cushion, pull ahead toward it
        # (otherwise the pull loop tops up only `frames` at a time and
        # the cushion would take forever to fill).
        target = frames if self._drain_started else max(
            frames, self._preroll_samples
        )
        buf = self._buf
        while len(buf) < target:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is _END:
                self._end_seen = True
                break
            mono = _stereo_to_mono(item)
            if mono is not None:
                buf = np.concatenate((buf, mono))

        n = min(frames, len(buf))
        if not self._drain_started:
            # Pre-roll: accumulate but stay silent until enough audio is
            # buffered -- streamed TTS chunks arrive with GPU jitter, and
            # stuttering at the very start is the most audible. Never
            # deadlocks a short reply: the stream's end releases the hold.
            if len(buf) < self._preroll_samples and not self._end_seen:
                outdata[:] = 0
                self._buf = buf
                return
            self._drain_started = True

        if n:
            outdata[:n, 0] = buf[:n]
            # Emit RMS for lip-sync (on the PortAudio thread, keep it fast)
            if self._rms_callback is not None:
                rms = float(np.sqrt(np.mean(buf[:n] ** 2)))
                # Schedule on the Qt main thread via postEvent or QTimer
                # We'll use a simple thread-safe queue approach for the callback
                try:
                    self._rms_callback(rms)
                except Exception:
                    pass
        if n < frames:
            outdata[n:, 0] = 0
        self._buf = buf[n:]

        if self._far_end_callback is not None:
            try:
                self._far_end_callback(outdata[:frames, 0])
            except Exception:  # noqa: BLE001 - AEC reference is best-effort
                pass

        if self._end_seen and len(self._buf) == 0:
            self._drained.set()

    # --- reader-thread API -------------------------------------------------
    def put(self, frame: bytes) -> None:
        """Queue one stereo s16le frame for playback."""
        self._q.put(frame)

    def finish(self) -> None:
        """Signal end-of-stream; playback drains and then completes."""
        self._q.put(_END)

    def wait_drained(self) -> None:
        """Block until every queued sample played through the device."""
        self._drained.wait()

    def abort(self) -> None:
        """Stop immediately (barge-in): discard queued audio."""
        self._drained.set()
        self._end_seen = True
        try:
            self._stream.abort()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        self.playing = False
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # noqa: BLE001
            pass


class SilentOutput:
    """Headless output device: consumes sources without a sound device.

    Same lifecycle contract as SdOutput (put/finish/wait_drained/
    abort/close, playing flag) so tests and --no-speaker runs exercise
    the identical reader-thread logic.
    """

    def __init__(self) -> None:
        self.playing = True
        self.frames_played = 0
        self._drained = threading.Event()
        self._aborted = False

    def put(self, frame: bytes) -> None:
        # 2ch * 2 bytes per sample per frame.
        self.frames_played += len(frame) // 4

    def finish(self) -> None:
        # Reading is instantaneous: EOF == fully drained.
        self._drained.set()

    def wait_drained(self) -> None:
        self._drained.wait()

    def abort(self) -> None:
        self._aborted = True
        self.playing = False
        self._drained.set()

    def close(self) -> None:
        self.playing = False


class LocalVoiceClient:
    """Duck-typed discord.VoiceClient for local playback.

    channel: object exposing .members (the pipeline parses it via
    _others_in_channel). Pass a channel listing only the owner -- that
    yields private-call behavior (the snappier live-mode silence
    threshold).
    """

    def __init__(
        self,
        guild_id: int = 0,
        channel=None,
        output_factory: Optional[Callable[[], object]] = None,
        rms_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.guild_id = guild_id
        self.channel = channel
        self._factory = output_factory or (lambda: SdOutput(rms_callback=rms_callback))
        self._out = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # --- discord.VoiceClient surface -----------------------------------

    def is_connected(self) -> bool:
        # Method, not property: voice/state.py calls vc.is_connected().
        return True

    def is_playing(self) -> bool:
        return self._out is not None and self._out.playing

    def play(self, source, after: Optional[Callable] = None) -> None:
        """Play a discord.AudioSource-style object to the speakers.

        Mirrors VoiceClient.play: raises ClientException if already
        playing (the pipeline relies on that clamp too).
        """
        with self._lock:
            if self.is_playing():
                raise _client_exception("Already playing audio.")
            out = self._factory()
            self._out = out

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - play() is async-driven
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        outter = out
        error_holder: list = [None]

        def _player() -> None:
            """Reader thread: pull source frames into the output device.

            Paced to real time (one 20ms frame per ~20ms, drift-
            corrected). The streaming source returns SILENCE frames
            while synthesis is still behind, and an unpaced reader
            queues them at CPU speed: one burst of starvation then plays
            back as seconds of inserted silence between her phrases --
            the fragmented voice. discord.py paces its own read loop;
            the local player must too.
            """
            frame_period = 1.0 / _READ_RATE_HZ
            next_due = time.perf_counter()
            try:
                while True:
                    chunk = source.read()
                    if not chunk:
                        break
                    outter.put(chunk)
                    if not outter.playing:  # stop() called mid-stream
                        break
                    next_due += frame_period
                    delay = next_due - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        # Fell behind (slow machine / hiccups): resync
                        # the schedule instead of accumulating debt.
                        next_due = time.perf_counter()
            except BaseException as exc:  # noqa: BLE001
                error_holder[0] = exc
                log.error(f"Local playback error: {exc!r}")
            finally:
                if outter.playing:
                    outter.finish()
                    # Let the device drain in real time ("wall" duration).
                    outter.wait_drained()
                outter.close()
                self._out = None
                # after() must fire from the loop thread, like discord.py.
                cb = after
                if cb is not None:
                    err = error_holder[0]
                    loop.call_soon_threadsafe(cb, err)

        self._thread = threading.Thread(
            target=_player, daemon=True, name="alice-local-player"
        )
        self._thread.start()

    def stop(self) -> None:
        """Abort playback immediately (barge-in / new-reply clamp)."""
        out = self._out
        if out is not None:
            out.abort()

    def cleanup(self) -> None:
        """Release the audio device (app shutdown)."""
        self.stop()


class LocalChannel:
    """Minimal stand-in for a discord voice channel.

    members: list of user-like objects (only .id and .bot are read).
    """

    def __init__(self, members) -> None:
        self.members = list(members)
