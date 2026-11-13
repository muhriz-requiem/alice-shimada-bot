"""
desktop/mic_input.py
====================
Microphone capture that feeds the EXISTING AliceVoiceSink logic --
mimicking what discord-ext-voice-recv delivers:

- 20ms blocks of 16-bit 48kHz interleaved STEREO s16le PCM
- wrapped as `data.pcm` bytes (sink.write deinterleaves to mono)

Reusing the sink means Silero VAD, speech buffering, silence windows,
observation()/revision tracking and the pipeline's monitor loop all run
unchanged: the desktop path and the Discord path are the same code.

Speaker-echo policy: while Alice is playing TTS the mic feed is GATED
(speakers feed back into the mic otherwise, and Alice would "hear
herself"). Barging in with a raised, sustained voice still gets
through: blocks above `barge_rms` RMS for `barge_blocks` consecutive
blocks (~0.4s) are fed while she speaks, and the existing live-mode
interrupt path takes it from there. Headset use makes this moot.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np

from bot.logging_setup import get_logger

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None

log = get_logger("desktop.mic_input")

_SAMPLE_RATE = 48000
_BLOCK_MS = 20
_BLOCKSIZE = _SAMPLE_RATE * _BLOCK_MS // 1000  # 960 samples


class MicInput:
    """Microphone -> window frames -> (gated) AliceVoiceSink.write."""

    def __init__(
        self,
        sink,
        owner,
        playing_fn: Callable[[], bool],
        device: Optional[int] = None,
        echo_gate: bool = True,
        barge_rms: float = 0.12,
        barge_blocks: int = 20,
        on_barge: Optional[Callable[[], None]] = None,
        agc: bool = False,
        agc_target: float = 0.12,
        agc_max_gain: float = 32.0,
        apm=None,
        suppress_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Build a mic feed.

        sink:       AliceVoiceSink instance (buffers + Silero VAD inside)
        owner:      user-like stub; its .id keys the sink buffers
        playing_fn: lambda -> True while the local voice client plays
        device:     sounddevice input index (None = system default)
        echo_gate:  True = drop mic blocks while Alice plays UNLESS a
                    sustained burst crosses `barge_rms` (headphone-less
                    setups). False = always feed (headphones: the mic
                    physically can't hear her; true barge-in at any
                    volume)  [see config.VOICE_DESKTOP_ECHO_GATE]
        barge_rms:  mono RMS above which a loud block counts as an
                    interrupt while Alice is speaking
        barge_blocks: consecutive loud blocks (~20ms each) needed before
                    the interrupt feed opens (~0.4s)
        on_barge:   optional hook fired the moment an interrupt is
                    detected (used by the tray UI's visual state).
        agc:        normalize faint capture devices toward `agc_target`
                    RMS (fast attack, slow release, gain capped at
                    `agc_max_gain`, never attenuates). The boosted signal
                    feeds the VAD/STT path and the barge-in burst check.
        apm:        optional desktop.dsp.MicApm (WebRTC AEC + NS). When
                    set, each block is echo-cancelled against Alice's
                    playback before gating, and the barge/burst decision
                    runs on the PROCESSED level: her cancelled echo is
                    quiet (gated), the user's voice survives (fed).
        """
        if sd is None:
            raise RuntimeError(
                "sounddevice is not available; local mic capture needs it "
                "(pip install sounddevice)"
            )
        self.sink = sink
        self.owner = owner
        self._playing_fn = playing_fn
        self._device = device
        self.echo_gate = bool(echo_gate)
        self.barge_rms = float(barge_rms)
        self.barge_blocks = int(barge_blocks)
        self.on_barge = on_barge
        self.agc = bool(agc)
        self._agc_target = float(agc_target)
        self._agc_max_gain = float(agc_max_gain)
        # Blocks below this are treated as silence: the gain envelope may
        # keep rising toward its cap, but their (tiny) contribution is
        # never chased up to the target level. 0.0008 (not 0.002): this
        # mic's actual SPEECH onset sits around 0.002 -- soft first
        # syllables must still count as speech, not silence.
        self._agc_floor = 0.0008
        self._agc_gain = 1.0
        self._agc_logged = False
        self._apm = apm
        # Discord-call stand-down: when a real Discord voice session is
        # active, the desktop companion goes quiet entirely -- otherwise
        # two Alices speak at once (the call's Alice + this one reacting
        # to the room audio through the same headset).
        self._suppress_fn = suppress_fn
        self._suppress_active = False
        self._suppress_logged = False
        # Tray Mute: checked FIRST in _handle_block; keeps the device
        # open so unmute resumes instantly.
        self.muted = False

        self._q: "queue.SimpleQueue" = queue.SimpleQueue()
        self._stream: Optional[sd.InputStream] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        # Echo/barge state
        self._burst = 0
        self._last_loud = 0.0
        self.stats = {"blocks": 0, "fed": 0, "gated": 0, "barge_fed": 0,
                      "muted": 0, "suppressed": 0}
        self._owner_stub_key = getattr(owner, "id", None)

    # --- PortAudio callback thread ---------------------------------------
    def _audio_callback(self, indata, frames, t, status) -> None:
        """PortAudio thread: enqueue the raw block; keep it cheap."""
        if status:
            log.debug(f"mic status: {status}")
        mono = indata[:, 0].reshape(-1)  # int16
        # RMS on float-normalized samples
        rms = float(np.sqrt(np.mean((mono.astype(np.float32) / 32768.0) ** 2)))
        # Interleave stereo the way discord's decoder emits it: LRLR...
        stereo = np.empty(len(mono) * 2, dtype=np.int16)
        stereo[0::2] = mono
        stereo[1::2] = mono
        self._q.put((rms, stereo.tobytes()))

    # --- feeder thread -----------------------------------------------------
    def _update_agc(self, rms: float) -> float:
        """Advance the AGC envelope for one block; return the gain to apply.

        Gain snaps DOWN toward what the current block needs (so close-up
        speech never overshoots into clipping) and creeps UP during
        quiet stretches (so faint far-field speech reaches the VAD/STT
        at a workable level). Never attenuates; never exceeds
        `self._agc_max_gain`.
        """
        if not self.agc:
            return 1.0
        if rms >= self._agc_floor:
            desired = min(self._agc_target / max(rms, 1e-6), self._agc_max_gain)
            if desired <= self._agc_gain:
                self._agc_gain = max(desired, 1.0)  # fast attack (down)
            else:
                self._agc_gain = min(
                    desired, self._agc_gain + (desired - self._agc_gain) * 0.25
                )
        else:
            # Silence: release slowly toward the cap so the next faint
            # utterance arrives already boosted.
            self._agc_gain = self._agc_gain + (
                self._agc_max_gain - self._agc_gain
            ) * 0.02
        if not self._agc_logged and self._agc_gain > 2.0:
            self._agc_logged = True
            log.info(
                "Mic AGC engaged (gain=%.1fx, target RMS %.2f)",
                self._agc_gain, self._agc_target,
            )
        return self._agc_gain

    def _apply_gain(self, pcm_bytes: bytes, gain: float) -> bytes:
        """Scale int16 (stereo-interleaved) PCM by `gain` with clip guard."""
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        scaled = np.clip(pcm.astype(np.float32) * gain, -32768.0, 32767.0)
        return scaled.astype(np.int16).tobytes()

    @staticmethod
    def _block_rms(pcm_bytes: bytes) -> float:
        """Mono RMS of one int16 stereo-interleaved block."""
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        if len(pcm) == 0:
            return 0.0
        mono = pcm.reshape(-1, 2).mean(axis=1) if len(pcm) % 2 == 0 else pcm
        return float(np.sqrt(np.mean((mono.astype(np.float32) / 32768.0) ** 2)))

    def _handle_block(self, rms: float, pcm_bytes: bytes) -> None:
        """Apply AGC + echo-gating for one 20ms block and feed the sink if allowed.

        Isolated from the queue loop so unit tests can drive blocks
        directly without audio devices or threads.
        """
        self.stats["blocks"] += 1
        if self.muted:
            self.stats["muted"] += 1
            return

        # Discord-call stand-down: a real Discord voice session owns the
        # conversation -- this feed must not produce a second Alice.
        if self._suppress_fn is not None and self._suppress_fn():
            if not self._suppress_logged:
                log.info("Desktop mic suppressed: Discord voice session active")
                self._suppress_logged = True
            self._suppress_active = True
            self.stats["suppressed"] += 1
            return
        if self._suppress_active:
            self._suppress_active = False
            self._suppress_logged = False
            log.info("Desktop mic resumed (no active Discord voice session)")

        # AGC FIRST: normalize the faint capture toward a sane level
        # BEFORE the APM sees it. WebRTC's NS treats low-level speech as
        # noise and would erase it (the 2026-11-11 speech-ratio collapse:
        # APM-first drove the VAD ratio back to ~0.2%). The boosted
        # signal is what the AEC/NS were designed to operate on.
        gain = self._update_agc(rms)
        rms_eff = min(rms * gain, 1.0)
        if gain > 1.01:
            pcm_bytes = self._apply_gain(pcm_bytes, gain)

        # Then the APM: subtract Alice's playback (far end) and suppress
        # residual noise; the gate/barge decisions below judge the
        # PROCESSED block so her cancelled echo can't re-trigger them.
        if self._apm is not None:
            try:
                pcm_bytes = self._apm.process_near(pcm_bytes)
                rms_eff = min(self._block_rms(pcm_bytes), 1.0)
            except Exception:  # noqa: BLE001 - DSP must never kill capture
                log.debug("APM process failed; feeding raw block", exc_info=True)

        if not self.echo_gate:
            # Ungated (headphones): continuous listening -- real barge-in
            # detection via VAD, no burst/timing heuristics needed.
            self._burst = 0
            self.sink.write(self.owner, SimpleNamespace(pcm=pcm_bytes))
            self.stats["fed"] += 1
            return
        playing = bool(self._playing_fn())
        if not playing:
            self._burst = 0
            self.sink.write(self.owner, SimpleNamespace(pcm=pcm_bytes))
            self.stats["fed"] += 1
        elif rms_eff > self.barge_rms:
            self._burst += 1
            if self._burst >= self.barge_blocks:
                if self._last_loud == 0.0 and self.on_barge is not None:
                    try:
                        self.on_barge()
                    except Exception:  # noqa: BLE001
                        pass
                self._last_loud = time.monotonic()
                self.sink.write(self.owner, SimpleNamespace(pcm=pcm_bytes))
                self.stats["barge_fed"] += 1
        else:
            self.stats["gated"] += 1
            if self._burst > 0:
                self._burst = 0
            if self._last_loud and time.monotonic() - self._last_loud > 1.0:
                self._last_loud = 0.0

    def _feeder_loop(self) -> None:
        """Drain queued blocks into the sink (with echo gating)."""
        while self._started:
            try:
                rms, pcm_bytes = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._handle_block(rms, pcm_bytes)

    def start(self) -> None:
        """Open the input stream and start the feeder thread."""
        if self._started:
            return
        self._stream = sd.InputStream(
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=_BLOCKSIZE,
            device=self._device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._started = True
        self._thread = threading.Thread(
            target=self._feeder_loop, daemon=True, name="alice-mic-feeder"
        )
        self._thread.start()
        dev = self._stream.device if self._stream is not None else self._device
        try:
            info = sd.query_devices(dev)
            dev_name = info.get("name", str(dev)) if isinstance(info, dict) else str(info)
        except Exception:  # noqa: BLE001
            dev_name = str(dev)
        self._device_name = dev_name
        log.info(f"Mic capture started: device={dev} ({dev_name}), 48kHz mono, 20ms blocks")

    @property
    def device_name(self) -> str:
        if getattr(self, "_device_name", None):
            return self._device_name
        dev = self._device
        try:
            info = sd.query_devices(dev)
            if isinstance(info, dict):
                return info.get("name", str(dev))
            return str(info)
        except Exception:  # noqa: BLE001
            return str(dev) if dev is not None else "default"

    def stop(self) -> None:
        """Stop capture and release the device."""
        self._started = False
        thr = self._thread
        if thr is not None:
            thr.join(timeout=2.0)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        log.info("Mic capture stopped")
