"""
desktop/voice_app.py
====================
Shared companion engine (Phase 2): ONE bootstrap that both the CLI
(`desktop/cli.py`, debug/self-test) and the tray app (`desktop/app.py`,
production) drive, so they can never drift apart.

Everything here keeps the Phase 1 verified semantics:

- Local voice client replicating Discord's transport contract
  (20ms s16le 48kHz interleaved stereo, b"" EOF, after() once).
- One mic = one speaker (diarization opt-in only).
- Speech-task supervision + settle wait + drained teardown
  (prevents 'Event loop is closed' / exit 1).
- Microphone maintained by MicInput (echo gate per VOICE_DESKTOP_ECHO_GATE).
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from bot.logging_setup import get_logger

log = get_logger("desktop.voice_app")

import contextlib

import config
from desktop.boot_timeline import boot_timeline
from desktop.local_client import SdOutput


class _OwnerStub:
    def __init__(self, user_id: int) -> None:
        self.id = user_id


def build_pipeline(owner_id: int):
    """Assemble bot stub + VoicePipeline (same recipe as manage.py latency)."""
    from ai import get_ai_client
    from character import load_character
    from desktop.boot_timeline import boot_timeline
    from voice.pipeline import VoicePipeline

    bot = SimpleNamespace()
    bot.user = _OwnerStub(owner_id)
    bot.character = load_character(config.CHARACTER_NAME)
    bot.ai_client = get_ai_client()
    pipeline = VoicePipeline(bot)
    boot_timeline.mark("pipeline_built")
    from voice import readiness

    readiness._state = readiness.STATE_READY
    return pipeline, bot


def wav_as_48k(path: str) -> np.ndarray:
    """Load a WAV and return mono float32 at 48kHz (the pipeline's input)."""
    import scipy.io.wavfile as wavfile
    import scipy.signal

    sr, raw = wavfile.read(str(path))
    if raw.dtype == np.int16:
        audio = raw.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        audio = raw.astype(np.float32) / 2147483648.0
    else:
        audio = np.asarray(raw, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 48000:
        audio = scipy.signal.resample(audio, int(len(audio) * 48000 / sr))
    return np.asarray(audio, dtype=np.float32)


async def wait_for_settle(
    pipeline, sink, client, owner_id: int, guild_id: int,
    speech_tasks: set, timeout: float = 120.0, settle: float = 2.0,
) -> None:
    """Wait until the whole turn (STT/decision/reply/playback) is done.

    speech_tasks: set of in-flight `VoicePipeline._process_speech` tasks
    (trivially aware of every awaited stage inside the utterance chain --
    hosted STT, diarization, decisions, reply generation and playback
    drains -- because they all happen inside that coroutine).

    A turn is settled when no speech task is running AND every secondary
    indicator is clear for `settle` seconds in a row:
    - mixer playback / local device playback
    - any speech still buffered (could be the next utterance)
    """
    deadline = asyncio.get_running_loop().time() + timeout
    settled_at: float | None = None
    while asyncio.get_running_loop().time() < deadline:
        busy = bool(speech_tasks)
        if pipeline.mixer.is_playing(guild_id) or client.is_playing():
            busy = True
        if sink.observation(owner_id) is not None:
            busy = True

        if busy:
            settled_at = None
            await asyncio.sleep(0.2)
            continue
        if settled_at is None:
            settled_at = asyncio.get_running_loop().time()
        elif asyncio.get_running_loop().time() - settled_at >= settle:
            return
        await asyncio.sleep(0.1)
    log.warning("Settle timeout reached (%.0fs); proceeding to teardown", timeout)


async def teardown(monitor_task, mic, client, sink, guild_id: int) -> None:
    """Stop capture + monitor, then let background work finish cleanly.

    Debug/logging worst case caused by tearing the loop down under live
    continuation is 'Event loop is closed' (exit 1) -- so drain every
    remaining task before asyncio.run's implicit close.
    """
    if mic is not None:
        mic.stop()
    if monitor_task is not None:
        monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await monitor_task
        log.info("Speech monitor cancelled")
        client.cleanup()
        from voice.state import voice_state

        voice_state.disconnect(guild_id)
        sink.cleanup()
        # Drain any stragglers so the loop closes with no pending work.
        current = asyncio.current_task()
        pending = [
            t for t in asyncio.all_tasks()
            if t is not current and not t.done()
        ]
        if pending:
            _, still_pending = await asyncio.wait(
                pending, timeout=5.0
            )
            for t in still_pending:
                log.warning("Background task did not finish in time: %r", t)


async def inject_speech(sink, owner, audio48k: np.ndarray) -> None:
    """Replay a mono 48k clip into the sink as 20ms stereo blocks."""
    block = 960  # 20ms @ 48k
    n = len(audio48k)
    for i in range(0, n, block):
        mono_i16 = (np.clip(audio48k[i:i + block], -1.0, 1.0) * 32767.0) \
            .astype(np.int16)
        stereo = np.empty(len(mono_i16) * 2, dtype=np.int16)
        stereo[0::2] = mono_i16
        stereo[1::2] = mono_i16
        sink.write(owner, SimpleNamespace(pcm=stereo.tobytes()))
        await asyncio.sleep(0.02)  # real-time pacing like mic frames
    log.info(f"Injected {n / 48000:.2f}s of speech as the owner.")


async def boot_companion(
    owner_id: int,
    guild_id: int = 0,
    device: int | None = None,
    silent: bool = False,
    diarize: bool = False,
    on_barge=None,
    start_mic: bool = True,
) -> SimpleNamespace:
    """One shared bootstrap: pipeline -> sink -> local client -> monitor.

    Returns a SimpleNamespace ("booted") holding everything teardown
    needs; the caller owns idling (CLI prints, thread idles for tray)
    and finally calls shutdown_companion(booted).

    start_mic=False builds the MicInput but leaves it unstarted -- the
    CLI self-test wants a deterministic injected-only turn (old Phase 1
    semantics); the tray session always stars the mic at boot.

    Desktop rationale (Phase 1, verified): one microphone means one
    speaker by definition, so per-utterance diarization is opt-in only
    (pyannote is ~1GB GPU and seconds-cold for nothing here).
    Predictive local STT is skipped inside the pipeline when hosted STT
    is on; MicInput respects config.VOICE_DESKTOP_ECHO_GATE.
    """
    from desktop.local_client import LocalChannel, LocalVoiceClient, SilentOutput

    # voice_state is the VoiceStateManager INSTANCE the pipeline uses.
    from voice.state import voice_state

    pipeline, bot = build_pipeline(owner_id)

    # Cold-start parity with voice/pipeline.init_pipeline(): the VAD hot
    # path never lazy-loads Silero by design, so without an explicit
    # preload the desktop would run on the RMS fallback. prewarm.py
    # normally did this at app launch, making this a no-op for the tray;
    # it still covers standalone CLI runs.
    try:
        from voice import vad as _vad

        if not _vad.is_model_loaded():
            t_vad = time.perf_counter()
            _vad.preload()
            boot_timeline.mark("vad_ready")
            log.info(
                "VAD preloaded in %.1fs",
                time.perf_counter() - t_vad,
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("VAD preload failed (RMS fallback stays): %s", exc)

    # Same for STT when running standalone (tray path schedules this in
    # prewarm.py instead): start the background model load so the first
    # utterance races it instead of paying for it. Hosted STT decodes
    # nothing locally -- don't pin the whisper weights into VRAM.
    try:
        if config.VOICE_STT_HOSTED:
            log.info("STT preload skipped (VOICE_STT_HOSTED=true; local engine stays lazy)")
        else:
            from voice import stt as _stt

            if not _stt.is_loaded():
                threading.Thread(
                    target=_stt.preload, name="stt-preload", daemon=True
                ).start()
                log.info("STT preload started in background")
    except Exception as exc:  # noqa: BLE001
        log.warning(f"STT preload schedule failed: {exc}")

    boot_timeline.mark("companion_boot")

    if not diarize:
        config.VOICE_DIARIZE_MODE = "none"

    # Trace in-flight utterance chains: the pipeline spawns its speech
    # processing as a background task; supervise the set so the runner
    # waits for the FULL turn (STT/decision/reply/playback) before
    # teardown (a mid-utterance loop close = 'Event loop is closed').
    speech_tasks: set = set()
    _orig_speech = pipeline._process_speech

    async def _speech_tracked(*a, **k):
        task = asyncio.current_task()
        speech_tasks.add(task)
        try:
            return await _orig_speech(*a, **k)
        finally:
            speech_tasks.discard(task)

    pipeline._process_speech = _speech_tracked

    sink = pipeline.get_sink(guild_id)

    # Local voice client (speakers or silent) + channel having just the owner.
    factory = SilentOutput if silent else None
    channel = LocalChannel(members=[bot.user])

    # MateEngine avatar is driven by desktop.lip_sync_subscriber (lip-sync events),
    # not RMS energy. No Qt overlay anymore.
    rms_callback = None

    # Resolve the output device by name each boot: Windows re-indexes
    # devices when hardware is plugged/unplugged (a hardcoded index
    # silently routes audio elsewhere or raises OSError).
    output_device = None
    g435_found = False
    try:
        import sounddevice as sd

        for idx, info in enumerate(sd.query_devices()):
            if info.get("max_output_channels", 0) > 0 and \
                    "G435" in str(info.get("name", "")):
                # Matches both wired ("Headphones (G435...") and the
                # Bluetooth variant ("Headset (G435 Bluetooth ...") --
                # the Bt name has a different prefix.
                output_device = idx
                g435_found = True
                break
        if output_device is None:
            log.info("G435 output not found; using system default output")
        else:
            log.info("Output device %d: %s", output_device,
                     sd.query_devices()[output_device]["name"])
    except Exception as exc:
        log.warning("Device enumeration failed; using default output: %s", exc)

    # WebRTC APM (AEC + NS) -- the Discord-class mic DSP. When active,
    # the mic stream is echo-cancelled against Alice's own playback, so
    # the raw speakers echo guard becomes unnecessary: the residual is
    # judged by the gate/barge logic instead (her cancelled echo drops,
    # the user's voice survives). Falls back to the gate-only path when
    # the package is missing or init fails.
    apm = None
    if config.VOICE_MIC_APM:
        from desktop.dsp import from_config as _apm_from_config

        apm = _apm_from_config(delay_ms=config.VOICE_APM_DELAY_MS)
        if apm is not None:
            log.info("Mic APM active -> echo gate bypassed (AEC handles speakers)")

    # VOICE_DESKTOP_ECHO_GATE=false asserts "the mic can't hear her" --
    # only true when she plays through headphones. When the G435 is
    # absent the output falls back to speakers, which the mic CAN hear:
    # her playback barges in on itself (glitchy chop) and her
    # backchannels reset the turn-end silence (turns drag on forever).
    # Force speakers mode in that case: mic drops while she plays unless
    # a sustained loud burst (a real interrupt) crosses the threshold.
    echo_gate = config.VOICE_DESKTOP_ECHO_GATE or not g435_found
    if apm is not None:
        echo_gate = False  # AEC-subtracted residual replaces the gate
    elif echo_gate != config.VOICE_DESKTOP_ECHO_GATE:
        log.warning(
            "G435 absent -> output is on speakers; forcing echo gate ON "
            "(reconnect the headset or update VOICE_DESKTOP_ECHO_GATE)"
        )

    client = LocalVoiceClient(
        guild_id=guild_id,
        channel=channel,
        output_factory=factory or (
            lambda: SdOutput(
                rms_callback=rms_callback,
                device=output_device,
                far_end_callback=(apm.process_far if apm is not None else None),
                preroll_ms=config.VOICE_TTS_PREROLL_MS,
            )
        ),
        rms_callback=rms_callback,
    )
    voice_state.connect(guild_id, channel_id=guild_id, voice_client=client)

    monitor_task = asyncio.create_task(pipeline._monitor_loop(guild_id))
    boot_timeline.mark("monitor_started")

    # Live microphone per phase-1 semantics.
    from desktop.mic_input import MicInput

    def _playing() -> bool:
        return client.is_playing()

    mic = MicInput(
        sink,
        owner=bot.user,
        playing_fn=_playing,
        device=device,
        echo_gate=echo_gate,
        agc=config.VOICE_MIC_AGC,
        agc_target=config.VOICE_MIC_AGC_TARGET,
        agc_max_gain=config.VOICE_MIC_AGC_MAX_GAIN,
        apm=apm,
        suppress_fn=lambda: voice_state.any_discord_voice_active(),
    )
    mic.on_barge = on_barge
    if start_mic:
        mic.start()

    # Screen glance service (provider + ambient governor). The tray owns
    # the permission toggle; without it no capture ever happens.
    look_service = None
    try:
        from desktop.look_service import LookService

        look_service = LookService(pipeline, mate_bridge=None)
        look_service.set_allowed(bool(getattr(config, "SCREEN_LOOK_ENABLED", False)))
        if config.SCREEN_LOOK_AMBIENT:
            look_service.start_ambient()
            log.info("Ambient screen glances enabled")
    except Exception as exc:  # noqa: BLE001 - glance is optional
        log.warning(f"LookService init failed (glances disabled): {exc}")

    boot_timeline.mark("session_ready")
    boot_timeline.log_summary("companion")

    return SimpleNamespace(
        pipeline=pipeline, bot=bot, sink=sink, client=client,
        monitor_task=monitor_task, mic=mic, speech_tasks=speech_tasks,
        owner_id=owner_id, guild_id=guild_id,
        look_service=look_service,
    )


async def shutdown_companion(booted) -> None:
    """Mirror of boot_companion: stop everything and drain the loop."""
    look_service = getattr(booted, "look_service", None)
    if look_service is not None:
        with contextlib.suppress(Exception):
            await look_service.stop_ambient()
    await teardown(
        booted.monitor_task, booted.mic, booted.client, booted.sink,
        booted.guild_id,
    )


async def inject_turn(booted, wav_path: str) -> None:
    """One replayed turn through the REAL pipeline (self-test helper)."""
    audio = wav_as_48k(wav_path)
    log.info(
        f"Self-test: replay {wav_path} ({len(audio) / 48000:.2f}s) | "
        f"live={config.VOICE_LIVE_MODE} "
        f"stt_hosted={config.VOICE_STT_HOSTED} tts={config.VOICE_TTS_ENGINE}"
    )
    await inject_speech(booted.sink, booted.bot.user, audio)
    await wait_for_settle(
        booted.pipeline, booted.sink, booted.client, booted.owner_id,
        booted.guild_id, speech_tasks=booted.speech_tasks,
    )


class CompanionSession:
    """Desktop voice session driven from a foreign thread (tray menu).

    Lifecycle runs in a dedicated daemon thread with its own asyncio
    loop, so the tray stays responsive while SoVITS/model boot happens.

    States: off -> booting -> ready -> stopping -> off ; or booting -> error.
    Thread-safe: start/stop/set_muted/status are callable from pystray's
    menu thread.
    """

    def __init__(
        self,
        device: int | None = None,
        owner_id: int | None = None,
        guild_id: int = 0,
        silent: bool = False,
        diarize: bool = False,
        on_barge=None,
    ) -> None:
        self._device = device
        self._owner_id = owner_id
        self._guild_id = guild_id
        self._silent = silent
        self._diarize = diarize
        self._on_barge = on_barge

        self.state = "off"
        self.error: str | None = None
        self.muted = False

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_evt: asyncio.Event | None = None
        self._booted: SimpleNamespace | None = None
        self._glance_allowed = False

    # --- public, tray-thread safe ---------------------------------------

    def start(self) -> bool:
        """Bring the voice session up. Returns False if already alive."""
        if self.state not in ("off", "error") or (
            self._thread is not None and self._thread.is_alive()
        ):
            return False
        self.error = None
        self.state = "booting"
        self._thread = threading.Thread(
            target=self._run_session, daemon=True, name="companion-voice"
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Bring the session down; blocks (join) until torn down or timeout."""
        thread = self._thread
        if thread is None:
            return
        loop, evt = self._loop, self._stop_evt
        if loop is not None and evt is not None:
            with contextlib.suppress(RuntimeError):
                # loop already closed; thread is winding down
                loop.call_soon_threadsafe(evt.set)
        thread.join(timeout=120.0)
        if thread.is_alive():
            log.warning("Companion thread did not stop within join timeout")
        self._thread = None

    def set_muted(self, muted: bool) -> None:
        self.muted = bool(muted)
        if self._booted is not None and self._booted.mic is not None:
            self._booted.mic.muted = self.muted

    def set_mic_device(self, device: int | None) -> None:
        self._device = device
        if self.state == "ready":
            was_muted = self.muted
            self.stop()
            self.muted = was_muted
            self.start()

    # --- screen glance (tray-thread safe) --------------------------------

    def glance_set_allowed(self, value: bool) -> None:
        """Tray permission gate: no capture happens while False."""
        self._glance_allowed = bool(value)
        booted = self._booted
        ls = getattr(booted, "look_service", None) if booted else None
        if ls is not None:
            ls.set_allowed(self._glance_allowed)

    def glance_now(self) -> None:
        """Tray 'Look at my screen now' — marshals into the session loop."""
        booted = self._booted
        ls = getattr(booted, "look_service", None) if booted else None
        loop = self._loop
        if ls is None or loop is None:
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(ls.look_now_async)

    @property
    def glance_allowed(self) -> bool:
        return getattr(self, "_glance_allowed", False)

    def status(self) -> dict:
        return {
            "state": self.state,
            "error": self.error,
            "muted": self.muted,
        }

    @property
    def mic_device_name(self) -> str:
        booted = self._booted
        if booted is None:
            return "not started"
        mic = getattr(booted, "mic", None)
        if mic is None:
            return "none"
        try:
            return mic.device_name
        except Exception:  # noqa: BLE001
            return "unknown"

    @property
    def mic_device(self) -> int | None:
        return self._device

    # --- session thread --------------------------------------------------

    def _run_session(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._stop_evt = asyncio.Event()
        try:
            self._loop.run_until_complete(self._session_coro())
        except Exception as exc:  # noqa: BLE001 surface to tray
            log.exception("Companion session failed")
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "error"
        finally:
            asyncio.set_event_loop(None)
            self._loop.close()
            self._loop = None

    async def _session_coro(self) -> None:
        booted = await boot_companion(
            owner_id=self._owner_id
            or (next(iter(config.OWNER_IDS)) if config.OWNER_IDS else 12345),
            guild_id=self._guild_id,
            device=self._device,
            silent=self._silent,
            diarize=self._diarize,
            on_barge=self._on_barge,
        )
        self._booted = booted
        if self.muted:
            booted.mic.muted = True
        ls = getattr(booted, "look_service", None)
        if ls is not None and self._glance_allowed:
            ls.set_allowed(True)
        if self._stop_evt.is_set():  # stop() raced with boot
            self.state = "stopping"
            await shutdown_companion(booted)
            self._booted = None
            self.state = "off"
            return
        self.state = "ready"
        log.info("Companion session ready (tray-managed)")
        await self._stop_evt.wait()
        self.state = "stopping"
        await shutdown_companion(booted)
        self._booted = None
        self.state = "off"
        log.info("Companion session stopped")
