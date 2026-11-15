"""
desktop/app.py
==============
Tray entry point for the Alice desktop companion (Phase 2+3).

Run WITHOUT a console window via the desktop shortcut:

    venv\\Scripts\\pythonw.exe -m desktop.app

(plain `python -m desktop.app` works for debugging - same app, console
logs visible.)

Behavior
--------
- Single instance per machine (`CreateMutexW`; a second launch exits).
- A pystray icon in the tray whose right-click menu owns ALL controls:
  Companion on/off, Mute mic, Discord bot (Phase 4), STT engine
  (hosted/local, runtime flip), MateEngine restart, Quit.
- The voice session runs in its own daemon thread with its own asyncio
  loop (desktop.voice_app.CompanionSession), so menu handlers never block
  the heavy SoVITS/model boot.
- Mute = mic feed paused (device stays open, instant resume).
- Avatar: rendered by the external MateEngineX renderer, driven over the
  MtBridge TCP protocol by desktop/mate_bridge.py; its process lifecycle
  is owned here via desktop/engine_manager.py.
- Cold start: desktop/prewarm.py warms VAD / GPT-SoVITS / STT in the
  background while the engine boots; desktop/boot_timeline.py records
  stage timings ("Desktop boot (...)" log lines; DESKTOP_BOOT_TIMELINE=0
  to silence).
"""

from __future__ import annotations

import ctypes
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from bot.logging_setup import get_logger, setup_logging

setup_logging()
log = get_logger("desktop.app")

import config
from desktop.boot_timeline import boot_timeline
from desktop.discord_service import get_discord_service
from desktop.engine_manager import MateEngineManager
from desktop.lip_sync_subscriber import LipSyncSubscriber
from desktop.prewarm import start_prewarm
from desktop.voice_app import CompanionSession

_MUTEX_NAME = "AliceShimadaBotDesktop"
_ERROR_ALREADY_EXISTS = 183  # winerror.ERROR_ALREADY_EXISTS

# use_last_error: CPython's documented pattern -- a *second* ctypes
# call ("kernel32.GetLastError()") may observe a reset error state,
# while get_last_error() reads the captured error of OUR call.
_kernel32: ctypes.CDLL | None = None
if sys.platform == "win32":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def acquire_single_instance() -> int | None:
    """Hold a named mutex for the process lifetime.

    Returns the handle on success (OWNED by this process), None when
    another instance is already running (that handle is closed again).

    bInitialOwner=True: fresh creation takes ownership (so ReleaseMutex
    in release_single_instance is meaningful). If the object already
    exists, Windows ignores bInitialOwner and we exit immediately.
    """
    if _kernel32 is None:
        return None  # non-Windows: no mutex mechanism (tests/CI)

    handle = _kernel32.CreateMutexW(None, True, _MUTEX_NAME)
    err = ctypes.get_last_error()
    if handle == 0 or err == _ERROR_ALREADY_EXISTS:
        if handle:
            _kernel32.CloseHandle(handle)
        return None
    return handle


def release_single_instance(handle: int) -> None:
    """Destroy the mutex object: release ownership AND close the handle.

    ReleaseMutex alone is NOT enough -- the object persists until its
    last handle is closed, so a skipped CloseHandle would make the next
    launch hit ERROR_ALREADY_EXISTS for the stale object.
    """
    if _kernel32 is None:
        return  # non-Windows: nothing was ever acquired

    _kernel32.ReleaseMutex(handle)
    _kernel32.CloseHandle(handle)


def set_stt_engine(engine: str) -> bool:
    """Runtime STT engine flip (the tray STT submenu calls this).

    Hosted = Groq whisper-large-v3-turbo (best accent accuracy; mic
    audio leaves the machine). Local = faster-whisper fallback.

    Both engines are chosen per call by reading config.VOICE_STT_HOSTED
    (voice/pipeline.py + voice/stt_hosted.py), so a plain attribute
    flip is the entire switch. A VRAM-unload hook for local->hosted is
    Phase-later (see plan desktop-companion.md).
    """
    key = engine.strip().lower()
    if key in ("hosted", "groq", "cloud"):
        hosted = True
    elif key in ("local", "cpu", "offline"):
        hosted = False
    else:
        raise ValueError(f"unknown STT engine: {engine!r}")
    if hosted == config.VOICE_STT_HOSTED:
        return False
    config.VOICE_STT_HOSTED = hosted
    log.info("STT engine switched via tray: %s", key)
    return True


def _load_tray_image():
    """ assets/alice.ico when present, else a procedural fallback."""
    from pathlib import Path

    from PIL import Image

    icon = Path(__file__).resolve().parent.parent / "assets" / "alice.ico"
    if icon.is_file():
        try:
            return Image.open(icon)
        except Exception:  # noqa: BLE001 - fall through to procedural
            log.warning("Could not load %s; using fallback icon", icon)
    return _fallback_image()


def _fallback_image():
    """Procedural icon: violet ring + white 'A' + red mute dot."""
    from PIL import Image, ImageDraw, ImageFont

    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([8, 8, size - 8, size - 8], fill=(30, 27, 46, 255))
    d.ellipse([8, 8, size - 8, size - 8], outline=(124, 92, 255, 255), width=14)
    try:
        font = ImageFont.load_default(size=150)
    except Exception:  # noqa: BLE001 - tiny emoji-ish square still works
        font = ImageFont.load_default()
    d.text((size / 2, size / 2), "A", anchor="mm", fill=(240, 238, 255, 255),
           font=font)
    return img


def run() -> None:
    # 1. single instance
    boot_timeline.mark("app_start")
    handle = acquire_single_instance()
    if handle is None:
        log.warning("Desktop companion already running - exiting second launch")
        sys.exit(0)
    boot_timeline.mark("single_instance_ok")

    # 2. companion session (its own loop thread; boot is heavy; the heavy
    #    model/server warm-up now happens in prewarm.py in parallel)
    companion = CompanionSession()
    boot_timeline.mark("companion_managed")

    # 2b. MateEngine lifecycle (avatar renderer) - started in background;
    #     the manager POLLS the bridge and takes over as soon as it answers
    #     (BOOT_WAIT is only the ceiling, not a fixed sleep).
    engine = MateEngineManager()
    engine.start_async()
    boot_timeline.mark("engine_managing")

    # 2c. Voice-stack prewarm: VAD -> GPT-SoVITS server -> STT model, all
    #     in the background so it overlaps with the engine boot and the
    #     tray UX instead of the first utterance. DESKTOP_PREWARM=0 opts out.
    start_prewarm()
    boot_timeline.mark("prewarm_started")

    # Lip-sync subscriber: drives MateEngine mouth from TTS events
    lip_sync = LipSyncSubscriber()
    lip_sync.set_engine_manager(engine)
    lip_sync.start()
    boot_timeline.mark("lip_sync_started")

    # 3. overlay window (Qt WebEngine) is legacy/dead - MateEngine is the avatar.
    #    Kept as None so any defensive references degrade gracefully.
    overlay = None

    import threading

    import pystray  # lazy: keeps headless imports cheap for tests

    def _get_overlay() -> object | None:
        return overlay

    def _text_companion(_item=None) -> str:
        if companion.state == "off":
            return "Companion: off"
        if companion.state == "ready":
            return "Companion: listening"
        if companion.state == "error":
            return f"Companion: error ({companion.error})"
        return f"Companion: {companion.state}..."

    def _text_mute(_item=None) -> str:
        return "Unmute mic" if companion.muted else "Mute mic"

    def _text_overlay(_item=None) -> str:
        ov = _get_overlay()
        if ov is None:
            return "Overlay: unavailable"
        if ov.error:
            return f"Overlay: error ({ov.error})"
        return "Overlay: hide" if ov.is_visible else "Overlay: show"

    def _text_clickthrough(_item=None) -> str:
        ov = _get_overlay()
        if ov is None:
            return "Click-through: N/A"
        return "Click-through: on" if ov.click_through else "Click-through: off"

    def _click_companion(_item=None, _icon=None) -> None:
        if companion.state == "off" or companion.state == "error":
            companion.start()
        else:
            companion.set_muted(False)
            companion.stop()

    def _click_mute(_item=None, _icon=None) -> None:
        companion.set_muted(not companion.muted)

    def _click_overlay(_item=None, _icon=None) -> None:
        ov = _get_overlay()
        if ov is None:
            log.warning("Overlay unavailable - Qt runtime missing")
            return
        ov.toggle()

    def _click_clickthrough(_item=None, _icon=None) -> None:
        ov = _get_overlay()
        if ov is None:
            return
        ov.set_click_through(not ov.click_through)

    def _click_reapply_transparency(_item=None, _icon=None) -> None:
        ov = _get_overlay()
        if ov is None:
            return
        ov.reapply_transparency()

    def _text_engine(_item=None) -> str:
        return f"Engine: {engine.status}"

    def _text_mic(_item=None) -> str:
        try:
            name = companion.mic_device_name
        except Exception:  # noqa: BLE001
            name = "unknown"
        return f"Mic: {name}"

    def _list_input_devices():
        out = [(None, "System default")]
        try:
            import sounddevice as sd
            for idx, info in enumerate(sd.query_devices()):
                if isinstance(info, dict) and info.get("max_input_channels", 0) > 0:
                    out.append((idx, info.get("name", f"Device {idx}")))
        except Exception as exc:  # noqa: BLE001
            log.warning("device enumeration failed: %s", exc)
        return out

    def _checked_mic_device(device):
        def _fn(_item=None) -> bool:
            return companion.mic_device == device
        return _fn

    def _click_mic_device(device):
        def _handler(_item=None, _icon=None) -> None:
            log.info("Mic device selected: %s", device)
            companion.set_mic_device(device)
        return _handler

    def _click_restart_engine(_item=None, _icon=None) -> None:
        log.info("Engine restart requested from tray")
        engine.restart_async()

    def _click_load_test_page(_item=None, _icon=None) -> None:
        ov = _get_overlay()
        if ov is None:
            return
        ov.load_test_page()

    def _click_load_vrm_page(_item=None, _icon=None) -> None:
        ov = _get_overlay()
        if ov is None:
            return
        ov.load_vrm_page()

    def _enabled_companion(_item=None) -> bool:
        return companion.state not in ("booting", "stopping")

    def _enabled_mute(_item=None) -> bool:
        return companion.state == "ready"

    def _enabled_overlay(_item=None) -> bool:
        return _get_overlay() is not None

    def _text_discord(_item=None) -> str:
        svc = get_discord_service()
        return "Discord bot: stop" if svc.state == "on" else "Discord bot: start"

    def _click_discord(_item=None, _icon=None) -> None:
        svc = get_discord_service()
        if svc.state == "on" or svc.state == "stopping":
            svc.stop()
        elif svc.state == "off" or svc.state == "error":
            svc.start()

    def _enabled_discord(_item=None) -> bool:
        # Disable while starting/stopping to avoid menu freeze
        svc = get_discord_service()
        return svc.state not in ("starting", "stopping")

    # --- Debug console ---
    # NOTE: must be nonlocal (not global) -- _text_debug reads this via
    # closure; a `global` write would create a module-level shadow the
    # label never sees and the menu text would never update.
    _debug_console = False

    def _text_debug(_item=None) -> str:
        return "Debug console: close" if _debug_console else "Debug console: open"

    def _click_debug(_item=None, _icon=None) -> None:
        nonlocal _debug_console
        if not _debug_console:
            _kernel32.AllocConsole()
            # Redirect stdout/stderr to the new console
            import io
            sys.stdout = io.TextIOWrapper(open('CONOUT$', 'w'), write_through=True)  # noqa: SIM115 -- handle must outlive this scope (it becomes sys.stdout)
            sys.stderr = io.TextIOWrapper(open('CONOUT$', 'w'), write_through=True)  # noqa: SIM115 -- same
            _debug_console = True
            print("=== Alice Debug Console ===")
            print("Logs will appear here.")
        else:
            _kernel32.FreeConsole()
            _debug_console = False

    def _enabled_debug(_item=None) -> bool:
        return True

    def _click_stt_hosted(_item=None, _icon=None) -> None:
        set_stt_engine("hosted")

    def _click_stt_local(_item=None, _icon=None) -> None:
        set_stt_engine("local")

    def _checked_hosted(_item=None) -> bool:
        return bool(config.VOICE_STT_HOSTED)

    def _checked_local(_item=None) -> bool:
        return not config.VOICE_STT_HOSTED

    def _click_quit(icon=None, _item=None) -> None:
        log.info("Quit requested from tray")
        companion.set_muted(False)
        # Stop companion with timeout
        try:
            companion.stop()
        except Exception:  # noqa: BLE001
            log.exception("Companion stop error")
        if overlay is not None:
            try:
                overlay.close()
            except Exception:  # noqa: BLE001
                log.exception("Overlay close error")
        # Stop MateEngine if we launched it
        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            log.exception("Engine stop error")
        try:
            lip_sync.stop()
        except Exception:  # noqa: BLE001
            log.exception("Lip-sync stop error")
        if icon is not None:
            icon.stop()
        # Force exit if icon.stop() doesn't terminate the process
        import os
        os._exit(0)

    # --- Screen glance ---------------------------------------------------
    def _text_look_now(_item=None) -> str:
        return "Look at my screen now"

    def _click_look_now(_item=None, _icon=None) -> None:
        log.info("Glance requested from tray")
        companion.glance_now()

    def _text_glance_allowed(_item=None) -> str:
        return "Screen glances allowed"

    def _click_glance_allowed(_item=None, icon=None) -> None:
        new_value = not companion.glance_allowed
        companion.glance_set_allowed(new_value)
        log.info(f"Screen glances permission: {new_value}")

    def _checked_glance_allowed(_item=None) -> bool:
        return companion.glance_allowed

    menu = pystray.Menu(
        pystray.MenuItem(_text_companion, _click_companion, default=True,
                         enabled=_enabled_companion),
        pystray.MenuItem(_text_mute, _click_mute, enabled=_enabled_mute),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(_text_look_now, _click_look_now),
        pystray.MenuItem(
            _text_glance_allowed, _click_glance_allowed,
            checked=_checked_glance_allowed,
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(
            "STT engine",
            pystray.Menu(
                pystray.MenuItem(
                    "Hosted (Groq, accurate w/ accents)",
                    _click_stt_hosted,
                    radio=True,
                    checked=_checked_hosted,
                ),
                pystray.MenuItem(
                    "Local fallback (CPU/GPU)",
                    _click_stt_local,
                    radio=True,
                    checked=_checked_local,
                ),
            ),
        ),
        pystray.Menu.SEPARATOR,
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(_text_engine, _click_restart_engine),
        pystray.MenuItem(
            "Microphone",
            pystray.Menu(
                *[
                    pystray.MenuItem(
                        name,
                        _click_mic_device(dev),
                        radio=True,
                        checked=_checked_mic_device(dev),
                    )
                    for dev, name in _list_input_devices()
                ]
            ),
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(_text_discord, _click_discord, enabled=_enabled_discord),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(_text_debug, _click_debug, enabled=_enabled_debug),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", _click_quit),
    )

    icon = pystray.Icon(
        "Alice", _load_tray_image(), "Alice",
        menu,
    )
    log.info("Tray icon starting (single instance acquired)")
    boot_timeline.mark("tray_running")
    boot_timeline.log_summary("tray")

    # pystray evaluates dynamic item text only when it renders the menu,
    # so a menu left open keeps showing "Companion: booting..." long after
    # the session went ready. Nudge it periodically; a no-op when closed.
    def _refresh_menu() -> None:
        while True:
            time.sleep(2.0)
            try:
                icon.update_menu()
            except Exception:  # noqa: BLE001 - icon gone (quit)
                return

    threading.Thread(target=_refresh_menu, daemon=True, name="menu-refresh").start()

    # --- pystray on the main thread -----------------------------------------
    # No overlay window anymore (MateEngine is the avatar), so pystray owns
    # the main thread's message loop -> responsive tray even while the
    # companion/engine boot in the background.
    try:
        import pythoncom

        pythoncom.CoInitialize()
        try:
            icon.run()
        except Exception:  # noqa: BLE001
            log.exception("Pystray tray loop error")
        finally:
            pythoncom.CoUninitialize()
    except ModuleNotFoundError:  # pywin32 not installed
        try:
            icon.run()
        except Exception:  # noqa: BLE001
            log.exception("Pystray tray loop error")

    # --- cleanup after tray exits -------------------------------------------
    try:
        companion.set_muted(False)
        companion.stop()
        engine.stop()
        lip_sync.stop()
    except Exception:  # noqa: BLE001
        log.exception("Companion cleanup error at exit")
    release_single_instance(handle)


if __name__ == "__main__":
    run()
