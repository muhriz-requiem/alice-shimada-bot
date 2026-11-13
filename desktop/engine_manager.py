"""MateEngine lifecycle manager for the tray app.

Owns the external MateEngineX renderer process: starts it on demand, waits
for boot, takes over the face via MtBridge, and terminates it on shutdown.

The tray app previously assumed MateEngine was already running (started
manually). This module makes the companion self-contained: launching the
app brings the avatar up, quitting takes it down.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import threading
import time

from bot.logging_setup import get_logger

log = get_logger("desktop.engine_manager")

DEFAULT_EXE = r"D:\Temp\opencode\MateEngine\MateEngineX.exe"
BOOT_WAIT = 32.0  # ceiling seconds for MateEngineX boot; active polling
                  # exits EARLY the moment the bridge answers, so a fast
                  # boot only costs what it actually took
BRIDGE_POLL_INTERVAL = 1.0
BRIDGE_SETTLE = 2.0  # grace after first ping: VRM/scene still loading
TAKEOVER_RETRIES = 15
TAKEOVER_DELAY = 2.0
HEALTH_INTERVAL = 10.0  # seconds between bridge health pings


def _engine_process_running() -> bool:
    """True if a MateEngineX.exe process exists (any owner)."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq MateEngineX.exe", "/NH"],
            capture_output=True, text=True, timeout=10,
        )
        return "MateEngineX.exe" in (out.stdout or "")
    except Exception:  # noqa: BLE001
        return False


class MateEngineManager:
    def __init__(self, exe_path: str | None = None,
                 boot_wait: float = BOOT_WAIT):
        self.exe_path = exe_path or os.environ.get("MATE_ENGINE_EXE", DEFAULT_EXE)
        self.boot_wait = boot_wait
        self._proc: subprocess.Popen | None = None
        self._owned = False  # True only if WE launched it
        self._status = "not started"
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._health_thread: threading.Thread | None = None
        self._health_stop = threading.Event()

    # -- public API --------------------------------------------------------

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def owned(self) -> bool:
        return self._owned

    def start_async(self) -> None:
        """Start/attach in a background thread; never blocks the caller."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="mate-engine")
        self._thread.start()

    def stop(self) -> None:
        """Terminate the engine only if we launched it."""
        self._health_stop.set()
        if self._health_thread is not None and self._health_thread.is_alive():
            self._health_thread.join(timeout=3.0)
        self._health_thread = None
        with self._lock:
            proc = self._proc
            owned = self._owned
        if owned and proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                with contextlib.suppress(Exception):
                    proc.kill()
        with self._lock:
            self._proc = None
            self._owned = False
            self._status = "stopped"

    def restart_async(self) -> None:
        """Stop (if owned) then start fresh in a background thread."""
        self.stop()
        self._health_stop.clear()
        # If it wasn't ours, the process may still be running; leave it.
        self.start_async()

    # -- internals ---------------------------------------------------------

    def _set_status(self, text: str) -> None:
        with self._lock:
            self._status = text

    def _run(self) -> None:
        from desktop.boot_timeline import boot_timeline

        timeline = boot_timeline
        timeline.mark("engine_start")
        already = _engine_process_running()
        if already:
            self._set_status("attaching to running engine...")
            self._owned = False
            timeline.mark("engine_attach")
        else:
            self._set_status("starting MateEngine...")
            try:
                self._proc = subprocess.Popen(
                    [self.exe_path],
                    cwd=os.path.dirname(self.exe_path) or None,
                )
                self._owned = True
                timeline.mark("engine_spawn")
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"engine launch failed: {exc}")
                return

        # Active boot poll (replaces the old blind `time.sleep(32)`): the
        # bridge usually opens well before the model finishes loading and
        # long before 32s; polling exits at first pong, then gives the
        # scene a short settle before touching the face.
        bridge = self._wait_for_bridge(self.boot_wait)
        if bridge is not None:
            timeline.mark("bridge_up")
            time.sleep(BRIDGE_SETTLE)
        else:
            self._set_status(f"boot timeout (bridge not up in {self.boot_wait:.0f}s)")

        self._set_status("taking over face...")
        ok = self._take_over()
        timeline.mark("takeover_done" if ok else "takeover_failed")
        if ok:
            self._set_status("ready")
        else:
            self._set_status("takeover failed (bridge?)")
        # Start health polling
        self._health_stop.clear()
        self._health_thread = threading.Thread(
            target=self._health_loop, daemon=True, name="mate-engine-health")
        self._health_thread.start()

    def _wait_for_bridge(self, timeout: float):
        """Poll MtBridge until it answers ping; returns the bridge or None.

        A successful ping only means 'renderer process is up' -- the VRM
        may still be loading, so callers must not treat this as avatar
        readiness. The `timeout` is the ceiling (BOOT_WAIT); early
        success just cuts the wait. Works identically for the attach
        path (engine already running: bridge is up, first poll wins).
        """
        from desktop.mate_bridge import MateBridge

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            br = MateBridge()
            try:
                if br.connect() and br.ping():
                    return br
            except Exception:  # noqa: BLE001
                pass
            with contextlib.suppress(Exception):
                br.close()
            remaining = deadline - time.monotonic()
            log.info(
                "Waiting for MateEngine bridge... (%.0fs left)",
                max(0.0, remaining),
            )
            time.sleep(BRIDGE_POLL_INTERVAL)
        return None

    def _take_over(self) -> bool:
        # Ensure import path (desktop/ on sys.path).
        try:
            from desktop.mate_bridge import MateBridge
        except Exception:  # noqa: BLE001
            import sys
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if here not in sys.path:
                sys.path.insert(0, here)
            from desktop.mate_bridge import MateBridge

        for _ in range(TAKEOVER_RETRIES):
            try:
                br = MateBridge()
                if br.connect():
                    br.take_over_face()
                    with contextlib.suppress(Exception):
                        br.clear_mouth()
                    return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(TAKEOVER_DELAY)
        return False

    def _health_loop(self) -> None:
        try:
            from desktop.mate_bridge import MateBridge
        except Exception:
            try:
                import os
                import sys
                here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if here not in sys.path:
                    sys.path.insert(0, here)
                from desktop.mate_bridge import MateBridge
            except Exception:
                return
        # One persistent connection for the whole loop (the old version
        # built a fresh MateBridge per ping and never closed it: ~8k dead
        # sockets + "MtBridge connected" lines per day).
        br = MateBridge()
        while not self._health_stop.is_set():
            self._health_stop.wait(HEALTH_INTERVAL)
            if self._health_stop.is_set():
                break
            try:
                if not br.is_connected():
                    br.connect()
                ok = br.is_connected() and br.ping()
            except Exception:  # noqa: BLE001
                ok = False
            with self._lock:
                if ok:
                    if self._status in ("bridge down", "takeover failed (bridge?)"):
                        self._status = "ready"
                        try:
                            br.take_over_face()
                            with contextlib.suppress(Exception):
                                br.clear_mouth()
                        except Exception:  # noqa: BLE001
                            pass
                else:
                    if self._status == "ready":
                        self._status = "bridge down"
                    br.close()  # force a fresh reconnect attempt next cycle

