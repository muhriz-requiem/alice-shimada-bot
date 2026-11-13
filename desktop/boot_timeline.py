"""desktop/boot_timeline.py
Cold-start stage timing for the desktop companion.

A process-wide singleton that records the first-seen timestamp of named
boot stages (engine_spawn, bridge_up, takeover_done, sovits_up,
stt_ready, session_ready, ...) and renders one summary line with
per-stage deltas.

Why: before this existed, cold-start behavior was anecdotal ("the tray
takes ~a minute"). With marks wired into app.py / engine_manager.py /
prewarm.py / voice_app.py, every real launch produces a
"Desktop boot (...)" log line that names exactly which stage ate the
time -- measurable, comparable across launches.

Enabled by default. Set DESKTOP_BOOT_TIMELINE=0 to silence (marks
become no-ops; zero overhead beyond a dict check).
"""
from __future__ import annotations

import os
import threading
import time

from bot.logging_setup import get_logger

log = get_logger("desktop.boot")

_ENABLED = os.getenv(
    "DESKTOP_BOOT_TIMELINE", "1"
).strip().lower() not in ("0", "false", "no", "off")


class BootTimeline:
    """First-seen stage recorder. Thread-safe; duplicate labels are no-ops."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._marks: list[tuple[str, float]] = []
        self._lock = threading.Lock()

    def mark(self, label: str) -> float | None:
        """Record `label` at boot-relative now (first occurrence only)."""
        if not _ENABLED:
            return None
        now = time.perf_counter()
        with self._lock:
            if any(existing == label for existing, _ in self._marks):
                return None
            offset = now - self._t0
            self._marks.append((label, offset))
        log.info("boot: %-24s @ %7.2fs", label, offset)
        return offset

    def span(self, start_label: str, end_label: str) -> float | None:
        """Seconds between the first-seen of two labels, if both exist."""
        with self._lock:
            first = dict(self._marks)
        if start_label in first and end_label in first:
            return first[end_label] - first[start_label]
        return None

    def log_summary(self, context: str) -> None:
        """One INFO line: total wall time + per-stage progression."""
        if not _ENABLED:
            return
        with self._lock:
            marks = list(self._marks)
        wall = time.perf_counter() - self._t0
        if not marks:
            log.info(
                "Desktop boot (%s): no stages recorded (%.1fs elapsed)",
                context, wall,
            )
            return
        prev = 0.0
        parts = []
        for label, off in marks:
            parts.append(f"{label} @{off:.1f}s ({off - prev:+.1f})")
            prev = off
        log.info(
            "Desktop boot (%s): %.1fs total | %s",
            context, wall, " | ".join(parts),
        )


# Process-wide singleton: every desktop module hand in marks.
boot_timeline = BootTimeline()
