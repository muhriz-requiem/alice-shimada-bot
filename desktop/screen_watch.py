"""
desktop/screen_watch.py
=======================
Screen capture for the Glance feature. Privacy contract:

- Captures happen ONLY when the pipeline's look hook asks (explicit
  utterance / ambient governor) AND the tray permission is on.
- The foreground window title is read first; denylisted titles are
  refused BEFORE anything is sent anywhere.
- Frames live in RAM only and are discarded after the vision request;
  nothing is written to disk.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import io
import sys
import time
from dataclasses import dataclass

import numpy as np

import config
from bot.logging_setup import get_logger

log = get_logger("desktop.screen_watch")

_user32: ctypes.CDLL | None = None
if sys.platform == "win32":  # None on non-Windows (feature off)
    _user32 = ctypes.windll.user32


@dataclass
class GlanceFrame:
    """One RAM-only screenshot: JPEG bytes + context. Never persisted."""

    jpeg: bytes
    width: int
    height: int
    title: str
    taken_at: float


def foreground_title() -> str:
    """Title of the foreground window ("" when unavailable)."""
    if _user32 is None:
        return ""
    try:
        hwnd = _user32.GetForegroundWindow()
        if not hwnd:
            return ""
        length = _user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value or ""
    except Exception:  # noqa: BLE001 - best-effort context
        return ""


def denylisted(title: str) -> bool:
    """True when the foreground window matches the privacy denylist."""
    t = (title or "").lower()
    if not t:
        return False
    for raw in config.SCREEN_LOOK_DENYLIST:
        needle = raw.strip().lower()
        if needle and needle in t:
            return True
    return False


def _capture_image(monitor_override: str | None = None):
    """PIL Image of the configured monitor (primary by default)."""
    from PIL import ImageGrab

    monitor = (monitor_override or config.SCREEN_LOOK_MONITOR or "primary")
    monitor = monitor.strip().lower()
    if monitor == "all":
        return ImageGrab.grab(all_screens=True)
    if monitor not in ("primary", "") and monitor.isdigit():
        # bbox of monitor N via ctypes EnumDisplayMonitors
        boxes: list[tuple[int, int, int, int]] = []

        def _cb(_hdc, _hmonitor, lptp, _lpdata, _dw):
            rect = ctypes.cast(lptp, ctypes.POINTER(ctypes.wintypes.RECT)).contents
            boxes.append((rect.left, rect.top, rect.right, rect.bottom))
            return True

        MONITORENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.wintypes.RECT),
            ctypes.c_double,
        )
        try:
            _user32.EnumDisplayMonitors(0, None, MONITORENUMPROC(_cb), 0)
        except Exception:  # noqa: BLE001
            boxes = []
        idx = int(monitor)
        if 0 <= idx < len(boxes):
            return ImageGrab.grab(bbox=boxes[idx])
    return ImageGrab.grab()


def _downscale_jpeg(img, max_width: int, quality: int) -> tuple[bytes, int, int]:
    w, h = img.size
    if w > max_width:
        nh = max(1, int(h * max_width / w))
        img = img.resize((max_width, nh))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue(), img.size[0], img.size[1]


def capture(monitor_override: str | None = None) -> GlanceFrame | None:
    """One privacy-checked screenshot, or None when refused/off."""
    if not (config.SCREEN_LOOK_ENABLED or config.SCREEN_LOOK_AMBIENT):
        return None
    title = foreground_title()
    if denylisted(title):
        log.info(f"Glance refused (denylist): {title[:60]!r}")
        return None
    try:
        img = _capture_image(monitor_override)
        jpeg, w, h = _downscale_jpeg(
            img, config.SCREEN_LOOK_MAX_WIDTH, config.SCREEN_LOOK_JPEG_QUALITY
        )
    except Exception as e:  # noqa: BLE001
        log.warning(f"Glance capture failed: {e}")
        return None
    return GlanceFrame(jpeg=jpeg, width=w, height=h, title=title, taken_at=time.time())


def change_fraction(prev: np.ndarray | None, cur: np.ndarray | None) -> float:
    """Fraction of meaningfully-changed pixels between two thumbnails.

    Both inputs are float32 grayscale arrays of the same shape; 0.0
    means identical. Used by the ambient governor so she only comments
    when the screen actually changed.
    """
    if prev is None or cur is None:
        return 1.0
    if prev.shape != cur.shape:
        return 1.0
    return float(np.mean(np.abs(cur - prev) > 0.06))


def thumbnail(img) -> np.ndarray:
    """Small grayscale fingerprint for change detection (RAM only)."""
    thumb = img.convert("L").resize((64, 36))
    return np.asarray(thumb, dtype=np.float32) / 255.0
