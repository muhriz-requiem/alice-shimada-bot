"""
voice/utils.py
===============
Shared utilities for the voice package.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# Common ffmpeg locations on Windows
_FFMPEG_PATHS = [
    Path(r"C:\ffmpeg\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe"),
    Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
    Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
]


def find_ffmpeg() -> str:
    """
    Find ffmpeg executable, checking PATH first then common locations.
    Returns the path string to use in subprocess calls.
    """
    # Check PATH first
    path = shutil.which("ffmpeg")
    if path:
        return path

    # Check common locations
    for p in _FFMPEG_PATHS:
        if p.exists():
            return str(p)

    # Fallback to just 'ffmpeg' and hope it's on PATH
    return "ffmpeg"
