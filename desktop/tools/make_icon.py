"""
desktop/tools/make_icon.py
==========================
One-off asset generator: writes `assets/alice.ico` for the tray icon
and the desktop shortcut, from `desktop.app._fallback_image()` so the
tray fallback and the shipped asset can never look different.

Run:    venv\\Scripts\\python.exe -m desktop.tools.make_icon
Output: D:\\AliceShimadaBot\\assets\\alice.ico
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

SIZES = [16, 24, 32, 48, 64, 128, 256]


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "assets"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "alice.ico"

    base = _fallback_rgb()

    # Same art in RGBA for .ico sizes.
    image = Image.frombytes("RGBA", (256, 256), base)
    image.save(out, format="ICO", sizes=[(s, s) for s in SIZES])
    print(f"Wrote {out} ({out.stat().st_size} bytes, sizes={SIZES})")


def _fallback_rgb() -> bytes:
    """Render the app's fallback art and return raw RGBA bytes."""
    import desktop.app as app

    img = app._fallback_image()
    return img.tobytes()


if __name__ == "__main__":
    main()
