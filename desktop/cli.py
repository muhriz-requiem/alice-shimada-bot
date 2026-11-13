"""
desktop/cli.py
==============
Observer CLI debug runner for the desktop companion (Phase 1; Phase 2
moved the shared engine into desktop/voice_app.py so the tray app and
this runner boot the identical session code path).

Modes
-----
1. Mic mode (default): listens on the microphone with the exact
   Discord path (AliceVoiceSink + Silero VAD + monitor loop) and plays
   Alice's SoVITS voice through the speakers.

2. Self-test mode (--inject wav): replays a WAV as if the owner spoke
   it, through the same code path -- all models (STT/Groq/SoVITS) stay
   real; only the mic is replaced. Use this to sanity check the loop
   without making a sound.

Run:
    python -m desktop.cli              # live mic loop
    python -m desktop.cli --list-devices
    python -m desktop.cli --device 1
    python -m desktop.cli --inject turn.wav
    python -m desktop.cli --no-speaker # silent self-test for CI
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import sounddevice as sd

from dotenv import load_dotenv

load_dotenv()

from bot.logging_setup import get_logger, setup_logging

setup_logging()
log = get_logger("desktop.cli")

import config
from desktop.voice_app import (
    boot_companion,
    inject_turn,
    shutdown_companion,
)


def _list_devices() -> None:
    devices = sd.query_devices()
    default_in, default_out = sd.default.device
    print("Available audio devices (default marked with *):")
    for i, info in enumerate(devices):
        mark = " "
        if i in (default_in, default_out):
            mark = "*"
        n_in = int(info.get("max_input_channels") or 0)
        n_out = int(info.get("max_output_channels") or 0)
        if not n_in and not n_out:
            continue
        kinds = []
        if n_in:
            kinds.append(f"in({n_in})")
        if n_out:
            kinds.append(f"out({n_out})")
        print(f"{mark} [{i}] {info.get('name', '?')} ({','.join(kinds)})")


def _parse_args(argv: Optional[list[str]] = None):
    import argparse

    p = argparse.ArgumentParser(
        prog="desktop companion",
        description="Alice desktop companion - headless voice loop",
    )
    p.add_argument("--device", type=int, default=None,
                   help="input device index (see --list-devices)")
    p.add_argument("--owner-id", type=int, default=None,
                   help="user id the mic feed is attributed to "
                        "(default OWNER_IDS[0])")
    p.add_argument("--guild-id", type=int, default=0,
                   help="desktop session key (0)")
    p.add_argument("--inject", type=str, default=None,
                   help="replay this WAV as the owner speaking (self-test)")
    p.add_argument("--loop", action="store_true",
                   help="keep the session alive after an injected turn")
    p.add_argument("--no-speaker", action="store_true",
                   help="discard playback audio (silent self-test)")
    p.add_argument("--diarize", action="store_true",
                   help="run per-utterance diarization on desktop turns "
                        "(off by default: one mic = one speaker)")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args(argv)


async def _run(args) -> None:
    from desktop.boot_timeline import boot_timeline

    boot_timeline.mark("cli_start")
    owner_id = args.owner_id or (
        next(iter(config.OWNER_IDS)) if config.OWNER_IDS else 12345
    )
    guild_id = args.guild_id

    booted = await boot_companion(
        owner_id=owner_id,
        guild_id=guild_id,
        device=args.device,
        silent=args.no_speaker,
        diarize=args.diarize,
        on_barge=lambda: print("(barge-in: raised voice detected)"),
        # Phase 1 semantics: --inject is a DETERMINISTIC self-test -- the
        # real microphone must not feed room noise mid-test. It only
        # starts for the mic loop (or after the injected turn with --loop).
        start_mic=not args.inject,
    )
    try:
        if args.inject:
            await inject_turn(booted, args.inject)
            print("Turn complete.")
            if not args.loop:
                # Give a beat for trailing log lines.
                await asyncio.sleep(1.0)
                return
            print("--loop given: idling; Ctrl+C to exit.")
            booted.mic.start()

        print()
        print("Desktop companion READY - speak to Alice. Ctrl+C to quit.")
        print()
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await shutdown_companion(booted)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    if args.list_devices:
        _list_devices()
        return
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nExit.")
    except config.ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
