"""
manage.py
==========
Command-line utility for maintenance tasks that don't require starting
the Discord bot at all -- this is the "easy updating with multiple raw
JSON files" entry point.

Usage:
    python manage.py fetch-reddit [--subreddit NAME] [--pages N]
    python manage.py update-reddit [--subreddit NAME]
    python manage.py refresh-reddit [--subreddit NAME] [--pages N]
    python manage.py reset-memory --user USER_ID
    python manage.py reset-memory --all
    python manage.py stats
    python manage.py eval-stt [--beam N] [--modes off loudnorm denoise both]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from bot.logging_setup import get_logger, setup_logging

setup_logging()
log = get_logger("manage")

import config
from memory import store as memory_store
from reddit import fetcher, updater


def _subreddit_keys(arg: str | None) -> list[str]:
    if arg:
        if arg not in config.REDDIT_SUBREDDITS:
            print(f"Unknown subreddit '{arg}'. Known: {list(config.REDDIT_SUBREDDITS)}")
            sys.exit(1)
        return [arg]
    return list(config.REDDIT_SUBREDDITS)


def cmd_fetch_reddit(args: argparse.Namespace) -> None:
    for sub in _subreddit_keys(args.subreddit):
        meta = config.REDDIT_SUBREDDITS[sub]
        count = fetcher.fetch_subreddit_raw(sub, meta["raw_dir"], pages=args.pages)
        fetcher.prune_old_raw_files(meta["raw_dir"])
        print(f"{sub}: fetched {count} raw page(s)")


def cmd_update_reddit(args: argparse.Namespace) -> None:
    for sub in _subreddit_keys(args.subreddit):
        stats = updater.build(sub)
        print(f"{sub}: {json.dumps(stats, indent=2)}")


def cmd_refresh_reddit(args: argparse.Namespace) -> None:
    cmd_fetch_reddit(args)
    cmd_update_reddit(args)


def cmd_reset_memory(args: argparse.Namespace) -> None:
    if args.all:
        confirm = input("This wipes ALL stored memory. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Cancelled.")
            return
        for user_id in memory_store.get_all_known_user_ids():
            memory_store.reset_user(user_id)
        print("All memory reset.")
    elif args.user:
        memory_store.reset_user(args.user)
        print(f"Memory reset for user {args.user}.")
    else:
        print("Specify --user USER_ID or --all")
        sys.exit(1)


def cmd_stats(args: argparse.Namespace) -> None:
    print(f"AI backend: {config.AI_BACKEND}")
    print(f"Character: {config.CHARACTER_NAME}")
    print(f"Stored messages: {memory_store.count_messages()}")
    for sub, meta in config.REDDIT_SUBREDDITS.items():
        count = 0
        if meta["posts_file"].exists():
            try:
                count = len(json.loads(meta["posts_file"].read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                count = -1
        raw_count = len(list(meta["raw_dir"].glob("*.json"))) if meta["raw_dir"].exists() else 0
        print(f"{meta['display_name']}: {count} posts ready, {raw_count} raw dump file(s)")


def cmd_eval_stt(args: argparse.Namespace) -> None:
    import stt_eval

    stt_eval.run(Path(config.STT_EVAL_DIR), args.beam, args.modes, args.candidates, args.clips)


def cmd_latency(args: argparse.Namespace) -> None:
    """Latency harness: replay a WAV turn through the real voice pipeline.

    Builds a stub bot + the genuine VoicePipeline, forces TTS readiness,
    then runs the turn end-to-end args.repeat times with REAL Groq and
    TTS/STT calls -- only the Discord playback loop is skipped (no voice
    client exists in the harness). Collects the pipeline's own
    'Voice stage timing' lines and prints p50/p95 per stage plus TTFA,
    so agent runs can measure Phase 0..3 improvements without a human
    speaking.
    """
    import asyncio
    import logging
    import re
    import statistics
    import time

    import numpy as np
    import scipy.io.wavfile as wavfile
    import scipy.signal

    from ai import get_ai_client
    from character import load_character
    from voice import readiness
    from voice.pipeline import VoicePipeline
    from voice.response import ResponseDecision

    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"ERROR: wav not found: {wav_path}")
        sys.exit(1)

    class _StubUser:
        id = int(args.user)

    class _StubBot:
        pass

    bot = _StubBot()
    bot.user = _StubUser()
    bot.character = load_character(config.CHARACTER_NAME)
    bot.ai_client = get_ai_client()

    pipeline = VoicePipeline(bot)
    # The harness never runs readiness._run(); force READY so TTS paths
    # accept work. (The GPT-SoVITS server itself must be up separately.)
    readiness._state = readiness.STATE_READY

    user_id = int(args.user)
    guild_id = 0

    async def _run() -> None:
        print(f"Load turn WAV: {wav_path.name}")
        sr, raw = wavfile.read(str(wav_path))
        if raw.dtype == np.int16:
            audio = raw.astype(np.float32) / 32768.0
        elif raw.dtype == np.int32:
            audio = raw.astype(np.float32) / 2147483648.0
        else:
            audio = np.asarray(raw, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 48000:
            audio = scipy.signal.resample(
                audio, int(len(audio) * 48000 / sr)
            ).astype(np.float32)
        print(
            f"Audio: {len(audio)/48000:.2f}s @48k | text: {args.text!r} | "
            f"live={config.VOICE_LIVE_MODE} stt={config.VOICE_STT_ENGINE} "
            f"tts={config.VOICE_TTS_ENGINE}"
        )

        timing_pat = re.compile(
            r"silence=(?P<silence>-?[0-9.]+) "
            r"stt=(?P<stt>-?[0-9.]+|-) "
            r"decision=(?P<decision>-?[0-9.]+|-) "
            r"ai_first=(?P<ai_first>-?[0-9.]+|-) "
            r"tts_first=(?P<tts_first>-?[0-9.]+|-) "
            r"ttfa=(?P<ttfa>-?[0-9.]+|-)"
        )
        stage_values: dict[str, list[float]] = {}
        ttfa_values: list[float] = []
        wall_values: list[float] = []
        # The timing handler can't know which run a log line belongs to,
        # so the loop toggles this off during warmup -- otherwise the
        # cold first response dominates TTFA's p95 forever.
        capture = {"active": False}

        class _TimingCapture(logging.Handler):
            """Pull per-run stage numbers out of the pipeline's log line."""

            def emit(self, record: logging.LogRecord) -> None:
                if not capture["active"]:
                    return
                msg = record.getMessage()
                if "Voice stage timing:" not in msg:
                    return
                m = timing_pat.search(msg.split("Voice stage timing:", 1)[1])
                if not m:
                    return
                for key, val in m.groupdict().items():
                    if val in (None, "", "-"):
                        continue
                    try:
                        value = float(val)
                    except ValueError:
                        continue
                    if key == "ttfa":
                        ttfa_values.append(value)
                    else:
                        stage_values.setdefault(key, []).append(value)

        handler = _TimingCapture()
        pl_logger = logging.getLogger("voice.pipeline")
        pl_logger.addHandler(handler)
        try:
            total = args.warmup + args.repeat
            for i in range(total):
                label = "warmup" if i < args.warmup else f"run{i - args.warmup + 1}"
                capture["active"] = label != "warmup"
                pipeline._last_reply[guild_id] = 0.0  # ignore reply cooldown
                # record_response() also stamps a cooldown on the decision
                # engine itself; without resetting it every post-warmup run
                # would be skipped ("decision=skip -- no reply measured").
                pipeline.response_engine._cooldown_until[guild_id] = 0.0
                pipeline.response_engine._last_response[guild_id] = 0.0
                t0 = time.monotonic()
                result = await pipeline.process_audio(guild_id, user_id, audio, 48000)
                if not result:
                    print(f"[{label}] no transcript (STT cold start?)")
                    continue
                text, language = result
                decision, _prio = pipeline.response_engine.should_respond(
                    guild_id, user_id, text,
                    is_owner=True, others_present=False,
                )
                if decision != ResponseDecision.RESPOND:
                    print(f"[{label}] decision={decision.value} -- no reply measured")
                    continue
                await pipeline._generate_and_respond(
                    guild_id, user_id, text, True, language,
                    t_speech_end=t0,
                    silence=0.3,
                )
                wall = time.monotonic() - t0
                if capture["active"]:  # warmup wall stays out of the stats
                    wall_values.append(wall)
                print(f"[{label}] decided=RESPOND wall={wall*1000:.0f}ms")
        finally:
            pl_logger.removeHandler(handler)

        def _pct(values: list[float], p: float) -> float:
            vals = sorted(values)
            idx = min(len(vals) - 1, int(round(p * (len(vals) - 1))))
            return vals[idx]

        print()
        if ttfa_values:
            print(f"TTFA  ({len(ttfa_values)} replies): "
                  f"p50={_pct(ttfa_values, 0.5):.2f}s p95={_pct(ttfa_values, 0.95):.2f}s "
                  f"mean={statistics.mean(ttfa_values):.2f}s")
        else:
            print("TTFA: no complete reply (TTS engine unavailable?)")
        if wall_values:
            print(f"Wall  ({len(wall_values)} replies): "
                  f"p50={_pct(wall_values, 0.5):.2f}s p95={_pct(wall_values, 0.95):.2f}s")
        if stage_values:
            print("Stage p50/p95:")
            for stage in sorted(stage_values):
                vals = stage_values[stage]
                print(f"  {stage:<10} p50={_pct(vals, 0.5):.2f}s p95={_pct(vals, 0.95):.2f}s")
        print()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="AliceShimadaBot maintenance CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch-reddit", help="Fetch new raw Reddit dumps")
    p_fetch.add_argument("--subreddit", help="Only this subreddit (default: all)")
    p_fetch.add_argument("--pages", type=int, default=None, help="Listing pages to pull")
    p_fetch.set_defaults(func=cmd_fetch_reddit)

    p_update = sub.add_parser("update-reddit", help="Rebuild posts.json from raw dumps")
    p_update.add_argument("--subreddit", help="Only this subreddit (default: all)")
    p_update.set_defaults(func=cmd_update_reddit)

    p_refresh = sub.add_parser("refresh-reddit", help="Fetch, then rebuild")
    p_refresh.add_argument("--subreddit", help="Only this subreddit (default: all)")
    p_refresh.add_argument("--pages", type=int, default=None)
    p_refresh.set_defaults(func=cmd_refresh_reddit)

    p_reset = sub.add_parser("reset-memory", help="Wipe stored memory")
    p_reset.add_argument("--user", help="Discord user ID to reset")
    p_reset.add_argument("--all", action="store_true", help="Reset every user (asks to confirm)")
    p_reset.set_defaults(func=cmd_reset_memory)

    p_stats = sub.add_parser("stats", help="Show quick stats")
    p_stats.set_defaults(func=cmd_stats)

    p_eval = sub.add_parser("eval-stt", help="Offline STT evaluation on captured clips")
    p_eval.add_argument("--beam", type=int, default=3, help="Whisper beam size")
    p_eval.add_argument("--modes", nargs="+", default=["off", "loudnorm", "denoise", "both"],
                        choices=["off", "loudnorm", "denoise", "both"])
    p_eval.add_argument("--candidates", nargs="+", default=None,
                        help="Subset of candidates to run")
    p_eval.add_argument("--clips", nargs="+", default=None,
                        help="Subset of clip stems to score")
    p_eval.set_defaults(func=cmd_eval_stt)

    p_lat = sub.add_parser("latency", help="Live-voice latency harness (real AI/TTS calls)")
    p_lat.add_argument("--wav", required=True, help="Turn audio WAV to replay through the pipeline")
    p_lat.add_argument("--text", default="Alice, did you eat lunch today?",
                       help="Hint text ignored by STT (documented intent of the clip)")
    p_lat.add_argument("--repeat", type=int, default=5, help="Timed runs after warmup")
    p_lat.add_argument("--warmup", type=int, default=1, help="Warmup runs excluded from stats")
    p_lat.add_argument("--user", default="100000000000000000",
                       help="Fake Discord user id for the harness")
    p_lat.set_defaults(func=cmd_latency)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
