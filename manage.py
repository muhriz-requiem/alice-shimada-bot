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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
