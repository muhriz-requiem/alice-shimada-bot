"""
reddit/rotation.py
=====================
Hands out a random post per subreddit without repeating one until the
whole pool has been shown (a shuffle-bag), same idea as the original
reddit_fetcher.py's UNSENT/USED pools. The one change: the remaining
queue is persisted to disk (rotation_state.json) after every draw, so a
bot restart doesn't immediately start repeating posts you've already
seen in the current session.

The queue stores post URLs (not full post dicts) so it stays valid
even after posts.json gets rebuilt by the updater -- on each draw we
look the URL up against the *current* posts.json and silently refill
if the queue and the current pool have drifted apart (e.g. the updater
removed a post that's still queued).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import config
from bot.logging_setup import get_logger

log = get_logger("reddit.rotation")


def _load_posts(posts_file: Path) -> list[dict]:
    if not posts_file.exists():
        return []
    try:
        with posts_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.error(f"Failed to load {posts_file}: {e}")
        return []


def _load_state() -> dict:
    if not config.REDDIT_ROTATION_STATE_PATH.exists():
        return {}
    try:
        with config.REDDIT_ROTATION_STATE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Rotation state unreadable ({e}); starting fresh.")
        return {}


def _save_state(state: dict) -> None:
    try:
        with config.REDDIT_ROTATION_STATE_PATH.open("w", encoding="utf-8") as f:
            json.dump(state, f)
    except OSError as e:
        log.error(f"Failed to save rotation state: {e}")


def get_random_post(subreddit: str) -> dict | None:
    meta = config.REDDIT_SUBREDDITS.get(subreddit)
    if not meta:
        log.error(f"Unknown subreddit key for rotation: {subreddit}")
        return None

    posts = _load_posts(meta["posts_file"])
    if not posts:
        log.warning(f"No posts available for r/{subreddit} yet.")
        return None

    posts_by_url = {p["url"]: p for p in posts}
    state = _load_state()
    queue = state.get(subreddit, [])

    # Refill if empty, or if the queue has drifted too far from the
    # current pool (e.g. after the updater rebuilt posts.json).
    valid_queue = [url for url in queue if url in posts_by_url]
    if not valid_queue:
        valid_queue = list(posts_by_url.keys())
        random.shuffle(valid_queue)
        log.info(f"Refilled rotation pool for r/{subreddit}: {len(valid_queue)} posts")

    chosen_url = valid_queue.pop()
    state[subreddit] = valid_queue
    _save_state(state)

    return posts_by_url[chosen_url]
