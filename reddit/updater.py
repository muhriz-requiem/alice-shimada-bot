"""
reddit/updater.py
====================
Builds each subreddit's clean posts.json from all of its raw dump
files: filters out NSFW/text/gallery posts, extracts a usable image
URL, deduplicates by post URL, and sorts by score. Ported from the
original reddit_updater.py essentially unchanged -- this logic was
already solid -- generalized to loop over config.REDDIT_SUBREDDITS
instead of two hardcoded subreddit names, so adding a subreddit to the
config is all "easy updating" requires here.
"""

from __future__ import annotations

import json
from pathlib import Path

import config
from bot.logging_setup import get_logger

log = get_logger("reddit.updater")


def extract_image_post(post: dict) -> dict | None:
    data = post.get("data", {})

    if data.get("over_18"):
        return None
    if data.get("is_self"):
        return None
    if data.get("is_gallery"):
        return None

    url = data.get("url_overridden_by_dest") or data.get("url") or ""
    if not url:
        return None

    is_reddit_hosted = "i.redd.it" in url or "preview.redd.it" in url
    is_image_ext = any(url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"))

    if not (is_reddit_hosted or is_image_ext):
        return None

    return {
        "title": data.get("title"),
        "url": f"https://reddit.com{data.get('permalink')}",
        "image": url.split("?")[0],
        "score": data.get("score", 0),
    }


def build(subreddit: str) -> dict:
    """Rebuilds one subreddit's posts.json from all its raw dump files."""
    meta = config.REDDIT_SUBREDDITS.get(subreddit)
    if not meta:
        log.error(f"Unknown subreddit key: {subreddit}")
        return {}

    raw_dir: Path = meta["raw_dir"]
    output_file: Path = meta["posts_file"]

    if not raw_dir.exists():
        log.warning(f"No raw folder for {subreddit} at {raw_dir}; nothing to build.")
        return {}

    cleaned: list[dict] = []
    seen: set[str] = set()

    stats = {"files_scanned": 0, "raw_posts": 0, "duplicates": 0, "skipped": 0}

    for raw_file in sorted(raw_dir.glob("*.json")):
        stats["files_scanned"] += 1
        try:
            with raw_file.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            posts = raw["data"]["children"]
        except (json.JSONDecodeError, KeyError, OSError) as e:
            log.error(f"Skipping unreadable raw file {raw_file.name}: {e}")
            continue

        stats["raw_posts"] += len(posts)

        for p in posts:
            post = extract_image_post(p)
            if not post:
                stats["skipped"] += 1
                continue

            key = post["url"]
            if key in seen:
                stats["duplicates"] += 1
                continue

            seen.add(key)
            cleaned.append(post)

    cleaned.sort(key=lambda x: x["score"], reverse=True)

    try:
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
    except OSError as e:
        log.error(f"Failed to write {output_file}: {e}")
        return stats

    stats["final_posts"] = len(cleaned)
    log.info(
        f"[{subreddit}] scanned={stats['files_scanned']} raw={stats['raw_posts']} "
        f"dupes={stats['duplicates']} skipped={stats['skipped']} final={stats['final_posts']}"
    )
    return stats


def build_all() -> dict[str, dict]:
    return {sub: build(sub) for sub in config.REDDIT_SUBREDDITS}


if __name__ == "__main__":
    build_all()
