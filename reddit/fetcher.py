"""
reddit/fetcher.py
====================
Fetches raw Reddit listing pages and writes them as numbered JSON dump
files under each subreddit's raw_dir, in the same
{"data": {"children": [...]}} shape Reddit's own listing API returns --
which is also exactly the shape reddit/updater.py already expects, so
no format translation is needed.

This is the piece that was actually missing from the original project:
reddit_updater.py could build posts.json from raw dumps, but nothing
in the codebase produced those dumps -- a leftover Chrome browser
profile in the project folder suggested they'd been collected by hand.
This fetcher replaces that with a plain HTTP call to Reddit's public,
no-auth-required JSON listing endpoint (the same endpoint your browser
hits when you add ".json" to a subreddit URL), paginated via Reddit's
"after" cursor.

No Reddit API credentials are required for this. If you later want to
switch to PRAW (Reddit's official API wrapper) for higher rate limits
or access to auth-only listings, that would be a self-contained
alternative implementation of `fetch_subreddit_raw` -- the rest of the
pipeline (updater, rotation) wouldn't need to change.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

import config
from bot.logging_setup import get_logger

log = get_logger("reddit.fetcher")

# Use a more descriptive User-Agent to avoid 403s
_HEADERS = {
    "User-Agent": f"{config.REDDIT_USER_AGENT} (by /u/{getattr(config, 'REDDIT_USERNAME', 'unknown')})"
}


def _fetch_page(subreddit: str, sort: str, time_range: str, after: str | None) -> dict | None:
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    params: dict[str, str | int] = {"limit": 100}
    if sort == "top":
        params["t"] = time_range
    if after:
        params["after"] = after

    max_retries = 3
    backoff = 2  # seconds

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=_HEADERS, params=params, timeout=15)
            if resp.status_code == 429:  # Rate limited
                retry_after = int(resp.headers.get("Retry-After", backoff))
                log.warning(f"Rate limited fetching r/{subreddit}, waiting {retry_after}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                log.error(f"Failed to fetch r/{subreddit} ({sort}, after={after}) after {max_retries} attempts: {e}")
                return None
            log.warning(f"Fetch failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {backoff}s...")
            time.sleep(backoff)
            backoff *= 2

    return None


# api.pullpush.io is a community-run archive of Reddit submissions that
# serves the same post objects as the real API -- no auth, no OAuth
# approval queue (Reddit ended self-serve app creation in Nov 2025 and
# 403s unauthenticated www.reddit.com JSON from most networks).
PULLPUSH_URL = "https://api.pullpush.io/reddit/search/submission/"
_PAGE_SIZE = 100

# Epoch windows for PullPush's `after` param, keyed by the same
# time-range names REDDIT_FETCH_TIME_RANGE already uses. The archive
# has no "top of month" ranking; the updater re-sorts everything by
# score anyway, so fetching the newest posts in the window is enough.
_PULLPUSH_WINDOWS = {
    "hour": 3600,
    "day": 86400,
    "week": 604800,
    "month": 2592000,
    "year": 31536000,
}


def _fetch_page_pullpush(
    subreddit: str, before: float | None = None, window_ts: int | None = None
) -> dict | None:
    """One page from the PullPush archive, wrapped in Reddit's listing shape.

    `before` is the created_utc cursor from the previous page's last
    post -- PullPush pages backwards with an inclusive epoch bound, so
    consecutive pages can overlap by one post (the caller dedupes).
    `window_ts` is the resolved lower bound (now - time_range) for the
    run; the caller never sends it together with a cursor that has
    already passed it (after >= before is an HTTP 400 on PullPush).
    """
    params: dict = {"subreddit": subreddit, "size": _PAGE_SIZE}
    if window_ts is not None:
        params["after"] = int(window_ts)
    if before is not None:
        params["before"] = int(before)

    max_retries = 3
    backoff = 2

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                PULLPUSH_URL, headers=_HEADERS, params=params, timeout=15
            )
            if resp.status_code == 429:  # Rate limited
                retry_after = int(resp.headers.get("Retry-After") or backoff)
                log.warning(
                    f"Rate limited by PullPush fetching r/{subreddit}, "
                    f"waiting {retry_after}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_after)
                backoff = max(backoff * 2, retry_after)
                continue
            resp.raise_for_status()
            posts = resp.json().get("data") or []
            # Re-wrap into Reddit's listing shape
            # ({"data": {"children": [{"kind": "t3", "data": post}]}})
            # so raw dumps, updater.extract_image_post, and rotation
            # all stay byte-compatible with the legacy source.
            children = [{"kind": "t3", "data": p} for p in posts]
            return {"data": {"children": children, "after": None}}
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                log.error(
                    f"Failed to fetch r/{subreddit} from PullPush "
                    f"(before={before}) after {max_retries} attempts: {e}"
                )
                return None
            log.warning(
                f"PullPush fetch failed (attempt {attempt + 1}/{max_retries}): "
                f"{e}, retrying in {backoff}s..."
            )
            time.sleep(backoff)
            backoff *= 2

    return None


def fetch_subreddit_raw(
    subreddit: str,
    raw_dir: Path,
    pages: int | None = None,
    sort: str | None = None,
    time_range: str | None = None,
) -> int:
    """
    Pulls up to `pages` listing pages (~100 posts each) from a
    subreddit and writes each as its own timestamped raw dump file.
    Returns the number of pages successfully fetched.
    """
    pages = pages or config.REDDIT_FETCH_PAGES
    sort = sort or config.REDDIT_FETCH_SORT
    time_range = time_range or config.REDDIT_FETCH_TIME_RANGE
    source = (getattr(config, "REDDIT_SOURCE", "pullpush") or "pullpush").strip().lower()

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the time window once for the whole run. Recomputing it
    # per page would drift forward mid-pagination, and pairing a stale
    # window bound with an older cursor is an HTTP 400 on PullPush.
    window_ts = None
    if source != "reddit":
        _window = _PULLPUSH_WINDOWS.get((time_range or "month").strip().lower())
        if _window is not None:
            window_ts = int(time.time()) - _window

    cursor = None
    fetched = 0
    run_stamp = int(time.time())
    seen_ids: set[str] = set()

    for page_num in range(1, pages + 1):
        if source == "reddit":
            data = _fetch_page(subreddit, sort, time_range, cursor)
        else:
            if cursor is not None and window_ts is not None and cursor <= window_ts:
                log.info(
                    f"r/{subreddit}: reached the {time_range} window "
                    f"boundary at page {page_num}, stopping."
                )
                break
            data = _fetch_page_pullpush(subreddit, before=cursor, window_ts=window_ts)
            if (
                data is not None
                and not data["data"].get("children")
                and cursor is None
                and window_ts is not None
            ):
                # Sparse/dormant subreddit: nothing inside the window.
                # Drop it for the rest of the run so the pool still gets
                # fed (updater dedupes by URL; rotation skips already-
                # shown posts, so old posts are safe).
                log.info(
                    f"r/{subreddit}: no posts in the last {time_range}; "
                    "fetching full archive instead."
                )
                window_ts = None
                data = _fetch_page_pullpush(subreddit, before=None, window_ts=None)
        if data is None:
            break

        children = data.get("data", {}).get("children", [])

        # PullPush pages backwards with an inclusive `before` bound, so
        # consecutive pages overlap (at minimum the boundary post).
        # Advance the cursor from the RAW page for guaranteed progress,
        # then drop anything already fetched this run.
        exhausted = False
        if source != "reddit":
            cursor = (
                children[-1].get("data", {}).get("created_utc") if children else None
            )
            exhausted = len(children) < _PAGE_SIZE
            children = [
                c for c in children
                if c.get("data", {}).get("id") not in seen_ids
            ]
            for c in children:
                seen_ids.add(c.get("data", {}).get("id"))
            data["data"]["children"] = children

        if not children:
            log.info(f"r/{subreddit}: no more new posts at page {page_num}, stopping early.")
            break

        out_path = raw_dir / f"raw_{run_stamp}_{page_num}.json"
        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except OSError as e:
            log.error(f"Failed to write {out_path}: {e}")
            break

        fetched += 1

        if source == "reddit":
            cursor = data.get("data", {}).get("after")
            if not cursor:
                log.info(f"r/{subreddit}: reached end of listing at page {page_num}.")
                break
        elif exhausted or cursor is None:
            log.info(f"r/{subreddit}: archive exhausted at page {page_num}.")
            break

        # Be polite between pages. PullPush rate-limits at 15 req/min
        # (soft) / 30 (hard) and documents ~4s per request as the pace
        # that stays under the soft limit; the legacy Reddit endpoint
        # keeps its original 2s.
        if page_num < pages:
            time.sleep(4 if source != "reddit" else 2)

    log.info(f"r/{subreddit}: fetched {fetched} page(s) -> {raw_dir}")
    return fetched


def prune_old_raw_files(raw_dir: Path, keep: int | None = None) -> int:
    """
    Deletes the oldest raw dump files beyond the retention count, to
    keep disk usage bounded on small hosts. Returns number deleted.
    """
    keep = keep if keep is not None else config.REDDIT_RAW_RETENTION_FILES

    files = sorted(raw_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    to_delete = files[keep:]

    deleted = 0
    for f in to_delete:
        try:
            f.unlink()
            deleted += 1
        except OSError as e:
            log.error(f"Failed to delete old raw file {f}: {e}")

    if deleted:
        log.info(f"Pruned {deleted} old raw file(s) from {raw_dir}")

    return deleted


def fetch_all() -> None:
    """Fetches and prunes every subreddit in config.REDDIT_SUBREDDITS."""
    if getattr(config, "REDDIT_FETCH_PARALLEL", True):
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for subreddit, meta in config.REDDIT_SUBREDDITS.items():
                futures.append(executor.submit(fetch_subreddit_raw, subreddit, meta["raw_dir"]))
                futures.append(executor.submit(prune_old_raw_files, meta["raw_dir"]))
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log.error(f"Parallel fetch task failed: {e}")
    else:
        for subreddit, meta in config.REDDIT_SUBREDDITS.items():
            fetch_subreddit_raw(subreddit, meta["raw_dir"])
            prune_old_raw_files(meta["raw_dir"])
