"""
tests/test_reddit_source.py
============================
Covers the REDDIT_SOURCE switch added after Reddit's Nov 2025 policy
change (self-serve API app creation ended; unauthenticated
www.reddit.com JSON returns 403 from most networks):

- PullPush pages are re-wrapped into Reddit's exact listing shape so
  updater/rotation/raw-dump format stay untouched
- pagination cursor = previous page's last created_utc (inclusive
  bound), with in-run dedupe of the overlapping boundary post
- time window follows REDDIT_FETCH_TIME_RANGE
- source="reddit" keeps the legacy www.reddit.com path
- fetch failure returns 0 and writes nothing (cache path unaffected)
- an end-to-end updater.build() consumes a PullPush dump correctly
"""

from __future__ import annotations

import json
import time

import pytest
import requests

import config
from reddit import fetcher

NOW = int(time.time())
DAY = 86400


def _post(i: int, created_utc: float, sub: str = "testsub") -> dict:
    return {
        "id": f"post{i}",
        "title": f"Title {i}",
        "selftext": "",
        "score": i,
        "created_utc": created_utc,
        "author": "someone",
        "subreddit": sub,
        "over_18": False,
        "is_self": False,
        "is_gallery": False,
        "url": f"https://i.redd.it/x{i}.jpg",
        "permalink": f"/r/{sub}/comments/post{i}",
    }


class FakeResp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200
        self.headers: dict = {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        pass


class FakeGet:
    """Scripted stand-in for requests.get, recording call params."""

    def __init__(self, effects: list) -> None:
        self.calls: list[dict] = []
        self._effects = list(effects)

    def __call__(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        effect = self._effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect if isinstance(effect, FakeResp) else FakeResp(effect)


def _pullpush_payload(posts: list[dict]) -> dict:
    return {"data": posts}


def _reddit_payload(posts: list[dict], after: str | None = None) -> dict:
    return {
        "data": {
            "children": [{"kind": "t3", "data": p} for p in posts],
            "after": after,
        }
    }


def _run(monkeypatch, effects, tmp_path, source="pullpush", pages=3, time_range="month"):
    fake = FakeGet(effects)
    monkeypatch.setattr(fetcher.requests, "get", fake)
    monkeypatch.setattr(fetcher.time, "sleep", lambda s: None)
    monkeypatch.setattr(config, "REDDIT_SOURCE", source)
    fetched = fetcher.fetch_subreddit_raw(
        "testsub", tmp_path, pages=pages, sort="top", time_range=time_range
    )
    return fetched, fake


def _dump_files(tmp_path):
    return sorted(tmp_path.glob("raw_*.json"))


# ---------------------------------------------------------------------
# PullPush source
# ---------------------------------------------------------------------


def test_pullpush_wraps_reddit_shape(tmp_path, monkeypatch):
    posts = [_post(1, 1700000000), _post(2, 1700000100)]
    fetched, fake = _run(monkeypatch, [_pullpush_payload(posts)], tmp_path)

    assert fetched == 1
    assert fake.calls[0]["url"] == fetcher.PULLPUSH_URL
    params = fake.calls[0]["params"]
    assert params["subreddit"] == "testsub"
    assert params["size"] == 100
    # month window -> after = now - 30d
    assert abs(params["after"] - (int(time.time()) - 2592000)) < 5
    assert "before" not in params

    files = _dump_files(tmp_path)
    assert len(files) == 1
    with files[0].open(encoding="utf-8") as f:
        raw = json.load(f)
    children = raw["data"]["children"]
    assert len(children) == 2
    assert children[0]["kind"] == "t3"
    assert children[0]["data"]["title"] == "Title 1"
    assert children[0]["data"]["url"] == "https://i.redd.it/x1.jpg"


def test_pullpush_time_range_window(tmp_path, monkeypatch):
    posts = [_post(1, 1700000000)]
    _, fake = _run(monkeypatch, [_pullpush_payload(posts)], tmp_path, time_range="week")
    assert abs(fake.calls[0]["params"]["after"] - (int(time.time()) - 604800)) < 5


def test_pullpush_pagination_cursor_and_exhaust(tmp_path, monkeypatch):
    # Posts must sit inside the month window: the caller stops paging
    # once the cursor walks out of it (after >= before = HTTP 400).
    page1 = [_post(i, NOW - 6 * DAY + i * 60) for i in range(100)]
    page2 = [_post(100 + i, NOW - 7 * DAY + i * 60) for i in range(5)]
    fetched, fake = _run(
        monkeypatch, [_pullpush_payload(page1), _pullpush_payload(page2)], tmp_path
    )

    assert fetched == 2
    assert len(fake.calls) == 2
    # Second page pages backwards from the first page's last post.
    assert fake.calls[1]["params"]["before"] == int(page1[-1]["created_utc"])
    # A short page means the archive is exhausted: no third call.
    assert len(_dump_files(tmp_path)) == 2


def test_pullpush_stops_at_window_boundary(tmp_path, monkeypatch):
    # Page 1 already reaches past the window (cursor older than
    # `after`): the next page would be after >= before -> PullPush 400.
    # The caller must stop instead of issuing that request.
    page1 = [_post(i, NOW - 40 * DAY + i * 60) for i in range(100)]
    fetched, fake = _run(
        monkeypatch,
        [_pullpush_payload(page1), _pullpush_payload([])],
        tmp_path,
        pages=3,
    )

    assert fetched == 1
    assert len(fake.calls) == 1  # boundary page never requested
    assert len(_dump_files(tmp_path)) == 1


def test_pullpush_dedupes_boundary_overlap(tmp_path, monkeypatch):
    page1 = [_post(i, NOW - 6 * DAY + i * 60) for i in range(100)]
    # Inclusive `before` returns the boundary post again plus new ones.
    overlap = page1[-1]
    page2 = [overlap] + [_post(200 + i, NOW - 8 * DAY + i * 60) for i in range(6)]
    fetched, _ = _run(
        monkeypatch, [_pullpush_payload(page1), _pullpush_payload(page2)], tmp_path
    )

    assert fetched == 2
    files = _dump_files(tmp_path)
    with files[1].open(encoding="utf-8") as f:
        children = json.load(f)["data"]["children"]
    ids = [c["data"]["id"] for c in children]
    assert "post99" not in ids  # boundary post not written twice
    assert len(ids) == 6


def test_pullpush_failure_returns_zero(tmp_path, monkeypatch):
    err = requests.exceptions.ConnectionError("blocked")
    fetched, _ = _run(monkeypatch, [err, err, err], tmp_path, pages=2)
    assert fetched == 0
    assert _dump_files(tmp_path) == []


def test_pullpush_drops_window_when_sparse_sub_empty(tmp_path, monkeypatch):
    # Sparse sub: nothing inside the month window, then the full
    # archive answers on the retry without the `after` param.
    posts = [_post(1, 1600000000), _post(2, 1600000100)]
    fetched, fake = _run(
        monkeypatch, [_pullpush_payload([]), _pullpush_payload(posts)], tmp_path
    )

    assert fetched == 1
    assert "after" in fake.calls[0]["params"]
    assert "after" not in fake.calls[1]["params"]  # window dropped
    files = _dump_files(tmp_path)
    assert len(files) == 1
    with files[0].open(encoding="utf-8") as f:
        children = json.load(f)["data"]["children"]
    assert len(children) == 2


def test_pullpush_window_empty_twice_stops_cleanly(tmp_path, monkeypatch):
    # Both the windowed and the unwindowed first page are empty:
    # one extra request, then a clean stop (no file, no retry storm).
    fetched, fake = _run(
        monkeypatch, [_pullpush_payload([]), _pullpush_payload([])], tmp_path
    )
    assert fetched == 0
    assert len(fake.calls) == 2
    assert _dump_files(tmp_path) == []


def test_pullpush_429_honors_retry_after(tmp_path, monkeypatch):
    r429 = FakeResp({})
    r429.status_code = 429
    r429.headers = {"Retry-After": "7"}
    posts = [_post(1, NOW - 2 * DAY)]
    fake = FakeGet([r429, _pullpush_payload(posts)])
    monkeypatch.setattr(fetcher.requests, "get", fake)
    sleeps: list[int] = []
    monkeypatch.setattr(fetcher.time, "sleep", sleeps.append)
    monkeypatch.setattr(config, "REDDIT_SOURCE", "pullpush")

    fetched = fetcher.fetch_subreddit_raw(
        "testsub", tmp_path, pages=1, sort="top", time_range="month"
    )

    assert fetched == 1
    assert sleeps == [7]  # Retry-After honored, not the default backoff
    assert len(_dump_files(tmp_path)) == 1


def test_end_to_end_updater_consumes_pullpush_dump(tmp_path, monkeypatch):
    posts = [_post(7, 1700000000), {**_post(8, 1700000100), "is_self": True}]
    fetched, _ = _run(monkeypatch, [_pullpush_payload(posts)], tmp_path)
    assert fetched == 1

    posts_file = tmp_path / "posts.json"
    monkeypatch.setattr(
        config, "REDDIT_SUBREDDITS",
        {"testsub": {"raw_dir": tmp_path, "posts_file": posts_file,
                     "display_name": "r/testsub", "embed_color": 0}},
    )
    from reddit import updater

    updater.build("testsub")
    with posts_file.open(encoding="utf-8") as f:
        cleaned = json.load(f)
    # is_self post filtered, image post kept with score sort intact.
    assert [p["title"] for p in cleaned] == ["Title 7"]
    assert cleaned[0]["url"] == "https://reddit.com/r/testsub/comments/post7"


# ---------------------------------------------------------------------
# Legacy source
# ---------------------------------------------------------------------


def test_reddit_source_keeps_legacy_url(tmp_path, monkeypatch):
    posts = [_post(1, 1700000000)]
    fetched, fake = _run(
        monkeypatch, [_reddit_payload(posts, after=None)], tmp_path, source="reddit"
    )

    assert fetched == 1
    assert "www.reddit.com/r/testsub/top.json" in fake.calls[0]["url"]
    assert fake.calls[0]["params"]["t"] == "month"
    files = _dump_files(tmp_path)
    with files[0].open(encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["data"]["children"][0]["data"]["title"] == "Title 1"
    # after=None on first page -> end of listing, no second call.
    assert len(fake.calls) == 1
