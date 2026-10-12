"""
Tests for reddit.rotation module.
"""

import json
import tempfile
from pathlib import Path

import config
from reddit.rotation import get_random_post


class TestRotation:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.posts_file = Path(self.temp_dir) / "posts.json"
        self.state_file = Path(self.temp_dir) / "rotation_state.json"

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_test_posts(self, count=5):
        posts = []
        for i in range(count):
            posts.append({
                "title": f"Post {i}",
                "url": f"https://reddit.com/r/test/post/{i}",
                "image": f"https://i.redd.it/image{i}.jpg",
                "score": 100 - i,
            })
        with self.posts_file.open("w") as f:
            json.dump(posts, f)
        return posts

    def _patch_config(self, monkeypatch):
        """Patch config for test subreddit."""
        monkeypatch.setattr(config, "REDDIT_SUBREDDITS", {
            "test_sub": {
                "posts_file": self.posts_file,
                "raw_dir": Path(self.temp_dir) / "raw",
                "display_name": "r/test",
                "embed_color": 0x000000,
            }
        })
        monkeypatch.setattr(config, "REDDIT_ROTATION_STATE_PATH", self.state_file)

    def test_get_random_post_returns_post(self, monkeypatch):
        self._patch_config(monkeypatch)
        self.create_test_posts(3)
        post = get_random_post("test_sub")
        assert post is not None
        assert "title" in post
        assert "url" in post

    def test_get_random_post_no_posts_file(self, monkeypatch):
        monkeypatch.setattr(config, "REDDIT_SUBREDDITS", {})
        post = get_random_post("nonexistent")
        assert post is None

    def test_get_random_post_empty_posts(self, monkeypatch):
        self._patch_config(monkeypatch)
        self.posts_file.write_text("[]")
        post = get_random_post("test_sub")
        assert post is None

    def test_rotation_shuffles(self, monkeypatch):
        self._patch_config(monkeypatch)
        self.create_test_posts(10)
        seen = set()
        for _ in range(10):
            post = get_random_post("test_sub")
            assert post is not None
            seen.add(post["url"])
        assert len(seen) > 1

    def test_state_persists_between_calls(self, monkeypatch):
        self._patch_config(monkeypatch)
        self.create_test_posts(5)
        first = get_random_post("test_sub")
        second = get_random_post("test_sub")
        assert first["url"] != second["url"]

    def test_refill_when_exhausted(self, monkeypatch):
        self._patch_config(monkeypatch)
        self.create_test_posts(2)
        seen = set()
        for _ in range(5):
            post = get_random_post("test_sub")
            assert post is not None
            seen.add(post["url"])
        assert len(seen) == 2

    def test_unknown_subreddit(self, monkeypatch):
        monkeypatch.setattr(config, "REDDIT_SUBREDDITS", {})
        post = get_random_post("unknown_subreddit_key")
        assert post is None
