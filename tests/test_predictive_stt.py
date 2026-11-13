"""
Tests for the predictive-STT state machine in voice/pipeline.py:
spawn conditions, freshness (revision) checks at the gate, and the
bounded wait. Uses a fake sink with the same observation/snapshot/
get_revision interface AliceVoiceSink exposes.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np

import config
from voice.pipeline import VoicePipeline

GUILD = 123
USER = 42


class FakeSink:
    def __init__(self, revision: int = 1):
        self.revision = revision
        self.obs = {
            "silence_duration": 0.4,
            "buffer_duration": 1.2,
            "revision": revision,
        }

    def observation(self, user_id: int):
        return dict(self.obs)

    def snapshot_audio(self, user_id: int):
        return (np.zeros(0, dtype=np.float32), self.revision)

    def get_revision(self, user_id: int):
        return self.revision


def _make_pipeline(monkeypatch, live: bool = True) -> VoicePipeline:
    bot = SimpleNamespace(
        character=SimpleNamespace(display_name="x"),
        ai_client=SimpleNamespace(),
        user=SimpleNamespace(id=1),
    )
    pipe = VoicePipeline(bot)
    pipe._live = live
    pipe.sinks[GUILD] = FakeSink()
    return pipe


def test_claim_predictive_none_when_no_decode(monkeypatch):
    pipe = _make_pipeline(monkeypatch)
    result, outcome = asyncio.run(pipe._claim_predictive(GUILD, USER))
    assert (result, outcome) == (None, "none")


def test_spawn_requires_live_and_start_threshold(monkeypatch):
    monkeypatch.setattr(config, "VOICE_STT_PREDICTIVE_START", 0.3)

    async def scenario():
        pipe = _make_pipeline(monkeypatch, live=False)
        pipe._maybe_spawn_predictive(GUILD, USER)
        assert not pipe._pred_stt  # dead when live disabled

        pipe._live = True
        monkeypatch.setattr(config, "VOICE_STT_PREDICTIVE_START", 5.0)
        pipe._maybe_spawn_predictive(GUILD, USER)
        assert not pipe._pred_stt  # silence below the start threshold

    asyncio.run(scenario())


def test_spawn_and_hit(monkeypatch):
    monkeypatch.setattr(config, "VOICE_STT_PREDICTIVE_START", 0.3)
    calls: list[tuple] = []

    async def scenario():
        pipe = _make_pipeline(monkeypatch)

        async def fake_decode(guild, user, audio, sr, revision):
            calls.append((guild, user, revision))
            entry = pipe._pred_stt.get((guild, user))
            if entry is not None and entry["revision"] == revision:
                entry["result"] = {"text": "hi", "language": "en"}

        pipe._predictive_decode = fake_decode
        pipe._maybe_spawn_predictive(GUILD, USER)
        key = (GUILD, USER)
        assert key in pipe._pred_stt
        task = pipe._pred_stt[key]["task"]
        assert task is not None
        await asyncio.sleep(0.05)  # let the decode finish

        result, outcome = await pipe._claim_predictive(GUILD, USER)
        assert outcome == "hit"
        assert result == {"text": "hi", "language": "en"}
        assert calls == [(GUILD, USER, 1)]

    asyncio.run(scenario())


def test_stale_result_rejected(monkeypatch):
    monkeypatch.setattr(config, "VOICE_STT_PREDICTIVE_START", 0.3)

    async def scenario():
        pipe = _make_pipeline(monkeypatch)

        async def fake_decode(guild, user, audio, sr, revision):
            entry = pipe._pred_stt.get((guild, user))
            if entry is not None and entry["revision"] == revision:
                entry["result"] = {"text": "stale", "language": "en"}

        pipe._predictive_decode = fake_decode
        pipe._maybe_spawn_predictive(GUILD, USER)
        await asyncio.sleep(0.05)
        # New speech arrived after the snapshot: buffer revisions diverge.
        pipe.sinks[GUILD].revision = 7
        result, outcome = await pipe._claim_predictive(GUILD, USER)
        assert outcome == "stale"
        assert result is None

    asyncio.run(scenario())


def test_miss_bounded_wait(monkeypatch):
    monkeypatch.setattr(config, "VOICE_STT_PREDICTIVE_START", 0.3)
    monkeypatch.setattr(config, "VOICE_STT_PREDICTIVE_WAIT", 0.05)

    async def scenario():
        pipe = _make_pipeline(monkeypatch)
        gate = asyncio.Event()

        async def stuck_decode(guild, user, audio, sr, revision):
            await gate.wait()
            entry = pipe._pred_stt.get((guild, user))
            if entry is not None:
                entry["result"] = {"text": "late", "language": "en"}

        pipe._predictive_decode = stuck_decode
        pipe._maybe_spawn_predictive(GUILD, USER)
        task = pipe._pred_stt[(GUILD, USER)]["task"]
        result, outcome = await pipe._claim_predictive(GUILD, USER)
        assert outcome == "miss"
        assert result is None
        gate.set()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_claim_pop_cleans_entry(monkeypatch):
    monkeypatch.setattr(config, "VOICE_STT_PREDICTIVE_START", 0.3)

    async def scenario():
        pipe = _make_pipeline(monkeypatch)
        pipe.sinks[GUILD].revision = 3
        entry = {
            "task": None, "revision": 3, "started": 0.0,
            "result": {"text": "cached", "language": "en"},
        }
        pipe._pred_stt[(GUILD, USER)] = entry
        result, outcome = await pipe._claim_predictive(GUILD, USER)
        assert outcome == "hit"
        assert result == entry["result"]
        assert (GUILD, USER) not in pipe._pred_stt

    asyncio.run(scenario())


def test_spawn_skipped_when_hosted_stt(monkeypatch):
    """Hosted STT is on -> no local predictive decode (wasted engine load)."""
    monkeypatch.setattr(config, "VOICE_STT_PREDICTIVE_START", 0.3)
    monkeypatch.setattr(config, "VOICE_STT_HOSTED", True)

    async def scenario():
        pipe = _make_pipeline(monkeypatch)
        pipe._maybe_spawn_predictive(GUILD, USER)
        assert not pipe._pred_stt

    asyncio.run(scenario())


def test_spawn_skipped_while_guild_busy(monkeypatch):
    monkeypatch.setattr(config, "VOICE_STT_PREDICTIVE_START", 0.3)

    async def scenario():
        pipe = _make_pipeline(monkeypatch)
        pipe._processing[GUILD] = True
        pipe._maybe_spawn_predictive(GUILD, USER)
        assert not pipe._pred_stt

        # Another user's decode owning the GPU also skips
        pipe._processing[GUILD] = False
        other = (GUILD, USER + 1)
        never = asyncio.Event()

        async def stub_decode(guild, user, audio, sr, revision):
            await never.wait()

        pipe._predictive_decode = stub_decode
        pipe._pred_stt[other] = {
            "task": asyncio.create_task(stub_decode(GUILD, USER + 1, None, 0, 1)),
            "revision": 1, "started": 0.0, "result": None,
        }
        pipe._maybe_spawn_predictive(GUILD, USER)
        assert (GUILD, USER) not in pipe._pred_stt

        # cleanup: kill the stub task
        pipe._pred_stt[other]["task"].cancel()

    asyncio.run(scenario())
