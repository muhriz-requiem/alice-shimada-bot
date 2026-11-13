"""
Tests for the screen-glance feature: intent matching, privacy denylist,
prompt-image attachment, provider gating, and the pipeline's look hook.
"""
import base64

import pytest

import config
from voice import screen_look as sl

# --- intent matching -----------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Alice, look at my screen", sl.LOOK_INTENT_EXPLICIT),
    ("What's on my screen right now?", sl.LOOK_INTENT_EXPLICIT),
    ("Can you see my screen?", sl.LOOK_INTENT_EXPLICIT),
    ("Check my screen", sl.LOOK_INTENT_EXPLICIT),
    ("What do you think of this?", sl.LOOK_INTENT_DEIXIS),
    ("What is this?", sl.LOOK_INTENT_DEIXIS),
    ("Can you see this?", sl.LOOK_INTENT_DEIXIS),
    ("Alice, did you eat lunch today?", sl.LOOK_INTENT_NONE),
    ("I had a sandwich", sl.LOOK_INTENT_NONE),
    ("", sl.LOOK_INTENT_NONE),
])
def test_match_look_intent(text, expected):
    assert sl.match_look_intent(text) == expected


# --- image attachment ----------------------------------------------------

def test_attach_image_to_last_user():
    messages = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "look"},
    ]
    jpeg = b"\xff\xd8fake"
    out = sl.attach_image_to_last_user(messages, jpeg, "inst", "My Title")
    # Original list untouched (privacy: no shared mutation)
    assert messages[-1]["content"] == "look"
    last = out[-1]
    assert last["role"] == "user"
    content = last["content"]
    assert content[0]["type"] == "text" and "inst" in content[0]["text"]
    assert "My Title" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    # Decodable base64 round-trip
    b64 = content[1]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(b64) == jpeg


def test_attach_image_appends_when_no_user_message():
    messages = [{"role": "system", "content": "persona"}]
    out = sl.attach_image_to_last_user(messages, b"x", "inst", "")
    assert len(out) == 2
    assert out[-1]["role"] == "user"


# --- denylist ------------------------------------------------------------

def test_denylist(monkeypatch):
    monkeypatch.setattr(config, "SCREEN_LOOK_DENYLIST", ["chase", "1password"])
    from desktop import screen_watch
    assert screen_watch.denylisted("Chase Bank - Account")
    assert screen_watch.denylisted("1Password")
    assert not screen_watch.denylisted("Visual Studio Code")
    assert not screen_watch.denylisted("My notes.txt - Notepad")


# --- change detection ----------------------------------------------------

def test_change_fraction():
    import numpy as np

    from desktop import screen_watch
    a = np.zeros((36, 64), dtype=np.float32)
    assert screen_watch.change_fraction(a, a.copy()) == 0.0
    assert screen_watch.change_fraction(None, a) == 1.0
    b = a.copy()
    b[:18, :] = 1.0  # half of the rows changed
    frac = screen_watch.change_fraction(a, b)
    assert 0.45 < frac < 0.55


# --- provider gating in the pipeline hook --------------------------------

class _FakeProvider:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.monitors = []

    async def __call__(self, monitor=None):
        self.calls += 1
        self.monitors.append(monitor)
        return self.result


class _Frame:
    jpeg = b"\xff\xd8fake"
    title = "Notepad"


class _StubBot:
    def __init__(self):
        import character as character_pkg
        self.character = character_pkg.load_character("alice")
        self.ai_client = _StubAIClient()
        self.user = type("U", (), {"id": 1})()
        self.memory_graph = _StubGraph()


class _StubAIClient:
    async def vision_chat(self, messages, **kw):
        return "Oh, you're coding!"


class _StubGraph:
    def __init__(self):
        self.added = []

    def add(self, user_id, text, keywords=None):
        self.added.append((user_id, text))


@pytest.mark.asyncio
async def test_look_hook_disabled_returns_refusal(monkeypatch):
    from voice.pipeline import VoicePipeline

    monkeypatch.setattr(config, "SCREEN_LOOK_ENABLED", False)
    monkeypatch.setattr(config, "SCREEN_LOOK_COOLDOWN_S", 0.0)
    bot = _StubBot()
    pipeline = VoicePipeline(bot)
    provider = _FakeProvider(None)
    pipeline.screen_look_provider = provider

    reply = await pipeline._screen_look_reply(
        guild_id=0, user_id_str="1", channel_id="0",
        text="Alice, look at my screen", look_kind=sl.LOOK_INTENT_EXPLICIT,
        messages=[], identity={}, relationship=None, history=[],
        voice_extra="",
    )
    assert reply == sl.REFUSE_PERMISSION
    assert provider.calls == 0  # nothing captured with the feature off


@pytest.mark.asyncio
async def test_look_hook_refuses_when_provider_denies(monkeypatch):
    from voice.pipeline import VoicePipeline

    monkeypatch.setattr(config, "SCREEN_LOOK_ENABLED", True)
    monkeypatch.setattr(config, "SCREEN_LOOK_COOLDOWN_S", 0.0)
    bot = _StubBot()
    pipeline = VoicePipeline(bot)
    pipeline.screen_look_provider = _FakeProvider(None)  # gate/denylist

    reply = await pipeline._screen_look_reply(
        guild_id=0, user_id_str="1", channel_id="0",
        text="Alice, look at my screen", look_kind=sl.LOOK_INTENT_EXPLICIT,
        messages=[], identity={}, relationship=None, history=[],
        voice_extra="",
    )
    assert reply == sl.REFUSE_PERMISSION


@pytest.mark.asyncio
async def test_look_hook_explicit_success_and_memory(monkeypatch):
    from voice.pipeline import VoicePipeline

    monkeypatch.setattr(config, "SCREEN_LOOK_ENABLED", True)
    monkeypatch.setattr(config, "SCREEN_LOOK_COOLDOWN_S", 0.0)
    monkeypatch.setattr(config, "SCREEN_LOOK_MEMORY", True)
    bot = _StubBot()
    pipeline = VoicePipeline(bot)
    pipeline.screen_look_provider = _FakeProvider(_Frame())

    reply = await pipeline._screen_look_reply(
        guild_id=0, user_id_str="1", channel_id="0",
        text="Alice, look at my screen", look_kind=sl.LOOK_INTENT_EXPLICIT,
        messages=[], identity={}, relationship=None, history=[],
        voice_extra="",
    )
    assert reply == "Oh, you're coding!"
    assert bot.memory_graph.added and "(glance)" in bot.memory_graph.added[0][1]


@pytest.mark.asyncio
async def test_look_hook_deixis_with_feature_off_falls_through(monkeypatch):
    from voice.pipeline import VoicePipeline

    monkeypatch.setattr(config, "SCREEN_LOOK_ENABLED", False)
    bot = _StubBot()
    pipeline = VoicePipeline(bot)
    pipeline.screen_look_provider = _FakeProvider(None)

    reply = await pipeline._screen_look_reply(
        guild_id=0, user_id_str="1", channel_id="0",
        text="what do you think of this?", look_kind=sl.LOOK_INTENT_DEIXIS,
        messages=[], identity={}, relationship=None, history=[],
        voice_extra="",
    )
    assert reply is None  # normal text reply handles it


# --- ambient governor eligibility ---------------------------------------

class _StubTurnDetector:
    def __init__(self, state):
        self._state = state

    def get_state(self, guild_id):
        return self._state


class _StubMixer:
    def __init__(self, playing):
        self._playing = playing

    def is_playing(self, guild_id):
        return self._playing


class _StubPipeline:
    def __init__(self, state, playing):
        self.turn_detector = _StubTurnDetector(state)
        self.mixer = _StubMixer(playing)
        self.bot = _StubBot()


def _make_service(monkeypatch, state, playing, **cfg):
    from desktop.look_service import LookService

    monkeypatch.setattr(config, "SCREEN_LOOK_ENABLED", True)
    monkeypatch.setattr(config, "SCREEN_LOOK_AMBIENT", cfg.get("ambient", True))
    monkeypatch.setattr(config, "SCREEN_LOOK_AMBIENT_COOLDOWN_S", 0.0)
    monkeypatch.setattr(config, "SCREEN_LOOK_AMBIENT_MAX_PER_HOUR", 12)
    svc = LookService(_StubPipeline(state, playing), mate_bridge=None)
    svc.set_allowed(cfg.get("allowed", True))
    return svc


@pytest.mark.asyncio
async def test_ambient_eligible_happy_path(monkeypatch):
    import time as _time

    from voice.turn_detector import TurnState

    svc = _make_service(monkeypatch, TurnState.IDLE, playing=False)
    svc._last_glance_at = _time.monotonic() - 1e6  # long ago
    assert svc._ambient_eligible()


@pytest.mark.asyncio
async def test_ambient_blocked_while_playing(monkeypatch):
    import time as _time

    from voice.turn_detector import TurnState

    svc = _make_service(monkeypatch, TurnState.IDLE, playing=True)
    svc._last_glance_at = _time.monotonic() - 1e6
    assert not svc._ambient_eligible()


@pytest.mark.asyncio
async def test_ambient_blocked_without_permission(monkeypatch):
    import time as _time

    from voice.turn_detector import TurnState

    svc = _make_service(monkeypatch, TurnState.IDLE, playing=False, allowed=False)
    svc._last_glance_at = _time.monotonic() - 1e6
    assert not svc._ambient_eligible()


@pytest.mark.asyncio
async def test_ambient_blocked_mid_conversation(monkeypatch):
    import time as _time

    from voice.turn_detector import TurnState

    svc = _make_service(monkeypatch, TurnState.MULTIPLE_SPEAKERS, playing=False)
    svc._last_glance_at = _time.monotonic() - 1e6
    assert not svc._ambient_eligible()


@pytest.mark.asyncio
async def test_ambient_blocked_on_cooldown(monkeypatch):
    import time as _time

    from voice.turn_detector import TurnState

    svc = _make_service(monkeypatch, TurnState.IDLE, playing=False)
    svc._last_glance_at = _time.monotonic()  # just glanced
    assert not svc._ambient_eligible()


@pytest.mark.asyncio
async def test_ambient_hourly_cap(monkeypatch):
    import time as _time

    from voice.turn_detector import TurnState

    svc = _make_service(monkeypatch, TurnState.IDLE, playing=False)
    svc._last_glance_at = _time.monotonic() - 1e6
    now = _time.time()
    svc._hour_stamps.extend([now] * 12)  # cap reached
    assert not svc._ambient_eligible()


@pytest.mark.asyncio
async def test_look_cooldown_suppresses_repeat(monkeypatch):
    from voice.pipeline import VoicePipeline

    monkeypatch.setattr(config, "SCREEN_LOOK_ENABLED", True)
    monkeypatch.setattr(config, "SCREEN_LOOK_COOLDOWN_S", 60.0)
    bot = _StubBot()
    pipeline = VoicePipeline(bot)
    pipeline.screen_look_provider = _FakeProvider(_Frame())

    first = await pipeline._screen_look_reply(
        guild_id=0, user_id_str="1", channel_id="0",
        text="look", look_kind=sl.LOOK_INTENT_EXPLICIT,
        messages=[], identity={}, relationship=None, history=[],
        voice_extra="",
    )
    second = await pipeline._screen_look_reply(
        guild_id=0, user_id_str="1", channel_id="0",
        text="look", look_kind=sl.LOOK_INTENT_EXPLICIT,
        messages=[], identity={}, relationship=None, history=[],
        voice_extra="",
    )
    assert first is not None
    assert second is None  # cooldown gates the repeat


def test_match_monitor_hint():
    assert sl.match_monitor_hint("Alice, look at my second screen") == "1"
    assert sl.match_monitor_hint("what's on the other monitor?") == "1"
    assert sl.match_monitor_hint("look at my 2nd monitor") == "1"
    assert sl.match_monitor_hint("look at my screen") is None
    assert sl.match_monitor_hint("") is None


@pytest.mark.asyncio
async def test_monitor_hint_reaches_provider(monkeypatch):
    from voice.pipeline import VoicePipeline

    monkeypatch.setattr(config, "SCREEN_LOOK_ENABLED", True)
    monkeypatch.setattr(config, "SCREEN_LOOK_COOLDOWN_S", 0.0)
    bot = _StubBot()
    pipeline = VoicePipeline(bot)
    provider = _FakeProvider(_Frame())
    pipeline.screen_look_provider = provider

    reply = await pipeline._screen_look_reply(
        guild_id=0, user_id_str="1", channel_id="0",
        text="look at my second screen", look_kind=sl.LOOK_INTENT_EXPLICIT,
        monitor=sl.match_monitor_hint("look at my second screen"),
        messages=[], identity={}, relationship=None, history=[],
        voice_extra="",
    )
    assert reply is not None
    assert provider.monitors == ["1"]


def test_record_glance_metrics():
    from metrics import record_glance
    from metrics.collector import glance_latency, glance_total

    before = glance_total.labels(outcome="ok")._value.get()
    record_glance("ok", 1.23)
    assert glance_total.labels(outcome="ok")._value.get() == before + 1
    record_glance("refused")
    record_glance("failed")
    assert glance_latency is not None
