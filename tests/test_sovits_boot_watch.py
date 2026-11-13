"""
tests/test_sovits_boot_watch.py
===============================
Covers the GPT-SoVITS boot telemetry + stall watchdog in voice/sovits.py:

- _boot_log_tip reads only bytes appended since the previous call, so a
  fresh boot never inherits the previous session's tail as its own progress
- a boot whose api_boot.log stops growing is abandoned and respawned
  inside one startup budget instead of blind-waiting the whole timeout
- a boot that crashes or a server that starts listening late is handled
  (respawned / kept, respectively)
- readiness.kick(on_ready=...) fires its callback exactly once once the
  TTS engine settles
- the metrics reconnect counter counts discord.py reconnect log lines
"""

from __future__ import annotations

import asyncio
import logging

import pytest

import config
import voice.readiness as readiness_module
from metrics import collector as metrics_collector
from voice import sovits

# ---------------------------------------------------------------------------
# _boot_log_tip
# ---------------------------------------------------------------------------

class _BootLog:
    """Repoints sovits at a temp dir and exposes its api_boot.log."""

    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "VOICE_SOVITS_SERVER_DIR", str(tmp_path))
        self.path = tmp_path / "api_boot.log"

    def write(self, data: bytes) -> None:
        with open(self.path, "ab") as f:
            f.write(data)


@pytest.fixture()
def boot_log(tmp_path, monkeypatch):
    return _BootLog(tmp_path, monkeypatch)


def test_tip_reads_only_new_lines(boot_log):
    # Baseline over a pre-existing (previous-session) tail.
    first = b"INFO: POST /tts 200 OK\n"
    boot_log.write(first)
    size, tip = sovits._boot_log_tip(0)
    assert size == len(first)
    assert tip == "INFO: POST /tts 200 OK"

    # The fresh server appends; tip must show only the new bytes.
    appended = b"Loading VITS weights\r  42%|....|\r"
    boot_log.write(appended)
    size, tip = sovits._boot_log_tip(size)
    assert size == len(first) + len(appended)
    assert "200 OK" not in tip
    assert "42%" in tip  # \r-separated progress bars count as line breaks


def test_tip_unchanged_size_reports_nothing(boot_log):
    boot_log.write(b"hi\n")
    size, _ = sovits._boot_log_tip(0)
    size2, tip = sovits._boot_log_tip(size)
    assert size2 == size
    assert tip == ""


def test_tip_missing_file(boot_log):
    assert sovits._boot_log_tip(0) == (0, "")


def test_tip_line_length_capped(boot_log):
    boot_log.write(("x" * 500 + "\n").encode())
    _, tip = sovits._boot_log_tip(0)
    assert len(tip) <= sovits._TIP_MAX_CHARS


# ---------------------------------------------------------------------------
# ensure_server / _wait_for_listen (stall watchdog + retries)
# ---------------------------------------------------------------------------

class _TcpScript:
    """Consumes prepared answers on every _tcp_ok() call, in order."""

    def __init__(self, *answers):
        self.answers = list(answers)

    def __call__(self, timeout: float = 1.0) -> bool:
        if len(self.answers) > 1:
            return self.answers.pop(0)
        return self.answers[0]


class _Recorder:
    def __init__(self):
        self.spawned: list[int] = []
        self.killed: list[int] = []

    def spawn(self) -> int:
        pid = 1000 + len(self.spawned) + 1
        self.spawned.append(pid)
        # Mimic _spawn_server's pid file so a follow-up attempt can
        # find and clean it up.
        with open(sovits._pid_file(), "w", encoding="utf-8") as f:
            f.write(str(pid))
        return pid

    def kill(self, pid: int) -> None:
        self.killed.append(pid)


@pytest.fixture()
def watch_env(tmp_path, monkeypatch):
    _BootLog(tmp_path, monkeypatch)  # fresh, empty api_boot.log
    rec = _Recorder()
    monkeypatch.setattr(config, "VOICE_SOVITS_STARTUP_TIMEOUT", 1000.0)
    monkeypatch.setattr(config, "VOICE_SOVITS_BOOT_STALL_TIMEOUT", 1.0)
    monkeypatch.setattr(config, "VOICE_SOVITS_BOOT_ATTEMPTS", 2)
    monkeypatch.setattr(sovits, "_POLL_S", 0.01)
    monkeypatch.setattr(sovits, "_RESTART_GRACE_S", 0.01)
    monkeypatch.setattr(sovits, "_spawn_server", rec.spawn)
    monkeypatch.setattr(sovits, "_kill_pid", rec.kill)
    monkeypatch.setattr(sovits, "_pid_alive", lambda pid: True)
    return rec


def test_already_listening_short_circuits(watch_env, monkeypatch):
    monkeypatch.setattr(sovits, "_tcp_ok", _TcpScript(True))
    assert asyncio.run(sovits.ensure_server()) is True
    assert watch_env.spawned == []


def test_stalled_boot_is_respawned(watch_env, monkeypatch):
    # Boot log never advances and the port never opens: both attempts
    # stall out; attempt 2 must clean up attempt 1's pid. Overall this
    # can't happen with real intervals (stall default 180s), the point
    # is the decision logic.
    monkeypatch.setattr(sovits, "_tcp_ok", _TcpScript(*[False] * 10))
    assert asyncio.run(sovits.ensure_server()) is False
    assert len(watch_env.spawned) == 2
    assert watch_env.killed == [1001]


def test_stalled_boot_then_survivor_is_kept(watch_env, monkeypatch):
    # Attempt 1 stalls (no boot-log progress), but by the time the
    # retry starts the server is finally listening -> keep it.
    monkeypatch.setattr(sovits, "_tcp_ok", _TcpScript(False, False, False, True))
    assert asyncio.run(sovits.ensure_server()) is True
    assert len(watch_env.spawned) == 1
    assert watch_env.killed == []


def test_late_listener_accepted_mid_wait(watch_env, monkeypatch):
    # No stall (watchdog disabled): the server simply comes up on the
    # second poll of the very first attempt.
    monkeypatch.setattr(config, "VOICE_SOVITS_BOOT_STALL_TIMEOUT", 0.0)
    monkeypatch.setattr(sovits, "_tcp_ok", _TcpScript(False, False, False, True))
    assert asyncio.run(sovits.ensure_server()) is True
    assert len(watch_env.spawned) == 1
    assert watch_env.killed == []


def test_crashed_boot_is_respawned(watch_env, monkeypatch):
    monkeypatch.setattr(sovits, "_tcp_ok", _TcpScript(*[False] * 10))
    monkeypatch.setattr(sovits, "_pid_alive", lambda pid: False)
    assert asyncio.run(sovits.ensure_server()) is False
    assert len(watch_env.spawned) == 2
    assert watch_env.killed == []  # nothing to kill: the process died


# ---------------------------------------------------------------------------
# readiness on-ready hook
# ---------------------------------------------------------------------------

async def test_on_ready_hook_fires_once(monkeypatch):
    monkeypatch.setattr(readiness_module, "_POLL_INTERVAL", 0.01)
    readiness_module._state = readiness_module.STATE_STARTING
    readiness_module._task = None
    readiness_module._on_ready_hook = None

    try:
        fired: list[bool] = []

        async def _settle() -> None:
            readiness_module._set_state(readiness_module.STATE_READY)

        monkeypatch.setattr(readiness_module, "_refresh", _settle)

        readiness_module.kick(on_ready=lambda: fired.append(True))
        for _ in range(300):
            if fired or readiness_module._task.done():
                break
            await asyncio.sleep(0.01)

        assert fired == [True]
        assert readiness_module.is_ready()
    finally:
        readiness_module._state = readiness_module.STATE_STARTING
        readiness_module._task = None
        readiness_module._on_ready_hook = None


# ---------------------------------------------------------------------------
# gateway reconnect metric
# ---------------------------------------------------------------------------

def _record(name: str, msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_reconnect_filter_counts_discord_reconnects():
    listener = metrics_collector._ReconnectCountingFilter()
    before = metrics_collector.gateway_reconnect_count()

    rec = _record("discord.client", "Attempting a reconnect in %.2fs")
    assert listener.filter(rec) is True  # never swallows records
    assert metrics_collector.gateway_reconnect_count() == before + 1

    assert listener.filter(_record("discord.client", "Shard ID None has sent a RESUME")) is True
    assert metrics_collector.gateway_reconnect_count() == before + 1
