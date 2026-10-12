"""
voice/sovits.py
===============
GPT-SoVITS TTS client and auto-managed API server.

Synthesizes text through the local GPT-SoVITS HTTP API (a fine-tuned
Alice model) and, when nothing is listening yet, starts the API server
process on demand and waits for it to become usable.
"""

from __future__ import annotations

import asyncio
import os
import re
import socket
import subprocess
import time
from typing import Optional
from urllib.parse import urlparse

import aiohttp

from bot.logging_setup import get_logger
import config

log = get_logger("voice.sovits")

_JA_RE = re.compile(r"[぀-ヿ㐀-鿿]")
_HEALTH_TTL = 5.0
_DEAD_BACKOFF = 10.0
_MIN_AUDIO_BYTES = 1000

_health_lock = asyncio.Lock()
_start_lock = asyncio.Lock()
_healthy_until = 0.0
_dead_until = 0.0


def detect_lang(text: str) -> str:
    """Resolve the text_lang field for the API (config override or auto)."""
    cfg = (config.VOICE_SOVITS_TEXT_LANG or "auto").strip().lower()
    if cfg in ("ja", "en"):
        return cfg
    return "ja" if _JA_RE.search(text) else "en"


def _endpoint() -> tuple[str, int]:
    parsed = urlparse(config.VOICE_SOVITS_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def _tcp_ok(timeout: float = 1.0) -> bool:
    host, port = _endpoint()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _pid_file() -> str:
    return os.path.join(config.VOICE_SOVITS_SERVER_DIR, "api.pid")


def _read_pid() -> Optional[int]:
    try:
        with open(_pid_file(), "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    # os.kill(pid, 0) terminates the process on Windows, so query instead.
    try:
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        return False


def _kill_pid(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            capture_output=True,
            timeout=15,
        )
    except Exception as e:
        log.warning(f"taskkill {pid} failed: {e}")


def _spawn_server() -> int:
    server_dir = config.VOICE_SOVITS_SERVER_DIR
    env = dict(os.environ)
    env["PATH"] = config.VOICE_SOVITS_FFMPEG_BINS + ";" + env.get("PATH", "")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    log_path = os.path.join(server_dir, "api_boot.log")
    log_fh = open(log_path, "ab")
    try:
        proc = subprocess.Popen(
            [
                config.VOICE_SOVITS_SERVER_PY,
                "-s",
                config.VOICE_SOVITS_SERVER_ENTRY,
                "-a", _endpoint()[0],
                "-p", str(_endpoint()[1]),
                "-c", config.VOICE_SOVITS_SERVER_CONFIG,
            ],
            cwd=server_dir,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000,
        )
    finally:
        log_fh.close()

    try:
        with open(_pid_file(), "w", encoding="utf-8") as f:
            f.write(str(proc.pid))
    except OSError as e:
        log.warning(f"Could not write api.pid: {e}")

    return proc.pid


async def ensure_server() -> bool:
    """Make sure the API server is listening; spawn it if not."""
    async with _start_lock:
        if _tcp_ok():
            return True

        stale = _read_pid()
        if stale is not None and _pid_alive(stale) and not _tcp_ok(timeout=0.5):
            log.info(f"GPT-SoVITS API not responding (pid {stale}); restarting it")
            await asyncio.to_thread(_kill_pid, stale)
            await asyncio.sleep(1.0)

        log.info(
            f"Starting GPT-SoVITS API server "
            f"({config.VOICE_SOVITS_SERVER_ENTRY} in {config.VOICE_SOVITS_SERVER_DIR})"
        )
        try:
            pid = await asyncio.to_thread(_spawn_server)
        except Exception as e:
            log.error(f"Could not start GPT-SoVITS API server: {e}")
            return False
        log.info(f"GPT-SoVITS API server pid {pid}")

        started = time.monotonic()
        deadline = started + config.VOICE_SOVITS_STARTUP_TIMEOUT
        next_log = started + 10.0
        while time.monotonic() < deadline:
            if _tcp_ok():
                _set_health(True)
                log.info(f"GPT-SoVITS API listening after {time.monotonic() - started:.0f}s")
                return True
            if not _pid_alive(pid):
                log.error("GPT-SoVITS API process exited during startup")
                return False
            if time.monotonic() >= next_log:
                log.info("Waiting for GPT-SoVITS API to come up...")
                next_log = time.monotonic() + 10.0
            await asyncio.sleep(0.5)

        log.error(
            f"GPT-SoVITS API did not listen within "
            f"{config.VOICE_SOVITS_STARTUP_TIMEOUT:.0f}s"
        )
        return False


async def health(force: bool = False) -> bool:
    """Cached liveness check for the API endpoint."""
    now = time.monotonic()
    if now < _dead_until:
        return False
    if not force and now < _healthy_until:
        return True
    async with _health_lock:
        now = time.monotonic()
        if not force and now < _healthy_until:
            return True
        ok = await asyncio.to_thread(_tcp_ok)
        _set_health(ok)
        return ok


def _set_health(ok: bool) -> None:
    global _healthy_until, _dead_until
    now = time.monotonic()
    if ok:
        _healthy_until = now + _HEALTH_TTL
        _dead_until = 0.0
    else:
        _healthy_until = 0.0
        _dead_until = now + _DEAD_BACKOFF


def mark_down() -> None:
    """Remember that the endpoint just failed a real request."""
    _set_health(False)


async def probe() -> bool:
    """End-to-end check: tiny synthesis through the API."""
    try:
        if not await health(force=True):
            return False
        wav_bytes, _sr = await generate_async("test.", text_lang="en", _no_retry=True)
        return len(wav_bytes) >= _MIN_AUDIO_BYTES
    except Exception as e:
        log.debug(f"SoVITS probe failed: {e}")
        return False


async def generate_async(
    text: str,
    text_lang: Optional[str] = None,
    _no_retry: bool = False,
) -> tuple[bytes, int]:
    """
    Synthesize text via the GPT-SoVITS API.

    Returns (wav_bytes, sample_rate). Raises on failure so callers can
    fall back to Edge-TTS.
    """
    if not text.strip():
        raise ValueError("Empty text")

    if not await health():
        if _no_retry or not await ensure_server():
            raise RuntimeError("GPT-SoVITS API is not available")
        if not await health(force=True):
            raise RuntimeError("GPT-SoVITS API did not become healthy")

    lang = text_lang or detect_lang(text)
    ref_audio = os.path.abspath(config.VOICE_SOVITS_REF_AUDIO)
    payload = {
        "text": text,
        "text_lang": lang,
        "ref_audio_path": ref_audio,
        "prompt_text": "",
        "prompt_lang": "en",
        "top_k": config.VOICE_SOVITS_TOP_K,
        "top_p": config.VOICE_SOVITS_TOP_P,
        "temperature": config.VOICE_SOVITS_TEMPERATURE,
        "speed_factor": config.VOICE_SOVITS_SPEED,
        "text_split_method": config.VOICE_SOVITS_SPLIT,
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": 0,
    }

    # Warm-up/probe calls (_no_retry) tolerate cold weight loading on a
    # fresh server, which can exceed the normal per-request timeout.
    total = (
        config.VOICE_SOVITS_WARMUP_TIMEOUT
        if _no_retry
        else config.VOICE_SOVITS_TIMEOUT
    )
    try:
        timeout = aiohttp.ClientTimeout(total=total)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(config.VOICE_SOVITS_URL, json=payload) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:300]
                    raise RuntimeError(f"API HTTP {resp.status}: {body}")
                data = await resp.read()
        if len(data) < _MIN_AUDIO_BYTES:
            raise RuntimeError(f"API returned {len(data)} bytes")
    except asyncio.TimeoutError:
        mark_down()
        raise RuntimeError(f"GPT-SoVITS timed out after {total:.0f}s") from None
    except Exception:
        mark_down()
        raise

    _set_health(True)
    return data, 32000
