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
from urllib.parse import urlparse

import aiohttp
import numpy as np

import config
from bot.logging_setup import get_logger

log = get_logger("voice.sovits")

_JA_RE = re.compile(r"[぀-ヿ㐀-鿿]")
_HEALTH_TTL = 5.0
_DEAD_BACKOFF = 10.0
_MIN_AUDIO_BYTES = 1000

_health_lock = asyncio.Lock()
_start_lock = asyncio.Lock()
_healthy_until = 0.0
_dead_until = 0.0

# The API server writes its own stdout/stderr here (appended across
# runs). While waiting for the port to open this file is our boot
# progress telemetry: the newest line is echoed into the bot's
# "Waiting for GPT-SoVITS API..." logs, and a boot whose log stops
# growing is treated as hung instead of burning the whole startup
# budget on a process that will never listen.
_BOOT_LOG_NAME = "api_boot.log"
_TIP_MAX_READ = 4096
_TIP_MAX_CHARS = 160

# Wait-loop cadence and the grace period after a kill before respawning
# (module constants so tests can shrink them).
_POLL_S = 0.5
_RESTART_GRACE_S = 1.0


def _monotonic() -> float:
    """Indirection over time.monotonic so tests can drive the clock."""
    return time.monotonic()


def _boot_log_path() -> str:
    return os.path.join(config.VOICE_SOVITS_SERVER_DIR, _BOOT_LOG_NAME)


def _boot_log_size() -> int:
    try:
        return os.path.getsize(_boot_log_path())
    except OSError:
        return 0


def _boot_log_tip(prev_size: int) -> tuple[int, str]:
    """Return (current_size, newest non-empty boot-log line).

    Only bytes appended after prev_size are considered, so a fresh
    boot never inherits the previous session's tail as its own
    progress. An unchanged or missing file returns (prev_size, "").
    """
    prev = max(0, prev_size)
    size = _boot_log_size()
    if size <= prev:
        return (prev, "")
    try:
        with open(_boot_log_path(), "rb") as f:
            read = min(size - prev, _TIP_MAX_READ)
            f.seek(size - read)
            data = f.read(read)
    except OSError:
        return (size, "")
    # tqdm rewrites bars with \r instead of \n, so treat both as line breaks.
    lines = [ln.strip() for ln in data.decode("utf-8", errors="replace").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return (size, "")
    return (size, lines[-1][-_TIP_MAX_CHARS:])


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


def _read_pid() -> int | None:
    try:
        with open(_pid_file(), encoding="utf-8") as f:
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
    # Unbuffered stdout: without -u the interpreter block-buffers into
    # api_boot.log when redirected, so the boot watchdog sees "no
    # output" for over two minutes on a healthy boot and the "Waiting
    # for GPT-SoVITS API..." echo lines stay blind.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log_path = os.path.join(server_dir, "api_boot.log")
    log_fh = open(log_path, "ab")  # noqa: SIM115 -- lifetime = server process; the boot watchdog tails it
    try:
        proc = subprocess.Popen(
            [
                config.VOICE_SOVITS_SERVER_PY,
                "-u",
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
    """Make sure the API server is listening; spawn it if not.

    Spawning may be retried (up to VOICE_SOVITS_BOOT_ATTEMPTS spawn
    cycles inside one VOICE_SOVITS_STARTUP_TIMEOUT budget): each boot
    is abandoned as soon as its api_boot.log stops growing for
    VOICE_SOVITS_BOOT_STALL_TIMEOUT seconds, so a hung cold start
    can't burn the whole budget blind-waiting. Set the stall timeout
    to 0 to wait unconditionally.

    While waiting, the server's own boot-log tail is echoed into the
    bot's logs, so a stall is diagnosable from the console.
    """
    async with _start_lock:
        if _tcp_ok():
            return True

        attempts = max(1, config.VOICE_SOVITS_BOOT_ATTEMPTS)
        deadline = _monotonic() + max(1.0, config.VOICE_SOVITS_STARTUP_TIMEOUT)

        for attempt in range(1, attempts + 1):
            if _monotonic() >= deadline:
                break
            if _tcp_ok():
                # A previously spawned server finished coming up while
                # we were giving up on it -- keep it, don't shoot it.
                _set_health(True)
                log.info("GPT-SoVITS API listening after a delayed start")
                return True

            stale = _read_pid()
            if stale is not None and _pid_alive(stale) and not _tcp_ok(timeout=0.5):
                if attempt == 1:
                    log.info(f"GPT-SoVITS API not responding (pid {stale}); restarting it")
                else:
                    log.info(
                        f"GPT-SoVITS boot attempt {attempt - 1} left an "
                        f"unresponsive pid {stale}; killing it before retrying"
                    )
                await asyncio.to_thread(_kill_pid, stale)
                await asyncio.sleep(_RESTART_GRACE_S)

            log.info(
                f"Starting GPT-SoVITS API server (attempt {attempt}/{attempts}, "
                f"{config.VOICE_SOVITS_SERVER_ENTRY} in {config.VOICE_SOVITS_SERVER_DIR})"
            )
            boot_size = _boot_log_size()
            try:
                pid = await asyncio.to_thread(_spawn_server)
            except Exception as e:
                log.error(f"Could not start GPT-SoVITS API server: {e}")
                return False
            log.info(f"GPT-SoVITS API server pid {pid}")

            if await _wait_for_listen(pid, deadline, boot_size):
                return True

        return False


async def _wait_for_listen(pid: int, deadline: float, boot_size: int) -> bool:
    """Wait for the spawned server to open its TCP port.

    Returns once the port accepts connections; returns False when the
    process exits, the overall deadline passes, or the server's boot
    log stops growing (stall watchdog).
    """
    started = _monotonic()
    next_log = started + 10.0
    last_progress = started
    tip = ""
    stall_timeout = max(0.0, config.VOICE_SOVITS_BOOT_STALL_TIMEOUT)

    while _monotonic() < deadline:
        if _tcp_ok():
            _set_health(True)
            log.info(f"GPT-SoVITS API listening after {_monotonic() - started:.0f}s")
            return True
        if not _pid_alive(pid):
            log.error("GPT-SoVITS API process exited during startup")
            return False

        new_size, tip = _boot_log_tip(boot_size)
        if new_size != boot_size:
            boot_size = new_size
            last_progress = _monotonic()

        now = _monotonic()
        if stall_timeout > 0 and now - last_progress > stall_timeout:
            log.error(
                f"GPT-SoVITS boot stalled: no api_boot.log progress for "
                f"{now - last_progress:.0f}s (last line: {tip or 'none'})"
            )
            return False

        if now >= next_log:
            detail = tip if tip else "no boot output yet"
            log.info(
                f"Waiting for GPT-SoVITS API to come up... ({detail}, "
                f"{now - started:.0f}s)"
            )
            next_log = _monotonic() + 10.0
        await asyncio.sleep(_POLL_S)

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


async def generate_stream(
    text: str,
    text_lang: str | None = None,
):
    """
    Streaming synthesis via the GPT-SoVITS API (streaming_mode=1).

    Yields float32 mono PCM chunks at the API's native 32kHz as the
    server generates them; the first chunk typically arrives within a
    few hundred milliseconds of the request instead of after the whole
    sentence has been synthesized. Caller feeds the chunks straight
    into a StreamingAudioSource.

    Raises on failure so the caller falls back to the non-streaming
    path (same contract as generate_async).
    """
    if not text.strip():
        raise ValueError("Empty text")

    if not await health():
        if not await ensure_server():
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
        # The streaming path receives ONE sentence (the caller already
        # split it). Server-side re-splitting (cut5) would carve tiny
        # fragments ("Yeah..." -> a 4-char segment), insert the 0.3s
        # inter-fragment silence mid-sentence, and push the chunk-overlap
        # math into its most fragile case -- audibly choppy prosody on
        # short replies. One sentence = one fragment = one AR stream.
        # (This api_v2 build rejects "none"; cut0 IS "no segmentation".)
        "text_split_method": "cut0",
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": config.VOICE_SOVITS_STREAM_MODE,
    }

    parser = _WavStreamParser()
    frames = 0
    try:
        timeout = aiohttp.ClientTimeout(total=config.VOICE_SOVITS_TIMEOUT)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(config.VOICE_SOVITS_URL, json=payload) as resp,
        ):
            if resp.status != 200:
                body = (await resp.text())[:300]
                raise RuntimeError(f"API HTTP {resp.status}: {body}")
            ctype = resp.headers.get("content-type", "")
            if ctype.startswith("application/json"):
                body = (await resp.text())[:300]
                raise RuntimeError(f"API returned JSON error: {body}")
            async for chunk in resp.content.iter_chunked(8192):
                for audio in parser.feed(chunk):
                    frames += len(audio)
                    yield audio
            for audio in parser.flush():
                frames += len(audio)
                yield audio

            if frames * 4 < _MIN_AUDIO_BYTES and not frames:
                # Nothing usable came out; force the caller's fallback.
                raise RuntimeError("GPT-SoVITS stream produced no audio")
    except TimeoutError:
        mark_down()
        raise RuntimeError(
            f"GPT-SoVITS stream timed out after {config.VOICE_SOVITS_TIMEOUT:.0f}s"
        ) from None
    except Exception:
        mark_down()
        raise

    _set_health(True)


class _WavStreamParser:
    """
    Incremental RIFF/WAVE parser for a server that streams a WAV in
    arbitrary network chunks: one header (RIFF/fmt/data chunk, sizes
    may be placeholders) followed by raw PCM data that can split at any
    byte position, even mid-sample.

    feed() returns whole float32 mono sample arrays as they become
    available; anything less than one full frame is carried over to the
    next chunk. flush() drains any final complete frames (odd trailing
    bytes are dropped -- a truncated final sample is not worth playing).
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._channels = 1
        self._bits = 16
        self._data_start: int | None = None

    # --- frame geometry -------------------------------------------------
    def _bytes_per_frame(self) -> int:
        return self._channels * (self._bits // 8)

    def _to_float(self, pcm: bytes) -> list[np.ndarray]:
        if not pcm:
            return []
        if self._bits == 32:
            audio = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if self._channels > 1:
            audio = audio.reshape(-1, self._channels).mean(axis=1)
        return [audio]

    # --- header -------------------------------------------------------
    def _try_parse_header(self) -> bool:
        buf = self._buf
        if len(buf) < 12:
            return False
        if bytes(buf[:4]) != b"RIFF" or bytes(buf[8:12]) != b"WAVE":
            raise ValueError("GPT-SoVITS stream is not WAV data")

        pos = 12
        while pos + 8 <= len(buf):
            cid = bytes(buf[pos:pos + 4])
            size = int.from_bytes(buf[pos + 4:pos + 8], "little")
            if cid == b"fmt ":
                if pos + 8 + size > len(buf):
                    return False  # need more header bytes
                # fmt body layout: format@0, channels@2, rate@4,
                # byteRate@8, blockAlign@12, bits@14 (all LE).
                body = pos + 8
                fmt_format = int.from_bytes(buf[body:body + 2], "little")
                if fmt_format not in (1, 0xFFFE):
                    raise ValueError(
                        f"Unsupported WAV encoding {fmt_format} in "
                        "GPT-SoVITS stream"
                    )
                self._channels = (
                    int.from_bytes(buf[body + 2:body + 4], "little") or 1
                )
                self._bits = (
                    int.from_bytes(buf[body + 14:body + 16], "little") or 16
                )
            elif cid == b"data":
                self._data_start = pos + 8
                return True
            elif size > 64 * 1024 * 1024:
                # A size this large before the data chunk means we're
                # misreading the stream; refuse rather than buffer it.
                raise ValueError(f"Implausible WAV chunk size {size} ({cid!r})")
            pos += 8 + size
        return False

    # --- byte ingestion -------------------------------------------------
    def feed(self, data: bytes) -> list[np.ndarray]:
        if not data:
            return []
        if self._data_start is None:
            self._buf.extend(data)
            if not self._try_parse_header():
                return []
            del self._buf[: self._data_start]
        else:
            self._buf.extend(data)
        return self._drain()

    def flush(self) -> list[np.ndarray]:
        if self._data_start is None:
            return []
        return self._drain()

    def _drain(self) -> list[np.ndarray]:
        frame = self._bytes_per_frame()
        if frame <= 0:
            return []
        usable = (len(self._buf) // frame) * frame
        if usable == 0:
            return []
        pcm = bytes(self._buf[:usable])
        del self._buf[:usable]
        return self._to_float(pcm)


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
    text_lang: str | None = None,
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
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(config.VOICE_SOVITS_URL, json=payload) as resp,
        ):
            if resp.status != 200:
                body = (await resp.text())[:300]
                raise RuntimeError(f"API HTTP {resp.status}: {body}")
            data = await resp.read()
            if len(data) < _MIN_AUDIO_BYTES:
                raise RuntimeError(f"API returned {len(data)} bytes")
    except TimeoutError:
        mark_down()
        raise RuntimeError(f"GPT-SoVITS timed out after {total:.0f}s") from None
    except Exception:
        mark_down()
        raise

    _set_health(True)
    return data, 32000
