"""Client for the phoneme worker (see tools/phoneme_worker.py).

Spawns a persistent helper process inside the GPT-SoVITS venv so every
sentence can be phonemized with the exact backend the TTS speaks with,
without paying the import cost more than once. Communicates via one
JSON object per line over stdio, mirroring how voice.sovits manages
the synthesis server (lazy ensure, health re-check on failure).
"""
import json
import os
import queue
import subprocess
import threading

try:  # repo-relative config (worker exe comes from the SoVITS settings)
    import config
except Exception:  # pragma: no cover - standalone use
    config = None

log = __import__("logging").getLogger("desktop.phonemes")

CREATE_NO_WINDOW = 0x08000000


class PhonemeError(RuntimeError):
    pass


class PhonemeWorker:
    """Lazy persistent worker subprocess with JSON-lines request/reply."""

    def __init__(self, python_exe: str | None = None,
                 start_timeout: float = 45.0) -> None:
        self._exe = python_exe
        self._start_timeout = start_timeout
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[dict | None] = queue.Queue()
        self._next_id = 1
        self._ready = False

    # --- lifecycle -------------------------------------------------

    def _exe_path(self) -> str:
        if self._exe and os.path.exists(self._exe):
            return self._exe
        self._exe = (config.VOICE_SOVITS_SERVER_PY
                     if config is not None else "python")
        return self._exe

    def _start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "tools", "phoneme_worker.py")
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        self._lines = queue.Queue()
        self._ready = False
        try:
            self._proc = subprocess.Popen(
                [self._exe_path(), "-u", worker],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=os.path.dirname(worker) or None,
                env=env,
                text=True, encoding="utf-8", errors="replace",
                creationflags=CREATE_NO_WINDOW,
            )
        except OSError as exc:
            self._proc = None
            raise PhonemeError(f"could not spawn phoneme worker: {exc}") from exc
        threading.Thread(target=self._reader, daemon=True).start()

        try:
            first = self._wait_line(self._start_timeout)
        except queue.Empty:
            self.close()
            raise PhonemeError(
                f"phoneme worker did not answer within {self._start_timeout:.0f}s"
            ) from None
        if not first or first.get("event") != "ready":
            self.close()
            raise PhonemeError(f"phoneme worker init failed: {first}")
        self._ready = True
        log.info("Phoneme worker ready (pid %s)", self._proc.pid)

    def _reader(self) -> None:
        while True:
            try:
                line = self._proc.stdout.readline()
            except (OSError, ValueError):
                line = ""
            if not line:
                self._lines.put(None)  # EOF -> all waiters fail
                return
            try:
                self._lines.put(json.loads(line))
            except ValueError:
                log.warning("phoneme worker sent junk line: %.120s", line)

    def _wait_line(self, timeout: float):
        return self._lines.get(timeout=timeout)

    # --- public API ------------------------------------------------

    def ensure(self) -> bool:
        try:
            self._start()
        except PhonemeError as exc:
            log.warning("%s", exc)
            return False
        return self._ready

    def phonemize(self, text: str, lang: str = "en",
                  timeout: float = 20.0) -> list[str]:
        """Return the phone list for `text` (may block until ready)."""
        if not self.ensure():
            raise PhonemeError("phoneme worker unavailable")
        if self._proc.poll() is not None:  # died meanwhile: respawn
            self._proc = None
            if not self.ensure():
                raise PhonemeError("phoneme worker respawn failed")
        rid = self._next_id
        self._next_id += 1
        line = json.dumps({"id": rid, "text": text, "lang": lang})
        assert self._proc.stdin is not None
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()
        while True:
            try:
                resp = self._wait_line(timeout)
            except queue.Empty:
                raise PhonemeError(
                    f"phoneme reply timeout after {timeout:.0f}s") from None
            if resp is None:
                self._proc = None
                raise PhonemeError("phoneme worker exited unexpectedly")
            if resp.get("id") != rid:
                continue  # stray/mismatched line: keep waiting for ours
            if not resp.get("ok"):
                raise PhonemeError(f"phonemize failed: {resp.get('err')}")
            return resp.get("phones", [])

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
        self._ready = False
