"""
desktop/mate_bridge.py
======================
Client for the MtBridge loopback link into MateEngine X.

Protocol (see MtBridge.cs in the game install): newline-terminated UTF-8
lines over TCP 127.0.0.1:8495 (env MT_BRIDGE_PORT). One line in, one or
more reply lines out; replies arrive asynchronously from the engine's
main thread, so callers use a quiet-window collect (no request ids in
the wire format).

Command surface (implemented in MtBridge):
  ping                    -> pong
  sys                     -> sys datapath <path>
  probe                   -> probe renderer=<name> shapes=<n> <i>:<name>...
  probeall                -> includes inactive SkinnedMeshRenderers
  avinfo                  -> avinfo isLoading=.. currentModel=.. mainModel=..
  preset <shape> <0..1>   -> applied <shape> <w> hits=<n>
  mouth <shape> <0..1>    -> same as preset (reserved for visemes)
  reset                   -> reset done zeroed=<n>
  load <vrm path>         -> load queued path=..
  show                    -> show done main=.. current=..
  roots / tree <name>     -> scene introspection
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bot.logging_setup import get_logger

log = get_logger("desktop.mate_bridge")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8495


class BridgeNotConnected(ConnectionError):
    """Raised when a command is attempted without a live link."""


def vrm_port() -> int:
    val = os.environ.get("MT_BRIDGE_PORT", "").strip()
    if val.isdigit():
        return int(val)
    return DEFAULT_PORT


def _read_gltf_json(path: str) -> Optional[dict]:
    """Read the glTF JSON of a .vrm/.glb/.gltf file (VRM = binary glTF)."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        if head[:4] == b"glTF":
            # GLB: 12-byte header, then chunks (length u32, type u32, data).
            with open(path, "rb") as f:
                f.read(12)
                chunk_len_bytes = f.read(4)
                if len(chunk_len_bytes) < 4:
                    return None
                chunk_len = int.from_bytes(chunk_len_bytes, "little")
                # skip chunk type (4 bytes), read JSON chunk
                f.read(4)
                return json.loads(f.read(chunk_len).decode("utf-8"))
        if head[:2] == b"PK":
            # defensive: some tools re-zip; keep the zip path too
            with zipfile.ZipFile(path) as z:
                with z.open("model.gltf") as fh:
                    return json.loads(fh.read().decode("utf-8"))
        with open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot read glTF JSON from %s: %s", path, exc)
        return None


def vrm_expression_map(vrm_path: str) -> Dict[str, str]:
    """Extract VRM 1.0 preset -> mesh blendshape-name mapping from a .vrm.

    Blendshape names come from mesh.extras.targetNames in the same order
    as the morph targets; expressions carry (mesh, index) binds.

    Returns {} when the file is missing or unparsable - callers must
    fall back to literal preset names (the engine matches by name).
    """
    gltf = _read_gltf_json(vrm_path)
    if not isinstance(gltf, dict):
        return {}

    meshes = gltf.get("meshes") or []
    nodes = gltf.get("nodes") or []
    target_names: List[List[str]] = []
    for mesh in meshes:
        tn = ((mesh.get("extras") or {}).get("targetNames")) or []
        target_names.append(list(tn))

    # VRM 1.0 binds reference a *node* that carries the mesh; 0.x binds
    # reference mesh directly. Resolve both to a mesh index.
    def _bind_mesh(bind: dict) -> Optional[int]:
        m = bind.get("mesh")
        if isinstance(m, int) and 0 <= m < len(target_names):
            return m
        node_i = bind.get("node")
        if isinstance(node_i, int) and 0 <= node_i < len(nodes):
            nm = (nodes[node_i] or {}).get("mesh")
            if isinstance(nm, int) and 0 <= nm < len(target_names):
                return nm
        return None

    exts = gltf.get("extensions") or {}
    ext = exts.get("VRMC_vrm") or exts.get("VRM") or {}
    out: Dict[str, str] = {}

    # VRM 1.0: expressions.preset = {presetName: expr}, expressions.custom = [expr]
    # VRM 0.x: blendShapeMaster.blendShapeGroups = [group with presetName + binds]
    def _walk(expr_entries) -> None:
        for preset, e in expr_entries:
            if not isinstance(e, dict):
                continue
            for bind in e.get("morphTargetBinds") or e.get("binds") or []:
                mesh_i = _bind_mesh(bind) if isinstance(bind, dict) else None
                t_i = bind.get("index") if isinstance(bind, dict) else None
                if (
                    mesh_i is not None
                    and isinstance(t_i, int)
                    and 0 <= t_i < len(target_names[mesh_i])
                ):
                    out[preset] = target_names[mesh_i][t_i]
                    break

    expr_root = (ext.get("expressions") or {})
    entries: List[Tuple[str, Any]] = []
    preset_dict = expr_root.get("preset")
    if isinstance(preset_dict, dict):
        entries.extend(preset_dict.items())
    custom_list = expr_root.get("custom")
    if isinstance(custom_list, list):
        entries.extend((e.get("name") or "", e) for e in custom_list)
    elif isinstance(custom_list, dict):
        entries.extend(custom_list.items())
    # VRM 0.x style array inside expressions (defensive; older dumps)
    if isinstance(expr_root, list):
        entries.extend((e.get("preset") or "", e) for e in expr_root)
    groups = ((ext.get("blendShapeMaster") or {}).get("blendShapeGroups")) or []
    if isinstance(groups, list):
        entries.extend(
            (g.get("presetName") or g.get("name") or "", g) for g in groups)
    _walk(entries)
    return out


def vrm_preset_weights(vrm_path: str) -> Dict[str, float]:
    """Default weights per VRM preset, from the model's JSON as shipped."""
    try:
        with zipfile.ZipFile(vrm_path) as z:
            with z.open("model.gltf") as f:
                gltf = json.loads(f.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot read %s: %s", vrm_path, exc)
        return {}
    exts = gltf.get("extensions") or {}
    ext = exts.get("VRMC_vrm") or exts.get("VRM") or {}
    out: Dict[str, float] = {}
    for section in ("expressions", "custom"):
        for expr in ext.get(section) or []:
            preset = expr.get("preset") or ""
            out[preset] = float(expr.get("defaultWeight") or 0.0)
    return out


# VRM preset keys used by the expression keyspace.
STANDARD_PRESETS = (
    "aa", "ih", "ou", "ee", "oh",
    "blink", "blinkLeft", "blinkRight",
    "happy", "angry", "sad", "surprised",
    "relaxed", "neutral", "lookUp", "lookDown",
    "lookLeft", "lookRight",
)

# Phoneme -> VRM viseme preset. Japanese-heavy pipeline; kana handled
# directly so GPT-SoVITS text can be lip-synced pre-timing.
PHONEME_TO_VISEME = {
    "a": "aa", "aa": "aa", "あ": "aa", "か": "aa", "が": "aa", "さ": "aa",
    "ざ": "aa", "た": "aa", "だ": "aa", "な": "aa", "や": "aa", "ら": "aa",
    "わ": "aa", "ば": "aa", "ぱ": "aa", "ま": "aa",
    "i": "ih", "ih": "ih", "い": "ih", "き": "ih", "ぎ": "ih", "し": "ih",
    "じ": "ih", "ち": "ih", "に": "ih", "ひ": "ih", "り": "ih", "び": "ih",
    "ぴ": "ih", "み": "ih",
    "u": "ou", "ou": "ou", "う": "ou", "く": "ou", "ぐ": "ou", "す": "ou",
    "ず": "ou", "つ": "ou", "ふ": "ou", "ぬ": "ou", "ゆ": "ou", "る": "ou",
    "ぶ": "ou", "ぷ": "ou", "む": "ou",
    "e": "ee", "ee": "ee", "え": "ee", "け": "ee", "げ": "ee", "せ": "ee",
    "ぜ": "ee", "て": "ee", "で": "ee", "ね": "ee", "へ": "ee", "れ": "ee",
    "べ": "ee", "ぺ": "ee", "め": "ee",
    "o": "oh", "oh": "oh", "お": "oh", "こ": "oh", "ご": "oh", "そ": "oh",
    "ぞ": "oh", "と": "oh", "ど": "oh", "の": "oh", "ほ": "oh", "ろ": "oh",
    "よ": "oh", "ぼ": "oh", "ぽ": "oh", "も": "oh",
    "n": "oh", "ん": "oh",
}

_KANA_SKIP = set("っャュョぁぃぅぇぉーヴ")


def kana_to_phonemes(text: str) -> List[str]:
    """Crude phoneme segmenter for viseme timing (no dictionary needed)."""
    out: List[str] = []
    for ch in text:
        cp = ord(ch)
        if ch in _KANA_SKIP:
            continue
        if 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
            out.append(ch)
        elif ch.isascii() and ch.isalpha() and ch.lower() in PHONEME_TO_VISEME:
            out.append(ch.lower())
    return out


def _num(v: float) -> str:
    s = ("%.3f" % v).rstrip("0").rstrip(".")
    return s or "0"


def _envelope(step: int, steps: int) -> float:
    """0..1..0-ish mouth envelope within one phoneme slot."""
    if steps <= 1:
        return 1.0
    t = step / (steps - 1)
    if t < 0.35:
        return t / 0.35
    if t > 0.8:
        return max(0.0, (1.0 - t) / 0.2)
    return 1.0


class MateBridge:
    """Reconnecting client for MtBridge. Thread-safe for senders."""

    def __init__(self, host: str = DEFAULT_HOST, port: Optional[int] = None,
                 connect_timeout: float = 3.0, quiet_window: float = 1.2) -> None:
        self.host = host
        self.port = port if port is not None else vrm_port()
        self.connect_timeout = connect_timeout
        self.quiet_window = quiet_window  # seconds of silence that end a reply
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._closed = False

    # -- link management ---------------------------------------------------

    def is_connected(self) -> bool:
        return self._sock is not None

    def connect(self, retries: int = 0, delay: float = 1.0) -> bool:
        self.close()
        for attempt in range(retries + 1):
            if self._closed:
                return False
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=self.connect_timeout)
                sock.settimeout(self.quiet_window)
                with self._lock:
                    self._sock = sock
                log.info("MtBridge connected on %s:%d", self.host, self.port)
                return True
            except OSError as exc:
                if attempt >= retries:
                    log.info("MtBridge not reachable (%s:%d): %s",
                             self.host, self.port, exc)
                    return False
                time.sleep(delay)
        return False

    def close(self) -> None:
        with self._lock:
            sock = self._sock
            self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def shutdown(self) -> None:
        self._closed = True
        self.close()

    def ensure_connected(self) -> None:
        if not self.is_connected():
            self.connect()
        if not self.is_connected():
            raise BridgeNotConnected(
                f"MateEngine bridge unreachable at {self.host}:{self.port}")

    # -- raw command -------------------------------------------------------

    def command(self, line: str, expect_quiet: Optional[float] = None) -> List[str]:
        """Send one line and collect replies until a quiet window passes."""
        self.ensure_connected()
        window = self.quiet_window if expect_quiet is None else expect_quiet
        deadline_guard = 8.0
        with self._lock:
            sock = self._sock
            if sock is None:
                raise BridgeNotConnected("link lost")
            sock.settimeout(window)
            try:
                sock.sendall((line.rstrip("\r\n") + "\n").encode("utf-8"))
            except OSError as exc:
                self.close()
                raise BridgeNotConnected(f"send failed: {exc}") from exc
            buf = b""
            started = time.time()
            while time.time() - started < deadline_guard:
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    break
                except OSError as exc:
                    self.close()
                    raise BridgeNotConnected(f"recv failed: {exc}") from exc
                if not chunk:
                    self.close()
                    raise BridgeNotConnected("bridge closed the link")
                buf += chunk
                # Heuristic: replies finish when their last line looks
                # terminal (applied/reset/pong/err/avinfo/show/load/...).
                tail = buf[-200:].split(b"\n")[-1]
                terminal = (
                    tail.startswith(b"applied ")
                    or tail.startswith(b"reset ")
                    or tail.startswith(b"pong")
                    or tail.startswith(b"err ")
                    or tail.startswith(b"avinfo ")
                    or tail.startswith(b"show ")
                    or tail.startswith(b"load queued")
                )
                if terminal and buf.endswith(b"\n"):
                    break
            text = buf.decode("utf-8", errors="replace")
            return [ln for ln in text.split("\n") if ln.strip()]

    # -- typed helpers -----------------------------------------------------

    def ping(self) -> bool:
        reply = self.command("ping")
        return any(r.strip() == "pong" for r in reply)

    def sys(self) -> str:
        for r in self.command("sys"):
            if r.startswith("sys datapath "):
                return r[len("sys datapath "):]
        return ""

    def load(self, path: str) -> bool:
        reply = self.command("load " + path, expect_quiet=2.0)
        return any(r.startswith("load queued") for r in reply)

    def show(self) -> str:
        reply = self.command("show", expect_quiet=2.0)
        for r in reply:
            if r.startswith("show ") or r.startswith("show err"):
                return r
        return ""

    @staticmethod
    def _parse_probe(line: str) -> Dict[str, Any]:
        fields = line.split()
        info: Dict[str, Any] = {"shapes": {}}
        for f in fields:
            if f.startswith("renderer="):
                info["renderer"] = f[len("renderer="):]
            elif f.startswith("root="):
                info["root"] = f[len("root="):]
            elif f.startswith("active="):
                info["active"] = f[len("active="):] == "True"
            elif f.startswith("shapes="):
                info["count"] = int(f[len("shapes="):])
        seen_shapes = False
        for f in fields:
            if seen_shapes:
                idx, sep, name = f.partition(":")
                if sep and idx.isdigit():
                    info["shapes"][int(idx)] = name
            elif f.startswith("shapes="):
                seen_shapes = True
        return info

    def probe(self, include_inactive: bool = False) -> Dict[str, Any]:
        reply = self.command("probeall" if include_inactive else "probe",
                             expect_quiet=2.5)
        renderers: List[Dict[str, Any]] = []
        statics: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {}
        for r in reply:
            if r.startswith("probeall total="):
                for part in r.split():
                    for key in ("total", "active", "statics", "staticsActive"):
                        if part.startswith(key + "="):
                            try:
                                meta[key] = int(part[len(key) + 1:])
                            except ValueError:
                                pass
            elif r.startswith("probe renderer="):
                renderers.append(self._parse_probe(r))
            elif r.startswith("probeall renderer="):
                item = self._parse_probe(
                    r.replace("probeall renderer=", "probe renderer=", 1))
                renderers.append(item)
            elif r.startswith("probeall static="):
                fields = r.split()
                entry: Dict[str, Any] = {}
                for f in fields:
                    if f.startswith("static="):
                        entry["name"] = f[len("static="):]
                    elif f.startswith("root="):
                        entry["root"] = f[len("root="):]
                    elif f.startswith("mesh="):
                        entry["mesh"] = f[len("mesh="):]
                    elif f.startswith("active="):
                        entry["active"] = f[len("active="):] == "True"
                statics.append(entry)
        return {"meta": meta, "renderers": renderers, "statics": statics}

    def avinfo(self) -> Dict[str, str]:
        reply = self.command("avinfo", expect_quiet=2.0)
        out: Dict[str, str] = {}
        for r in reply:
            if r.startswith("avinfo err:") :
                out["error"] = r
            elif r.startswith("avinfo "):
                for part in r.split()[1:]:
                    k, _, v = part.partition("=")
                    out[k] = v
        out.setdefault("raw", next(iter(reply), ""))
        return out

    def reset(self) -> bool:
        return any(r.startswith("reset done") for r in
                   self.command("reset", expect_quiet=2.0))

    def reset_all(self) -> bool:
        """Zero every blendshape (Alice-style kana names included)."""
        return any(r.startswith("reset done") for r in
                   self.command("reset all", expect_quiet=2.0))

    def set_expression(self, shape_name: str, weight: float) -> bool:
        """Drive one blendshape by mesh name on every renderer that has it.

        Raw renderer writes get clobbered by the engine's own facial driver;
        prefer set_engine_preset for anything visible.
        """
        weight = max(0.0, min(1.0, float(weight)))
        reply = self.command(
            "preset " + shape_name + " " + _num(weight), expect_quiet=2.0)
        for r in reply:
            if r.startswith("applied ") and " hits=" in r:
                return not r.endswith("hits=0")
        return False

    def set_engine_preset(self, preset: str, weight: float) -> bool:
        """Drive an expression through the engine's Vrm10Runtime layer.

        preset: aa/ih/ou/ee/oh/blink/blinkLeft/blinkRight/happy/angry/sad/
        surprised/relaxed/neutral/lookUp/lookDown/lookLeft/lookRight, or
        custom:<name> for a custom expression key. The bridge keeps a sticky
        per-key overlay so the engine's per-frame rewrites can't erase it.
        """
        weight = max(0.0, min(1.0, float(weight)))
        reply = self.command(
            "exp " + preset + " " + _num(weight), expect_quiet=2.0)
        for r in reply:
            if r.startswith("exp done"):
                return True
            if r.startswith("exp err"):
                log.warning("exp %s failed: %s", preset, r)
                return False
        return False

    def take_over_face(self) -> bool:
        """Enable the sticky overlay and stop the engine's facial driver."""
        reply = self.command("lock 1", expect_quiet=2.0)
        locked = any(r.startswith("lock done") for r in reply)
        # zero residual engine-written weights right after taking over
        self.command("ereset", expect_quiet=3.0)
        return locked

    def release_face(self) -> None:
        """Hand facial control back to the engine."""
        self.command("ereset", expect_quiet=3.0)
        self.command("lock 0", expect_quiet=2.0)

    def query_expressions(self) -> Dict[str, float]:
        """Read back the engine's live expression weights."""
        reply = self.command("equery", expect_quiet=2.5)
        out: Dict[str, float] = {}
        for r in reply:
            if r.startswith("equery "):
                for part in r.split()[1:]:
                    k, _, v = part.partition("=")
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
        return out

    def set_vrm_preset(self, preset: str, weight: float,
                       shape_map: Optional[Dict[str, str]] = None) -> bool:
        name = (shape_map or {}).get(preset, preset)
        return self.set_expression(name, weight)

    # -- lip-sync (engine-layer expressions) --------------------------------

    def clear_mouth(self, shape_map: Optional[Dict[str, str]] = None) -> None:
        for p in ("aa", "ih", "ou", "ee", "oh"):
            self.set_engine_preset(p, 0.0)

    def speak_visemes(self, phonemes: Sequence[str],
                      seconds_per_phoneme: float = 0.115,
                      shape_map: Optional[Dict[str, str]] = None) -> None:
        """Stream a viseme timeline through the engine's expression layer.

        writes persist until the next write of the same key, so each phoneme
        is one boundary-triggered `exp` write rather than a per-frame flood.
        Audio stays on the Python side; timing is a simple time-coupling.
        """
        if not phonemes:
            return
        self.command("lock 1", expect_quiet=2.0)
        try:
            prev: Optional[str] = None
            for i, ph in enumerate(phonemes):
                vis = (PHONEME_TO_VISEME.get(ph.lower())
                       or PHONEME_TO_VISEME.get(ph))
                if vis is None:
                    continue
                if prev is not None and vis != prev:
                    self.set_engine_preset(prev, 0.0)
                self.set_engine_preset(vis, 0.9)
                prev = vis
                time.sleep(seconds_per_phoneme)
            if prev is not None:
                self.set_engine_preset(prev, 0.0)
        except (OSError, BridgeNotConnected) as exc:
            log.warning("viseme stream interrupted: %s", exc)

    def speak_timeline(self, events: Sequence[Tuple[Optional[str], float,
                                                     float]]) -> None:
        """Play a precomputed mouth timeline over a dedicated fire-and-forget link.

        The engine acks every line within ~10 ms and drains a whole send
        batch in one pump tick (probe_pump.py, 2026-11-06), so the old
        per-event command() round-trips -- each billed a full quiet window
        client-side -- were pure self-inflicted latency. The timeline is
        instead scheduled on the client clock against a private raw socket:

          - "lock 1" first (sticky overlay; the caller is responsible for
            take_over_face() beforehand).
          - each viseme change is one "exp <key> <w>" write; identical
            shapes are only slept over (engine overlay stays sticky).
          - sleeps use absolute deadlines so timer jitter does not
            accumulate slot-to-slot.
          - a daemon reader drains replies off the socket (keeps the TCP
            buffers clear); on close it is joined briefly.

        The timeline ends with every written mouth weight back at 0, leaving
        the face owned as usual.
        """
        events = [e for e in events if len(e) == 3 and e[2] > 0.004]
        if not events:
            return
        # Private link: never share a socket with command()-driven callers.
        try:
            raw = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout)
        except OSError as exc:
            log.warning("mouth timeline: connect failed: %s", exc)
            return
        raw.settimeout(0.05)
        reply_count = {"n": 0}
        err_lines = []

        def _drain():
            buf = b""
            while True:
                try:
                    chunk = raw.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line_b, buf = buf.split(b"\n", 1)
                    text = line_b.decode("utf-8", errors="replace").strip()
                    if text:
                        reply_count["n"] += 1
                        if text.startswith("exp err"):
                            err_lines.append(text)
            if buf.strip():
                text = buf.decode("utf-8", errors="replace").strip()
                if text:
                    reply_count["n"] += 1
                    if text.startswith("exp err"):
                        err_lines.append(text)

        reader = threading.Thread(target=_drain, daemon=True)
        reader.start()
        try:
            # Arm the sticky overlay and let the engine settle one pump tick.
            raw.sendall(b"lock 1\n")
            time.sleep(0.01)
            t_start = time.perf_counter()
            t = t_start
            prev = None
            for preset, weight, secs in events:
                slot = max(0.004, float(secs))
                target = t + slot
                if preset is None or weight <= 0.0:
                    if prev is not None:
                        raw.sendall(
                            ("exp %s 0\n" % prev).encode("utf-8"))
                        prev = None
                else:
                    if preset != prev:
                        if prev is not None:
                            raw.sendall(
                                ("exp %s 0\n" % prev).encode("utf-8"))
                        raw.sendall(
                            ("exp %s %s\n"
                             % (preset, _num(float(weight)))).encode("utf-8"))
                        prev = preset
                # Absolute-deadline sleep: overshoot stays inside one slot.
                while True:
                    remain = target - time.perf_counter()
                    if remain <= 0.0:
                        break
                    time.sleep(min(remain, 0.25))
                t = target
            if prev is not None:
                raw.sendall(("exp %s 0\n" % prev).encode("utf-8"))
                prev = None
            # Let the reader collect the trailing acks before teardown.
            time.sleep(0.15)
        except (OSError, BridgeNotConnected) as exc:
            log.warning("mouth timeline interrupted: %s", exc)
        finally:
            try:
                raw.close()
            except OSError:
                pass
            reader.join(timeout=0.5)
        if err_lines:
            log.warning("mouth timeline exp errors: %s", err_lines[:5])
        log.debug(
            "mouth timeline: %d slots, %.2f s span, %d replies",
            len(events), time.perf_counter() - t_start, reply_count["n"])

