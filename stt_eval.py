"""
stt_eval.py
===========
Offline STT evaluation. Scores candidate engines x preprocessing modes on
the captured voice clips (stt_eval/clips/) and writes a report.

Run via: python manage.py eval-stt [--beam N] [--modes off loudnorm ...]

The candidate matrix is 4 engines x 4 preprocessing modes. Each candidate
is loaded once and run over every clip; models are unloaded between
candidates. WER is computed only over clips present in refs.csv (human-
corrected references); uncorrected clips still get latency/disagreement
stats.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

import config
from voice import preprocess, stt

# (label, config overrides applied before loading the candidate)
_CANDIDATES = [
    (
        "moonshine-base",
        {
            "VOICE_STT_ENGINE": "moonshine",
            "VOICE_STT_MOONSHINE_EN": "UsefulSensors/moonshine-base",
            "VOICE_STT_COMPUTE_TYPE": "float16",
        },
    ),
    (
        "moonshine-streaming-medium",
        {
            "VOICE_STT_ENGINE": "moonshine",
            "VOICE_STT_MOONSHINE_EN": "UsefulSensors/moonshine-streaming-medium",
            "VOICE_STT_COMPUTE_TYPE": "float16",
        },
    ),
    (
        "whisper-medium",
        {
            "VOICE_STT_ENGINE": "whisper",
            "VOICE_STT_MODEL": "medium",
            "VOICE_STT_COMPUTE_TYPE": "int8_float16",
        },
    ),
    (
        "whisper-turbo",
        {
            "VOICE_STT_ENGINE": "whisper",
            "VOICE_STT_MODEL": "large-v3-turbo",
            "VOICE_STT_COMPUTE_TYPE": "int8_float16",
        },
    ),
]

_MODES = ["off", "loudnorm", "denoise", "both"]


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _wer(ref: str, hyp: str) -> float:
    """Word-level Levenshtein distance normalized by reference length."""
    r = _normalize(ref).split()
    h = _normalize(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def _load_clips(eval_dir: Path) -> list[tuple[Path, dict]]:
    clips = []
    for wav in sorted(eval_dir.glob("clip_*.wav")):
        side = eval_dir / f"{wav.stem}.json"
        meta = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
        clips.append((wav, meta))
    return clips


def _load_refs(eval_dir: Path) -> dict[str, str]:
    refs = {}
    refs_path = eval_dir / "refs.csv"
    if refs_path.exists():
        with open(refs_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                refs[row["clip"]] = row["text"]
    return refs


def _ensure_refs(eval_dir: Path, clips: list[tuple[Path, dict]]) -> None:
    refs_path = eval_dir / "refs.csv"
    if refs_path.exists():
        return
    with open(refs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "text"])
        for wav, meta in clips:
            w.writerow([wav.stem, meta.get("text", "")])
    print(f"Wrote {refs_path} -- correct the 'text' column by listening, then re-run.")


def _vram_mb() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0.0


def _set_config(overrides: dict) -> None:
    for k, v in overrides.items():
        setattr(config, k, v)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(str(path))
    audio = data.astype(np.float32)
    if data.dtype == np.int16:
        audio = audio / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _run_candidate(
    label: str,
    overrides: dict,
    clips: list[tuple[Path, dict]],
    modes: list[str],
    beam: int,
) -> list[dict]:
    """Load one candidate, run all clips x modes, return per-clip rows."""
    _set_config(overrides)
    stt._WHISPER_BEAM = beam
    stt.unload()

    t0 = time.monotonic()
    if overrides.get("VOICE_STT_ENGINE") == "moonshine":
        stt.preload()
    else:
        stt.preload()
    load_s = time.monotonic() - t0
    vram = _vram_mb()
    print(f"  [{label}] loaded in {load_s:.1f}s, VRAM={vram:.0f}MB", flush=True)

    rows = []
    for wav, meta in clips:
        audio, sr = _read_wav(wav)
        for mode in modes:
            t1 = time.monotonic()
            audio_16k = preprocess.prepare(audio, sr, mode)
            result = stt.transcribe(audio_16k, 16000, language="en")
            lat = (time.monotonic() - t1) * 1000
            rows.append({
                "clip": wav.stem,
                "candidate": label,
                "mode": mode,
                "ref": meta.get("text", ""),
                "hyp": result.get("text", ""),
                "latency_ms": round(lat, 1),
                "noise_floor_rms": meta.get("noise_floor_rms", 0.0),
                "vram_mb": round(vram, 0),
            })
        print(f"    {wav.stem} done", flush=True)

    stt.unload()
    return rows


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((pct / 100.0) * (len(s) - 1)))
    return s[k]


def _write_report(eval_dir: Path, rows: list[dict], refs: dict[str, str]) -> None:
    # Aggregate per (candidate, mode)
    agg: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["candidate"], r["mode"])
        a = agg.setdefault(key, {"lats": [], "wers": [], "disagree": 0, "n": 0, "vram": r["vram_mb"]})
        a["lats"].append(r["latency_ms"])
        a["n"] += 1
        if r["clip"] in refs and refs[r["clip"]].strip():
            a["wers"].append(_wer(refs[r["clip"]], r["hyp"]))
        if r["hyp"] != r["ref"]:
            a["disagree"] += 1

    # Baseline = moonshine-base / off for disagreement reference
    baseline = {(r["clip"]): r["hyp"] for r in rows if r["candidate"] == "moonshine-base" and r["mode"] == "off"}

    lines = [
        "# STT Eval Report",
        "",
        f"Candidates: {len(set(r['candidate'] for r in rows))} | Modes: {len(set(r['mode'] for r in rows))} | Clips: {len(set(r['clip'] for r in rows))}",
        f"Corrected refs: {len(refs)}",
        "",
        "| candidate | mode | WER | p50 ms | p95 ms | disagree vs base | VRAM MB |",
        "|---|---|---|---|---|---|---|",
    ]
    csv_rows = []
    for (cand, mode), a in sorted(agg.items()):
        wer = statistics.mean(a["wers"]) if a["wers"] else float("nan")
        p50 = _percentile(a["lats"], 50)
        p95 = _percentile(a["lats"], 95)
        lines.append(
            f"| {cand} | {mode} | {wer:.3f} | {p50:.0f} | {p95:.0f} | {a['disagree']}/{a['n']} | {a['vram']:.0f} |"
        )
        csv_rows.append({
            "candidate": cand, "mode": mode, "wer": f"{wer:.4f}",
            "p50_ms": f"{p50:.0f}", "p95_ms": f"{p95:.0f}",
            "disagree": a["disagree"], "n": a["n"], "vram_mb": f"{a['vram']:.0f}",
        })

    # Worst clips (highest WER) for listening
    lines += ["", "## Worst clips (by WER, corrected refs only)", ""]
    worst = []
    for r in rows:
        if r["clip"] in refs and refs[r["clip"]].strip():
            w = _wer(refs[r["clip"]], r["hyp"])
            worst.append((w, r["clip"], r["candidate"], r["mode"], r["hyp"]))
    worst.sort(reverse=True)
    for w, clip, cand, mode, hyp in worst[:15]:
        lines.append(f"- {clip} [{cand}/{mode}] WER={w:.2f}: {hyp!r}")

    (eval_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(eval_dir / "report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["candidate", "mode", "wer", "p50_ms", "p95_ms", "disagree", "n", "vram_mb"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nWrote {eval_dir / 'report.md'} and report.csv")


def run(eval_dir: Path, beam: int, modes: list[str], candidates: list[str] | None = None,
        clips: list[str] | None = None) -> None:
    all_clips = _load_clips(eval_dir)
    if not all_clips:
        print(f"No clips found in {eval_dir}")
        return
    if clips:
        all_clips = [(w, m) for w, m in all_clips if w.stem in clips]
    print(f"Found {len(all_clips)} clips")
    _ensure_refs(eval_dir, clips)
    refs = _load_refs(eval_dir)

    cand_list = _CANDIDATES
    if candidates:
        cand_list = [(l, o) for l, o in _CANDIDATES if l in candidates]
    rows = []
    for label, overrides in cand_list:
        rows.extend(_run_candidate(label, overrides, all_clips, modes, beam))

    _write_report(eval_dir, rows, refs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline STT evaluation")
    parser.add_argument("--beam", type=int, default=3, help="Whisper beam size")
    parser.add_argument("--modes", nargs="+", default=_MODES, choices=_MODES)
    parser.add_argument("--candidates", nargs="+", default=None,
                        help="Subset of candidates to run")
    parser.add_argument("--clips", nargs="+", default=None,
                        help="Subset of clip stems to score")
    args = parser.parse_args()
    run(Path(config.STT_EVAL_DIR), args.beam, args.modes, args.candidates, args.clips)


if __name__ == "__main__":
    main()
