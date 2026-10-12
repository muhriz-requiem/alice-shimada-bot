"""
tests/test_stt_eval.py
======================
Covers the pure helpers in stt_eval.py (the manage.py eval-stt runner):
WER scoring, normalization, refs.csv handling, wav reading, percentiles.

Model loading/transcription is exercised live, not here.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import scipy.io.wavfile as wavfile

import stt_eval


def test_wer_identical_is_zero():
    assert stt_eval._wer("hello world", "hello world") == 0.0


def test_wer_completely_different():
    assert stt_eval._wer("a b c d", "w x y z") == 1.0


def test_wer_partial():
    # 1 substitution out of 4 words
    assert stt_eval._wer("a b c d", "a x c d") == 0.25


def test_wer_empty_ref():
    assert stt_eval._wer("", "") == 0.0
    assert stt_eval._wer("", "something") == 1.0


def test_normalize_case_and_whitespace():
    assert stt_eval._normalize("  Hello   World ") == "hello world"


def test_ensure_refs_creates_csv(tmp_path):
    (tmp_path / "clip_1.wav").write_bytes(b"x")
    (tmp_path / "clip_1.json").write_text(
        json.dumps({"text": "hello there"}), encoding="utf-8"
    )
    clips = [(tmp_path / "clip_1.wav", {"text": "hello there"})]
    stt_eval._ensure_refs(tmp_path, clips)

    refs = stt_eval._load_refs(tmp_path)
    assert refs == {"clip_1": "hello there"}


def test_ensure_refs_does_not_overwrite(tmp_path):
    refs_path = tmp_path / "refs.csv"
    refs_path.write_text("clip,text\nclip_1,corrected\n", encoding="utf-8")
    clips = [(tmp_path / "clip_1.wav", {"text": "machine"})]
    stt_eval._ensure_refs(tmp_path, clips)

    refs = stt_eval._load_refs(tmp_path)
    assert refs == {"clip_1": "corrected"}


def test_read_wav(tmp_path):
    sr = 48000
    audio = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 32767).astype(np.int16)
    wavfile.write(str(tmp_path / "t.wav"), sr, audio)
    out, out_sr = stt_eval._read_wav(tmp_path / "t.wav")
    assert out_sr == sr
    assert out.dtype == np.float32
    assert len(out) == sr


def test_percentile():
    assert stt_eval._percentile([1, 2, 3, 4, 5], 50) == 3
    assert stt_eval._percentile([1, 2, 3, 4, 5], 95) == 5
    assert stt_eval._percentile([], 50) == 0.0
