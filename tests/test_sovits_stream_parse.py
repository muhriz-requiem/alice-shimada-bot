"""
Tests for voice.sovits._WavStreamParser: the incremental RIFF/WAVE
parser for GPT-SoVITS streaming responses. Contract: feeding the same
audio split at arbitrary byte boundaries yields the identical float32
mono sample array as decoding the WAV with the stdlib wave module.
"""

from __future__ import annotations

import io
import wave

import numpy as np

from voice.sovits import _WavStreamParser


def _reference_decode(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr


def _make_wav(samples: np.ndarray, sr: int, sampwidth: int = 2, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        frames = samples
        if channels > 1:
            frames = np.tile(samples[:, None], (1, channels))
        pcm = (frames * 32767.0).astype(np.int16) if sampwidth == 2 else (
            (frames * 2147483647.0).astype(np.int32)
        )
        if channels > 1:
            pcm = pcm.reshape(-1)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _collect(parser, wav: bytes, chunk_sizes=None) -> np.ndarray:
    out: list[np.ndarray] = []
    if chunk_sizes is None:
        for audio in parser.feed(wav):
            out.append(audio)
    else:
        pos = 0
        sizes = list(chunk_sizes)
        i = 0
        while pos < len(wav):
            size = sizes[i % len(sizes)] or 1
            i += 1
            piece = wav[pos:pos + size]
            pos += size
            for audio in parser.feed(piece):
                out.append(audio)
    for audio in parser.flush():
        out.append(audio)
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


class TestFullParse:
    def test_mono_16bit_whole(self):
        rng = np.random.default_rng(7)
        samples = rng.uniform(-0.5, 0.5, 1000).astype(np.float32)
        wav = _make_wav(samples, 32000)
        ref, sr = _reference_decode(wav)
        assert sr == 32000
        parser = _WavStreamParser()
        got = _collect(parser, wav)
        assert np.allclose(got, ref)

    def test_odd_byte_splitting(self):
        rng = np.random.default_rng(11)
        samples = rng.uniform(-0.3, 0.3, 512).astype(np.float32)
        wav = _make_wav(samples, 32000)
        ref, _ = _reference_decode(wav)
        parser = _WavStreamParser()
        got = _collect(parser, wav, chunk_sizes=[1, 3, 7, 1])
        assert np.allclose(got, ref)

    def test_random_byte_splitting(self):
        rng = np.random.default_rng(3)
        samples = rng.uniform(-0.4, 0.4, 777).astype(np.float32)
        wav = _make_wav(samples, 32000)
        ref, _ = _reference_decode(wav)
        sizes = [rng.integers(1, 40).item() for _ in range(64)]
        parser = _WavStreamParser()
        got = _collect(parser, wav, chunk_sizes=sizes)
        assert np.allclose(got, ref)

    def test_header_split_mid_chunk(self):
        # Byte-level cut inside the fmt chunk must be tolerated
        rng = np.random.default_rng(5)
        samples = rng.uniform(-0.5, 0.5, 256).astype(np.float32)
        wav = _make_wav(samples, 32000)
        ref, _ = _reference_decode(wav)
        cut = wav.index(b"fmt ") + 5  # inside the fmt chunk
        parser = _WavStreamParser()
        got = _collect(parser, wav, chunk_sizes=[cut, 13])
        assert np.allclose(got, ref)

    def test_stereo_averaged_to_mono(self):
        rng = np.random.default_rng(9)
        samples = rng.uniform(-0.5, 0.5, 300).astype(np.float32)
        wav = _make_wav(samples, 24000, channels=2)
        ref, _ = _reference_decode(wav)
        parser = _WavStreamParser()
        got = _collect(parser, wav, chunk_sizes=[3, 11, 5])
        assert np.allclose(got, ref)

    def test_int32_samples(self):
        rng = np.random.default_rng(13)
        samples = rng.uniform(-0.5, 0.5, 300).astype(np.float32)
        wav = _make_wav(samples, 24000, sampwidth=4)
        ref, _ = _reference_decode(wav)
        parser = _WavStreamParser()
        got = _collect(parser, wav, chunk_sizes=[2, 9, 4])
        assert np.allclose(got, ref, atol=1e-6)


class TestRobustness:
    def test_empty_feed(self):
        parser = _WavStreamParser()
        assert parser.feed(b"") == []
        assert parser.flush() == []

    def test_not_wav_raises(self):
        parser = _WavStreamParser()
        try:
            parser.feed(b"OGG!notawavefmt......")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_truncated_header_no_data(self):
        parser = _WavStreamParser()
        assert parser.feed(b"RIFF") == []
        assert parser.feed(b"\x24\x00\x00\x00WAVE") == []

    def test_odd_trailing_byte_dropped(self):
        rng = np.random.default_rng(17)
        samples = rng.uniform(-0.3, 0.3, 100).astype(np.float32)
        wav = _make_wav(samples, 32000)
        ref, _ = _reference_decode(wav)
        parser = _WavStreamParser()
        got = _collect(parser, wav + b"\xff")  # stray trailing byte
        assert got.shape[0] == ref.shape[0]
        assert np.allclose(got, ref)
