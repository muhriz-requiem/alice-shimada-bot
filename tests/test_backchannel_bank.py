"""
Tests for voice.backchannel.BackchannelBank: the prefetched bank of
spoken acknowledgments ("Hm?" address-acks + backchannels) used by the
live pipeline.
"""

from __future__ import annotations

import asyncio

import numpy as np

from voice.backchannel import (
    _MAX_BANKED_SECONDS,
    ACK_EN,
    ACK_JA,
    BACKCHANNELS_JA,
    BackchannelBank,
)


def _wav_bytes_for(audio: np.ndarray, sr: int = 16000) -> bytes:
    from voice import tts

    return tts.array_to_wav_bytes(audio, sample_rate=sr)


def _patch_tts(monkeypatch, behavior):
    """behavior: phrase -> (wav_bytes, sr, engine) or Exception."""
    async def fake_generate_tts(phrase: str):
        out = behavior(phrase)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr("voice.tts.generate_tts", fake_generate_tts)
    monkeypatch.setattr("voice.rvc.is_available", lambda: False)


class TestBankConstruction:
    def test_not_ready_before_build(self):
        bank = BackchannelBank(size_per_lang=4)
        assert not bank.ready
        assert bank.pick_ack("ja") is None

    def test_build_synthesizes_all_pools(self, monkeypatch):
        audio = np.zeros(3200, dtype=np.float32)  # 0.2s @16k
        seen: list[str] = []
        _patch_tts(
            monkeypatch,
            lambda phrase: (seen.append(phrase), _wav_bytes_for(audio), 16000, "edge")[1:],
        )
        bank = BackchannelBank(size_per_lang=4)
        asyncio.run(bank.build())
        assert bank.ready
        # Address-acks and regular backchannels each banked
        ja_ack = bank.pick_ack("ja")
        assert ja_ack is not None
        _, _, wav = ja_ack
        assert isinstance(wav, bytes) and wav
        # ...and phrases from both banks were requested
        assert any(p in ACK_JA for p in seen)
        assert any(p in BACKCHANNELS_JA for p in seen)

    def test_built_entries_land_in_both_stores(self, monkeypatch):
        audio = np.zeros(3200, dtype=np.float32)
        _patch_tts(
            monkeypatch,
            lambda phrase: (_wav_bytes_for(audio), 16000, "edge"),
        )
        bank = BackchannelBank(size_per_lang=2)
        asyncio.run(bank.build())
        stats = bank.stats()
        assert stats["built"]
        assert sum(stats["acks"].values()) >= 2
        assert sum(stats["backchannels"].values()) >= 2

    def test_lang_fallback(self, monkeypatch):
        audio = np.zeros(3200, dtype=np.float32)
        _patch_tts(monkeypatch, lambda phrase: (_wav_bytes_for(audio), 16000, "edge"))
        bank = BackchannelBank(size_per_lang=2)
        asyncio.run(bank.build())
        assert bank.pick_ack("fr") is not None  # falls back to any bank

    def test_pool_tolerates_failures(self, monkeypatch):
        audio = np.zeros(3200, dtype=np.float32)
        failing = BACKCHANNELS_JA[0]

        def behavior(phrase: str):
            if phrase == failing:
                return RuntimeError("TTS down")
            return (_wav_bytes_for(audio), 16000, "edge")

        _patch_tts(monkeypatch, behavior)
        bank = BackchannelBank(size_per_lang=4)
        asyncio.run(bank.build())
        stats = bank.stats()
        assert stats["built"]
        # The failed phrase did not take a slot from the bank (entries
        # only append on success)
        assert sum(stats["backchannels"].values()) >= 3

    def test_long_clips_skipped(self, monkeypatch):
        long_audio = np.zeros(int(16000 * (_MAX_BANKED_SECONDS + 1)), dtype=np.float32)
        short_audio = np.zeros(1600, dtype=np.float32)
        calls: list[str] = []

        def behavior(phrase: str):
            calls.append(phrase)
            wav = _wav_bytes_for(long_audio) if phrase == ACK_JA[0] else _wav_bytes_for(short_audio)
            return (wav, 16000, "edge")

        _patch_tts(monkeypatch, behavior)
        bank = BackchannelBank(size_per_lang=2)
        asyncio.run(bank.build())
        picked = bank.pick_ack("ja")
        assert picked is not None
        audio, sr, _ = picked
        assert len(audio) / sr <= _MAX_BANKED_SECONDS


class TestBankPools:
    def test_ack_pools_are_short(self):
        for phrase in ACK_JA + ACK_EN:
            assert len(phrase) <= 6

    def test_pool_cycles_to_size(self):
        bank = BackchannelBank(size_per_lang=10)
        pool = bank._pool("ja", BACKCHANNELS_JA)
        assert len(pool) == 10

    def test_pool_empty_when_disabled(self):
        bank = BackchannelBank(size_per_lang=0)
        assert bank._pool("ja", BACKCHANNELS_JA) == []
