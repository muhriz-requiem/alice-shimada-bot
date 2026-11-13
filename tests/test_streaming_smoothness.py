"""
Tests for streamed-TTS smoothing: sentence merging in the worker
(fewer synthesis gaps between short sentences) and the desktop
SdOutput pre-roll (hold the device start until a cushion is buffered).
"""
import asyncio
import queue
import threading

import numpy as np

import voice.readiness
from voice import streaming_tts
from voice.streaming_source import StreamingAudioSource

# --- worker sentence merge ---------------------------------------------------

async def _run_stream(text: str, calls: list) -> StreamingAudioSource:
    async def fake_synth(sentence, guild_id, source, mood, use_sovits_stream=False):
        calls.append((sentence, use_sovits_stream))
        await asyncio.sleep(0.005)
        return 0.005

    async def deltas():
        yield text

    original = streaming_tts._synthesize_and_add
    streaming_tts._synthesize_and_add = fake_synth
    try:
        source, spoken, settled = await streaming_tts.stream_response_deltas(
            deltas(), 0
        )
        # Let the background worker finish adding remaining sentences.
        for _ in range(50):
            if source.is_done:
                break
            await asyncio.sleep(0.01)
        return source
    finally:
        streaming_tts._synthesize_and_add = original


def test_worker_merges_already_arrived_sentences(monkeypatch):
    monkeypatch.setattr(voice.readiness, "is_ready", lambda: True)
    calls: list = []
    source = asyncio.run(
        _run_stream("Hi Father. I had a sandwich today. It was good. Thanks for asking.", calls)
    )

    assert source.is_done
    # First sentence streams alone (the TTFA lever, SoVITS streaming);
    # everything already arrived after it merges into ONE synthesis job
    # instead of three -> two calls, not four.
    assert len(calls) == 2, f"expected merged synthesis, got {calls}"
    assert calls[0] == ("Hi Father.", True)
    assert calls[1] == (
        "I had a sandwich today. It was good. Thanks for asking.", False
    )


# --- SdOutput pre-roll ---------------------------------------------------------

def _make_output(preroll_samples: int) -> object:
    from desktop.local_client import SdOutput

    out = SdOutput.__new__(SdOutput)  # no real sounddevice stream
    out.playing = True
    out._q = queue.SimpleQueue()
    out._buf = np.zeros(0, dtype=np.float32)
    out._end_seen = False
    out._drained = threading.Event()
    out._rms_callback = None
    out._far_end_callback = None
    out._preroll_samples = preroll_samples
    out._drain_started = preroll_samples == 0
    return out


def _stereo_frame(amplitude: int, n: int = 960) -> bytes:
    arr = np.full(n * 2, amplitude, dtype=np.int16)
    return arr.tobytes()


def test_sdoutput_preroll_holds_then_releases():

    out = _make_output(preroll_samples=12000)  # 250ms @ 48kHz
    outdata = np.zeros((480, 1), dtype=np.float32)

    # Below the cushion: silent, nothing consumed.
    out._q.put(_stereo_frame(16000))
    out._audio_callback(outdata, 480, None, None)
    assert not out._drain_started
    assert np.all(outdata == 0)

    # Fill past the cushion: playback starts, real samples emitted.
    for _ in range(14):
        out._q.put(_stereo_frame(16000))
    out._audio_callback(outdata, 480, None, None)
    assert out._drain_started
    assert np.allclose(outdata[:, 0], 16000 / 32768, atol=1e-6)


def test_sdoutput_preroll_short_reply_no_deadlock():
    # A reply shorter than the cushion must still play (and finish):
    # the stream's end releases the hold.
    from desktop import local_client as lc

    out = _make_output(preroll_samples=12000)
    outdata = np.zeros((480, 1), dtype=np.float32)

    out._q.put(_stereo_frame(16000))
    out._q.put(_stereo_frame(16000))
    out._q.put(lc._END)  # end-of-stream marker (identity-compared)

    out._audio_callback(outdata, 480, None, None)
    assert out._drain_started                      # released by end-of-stream
    assert np.allclose(outdata[:, 0], 16000 / 32768, atol=1e-6)
    assert len(out._buf) == 1920 - 480             # rest buffered, draining


# --- local player pacing --------------------------------------------------------

def test_local_player_paces_starved_source():
    # The streaming source returns SILENCE frames while synthesis is
    # behind. The local player must consume at ~50 Hz (real time) --
    # unpaced, it queues silence at CPU speed and one starvation burst
    # plays back as seconds of inserted silence (the fragmented voice).
    import time as _time

    from desktop.local_client import LocalVoiceClient, SilentOutput

    class _StarvedSource:
        def __init__(self):
            self.reads = 0

        def read(self):
            self.reads += 1
            return b"\x00" * 3840  # one 20ms silence frame

    out = SilentOutput()
    client = LocalVoiceClient(output_factory=lambda: out)
    src = _StarvedSource()
    client.play(src, after=lambda err: None)
    try:
        _time.sleep(1.0)
    finally:
        client.stop()
    rate = src.reads / 1.0
    assert 25 <= rate <= 90, f"player consumed at {rate:.0f} frames/s (want ~50)"
