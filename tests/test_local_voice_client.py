"""
tests/test_local_voice_client.py
================================
Unit tests for the desktop companion's local playback layer
(desktop.local_client.LocalVoiceClient / SdOutput / SilentOutput) and
the desktop microphone gating logic (desktop.mic_input.MicInput).

discord.AudioSource contract assumed here (mirrors what the pipeline
hands a voice client):
- .read() -> bytes: 20ms of 16-bit 48kHz interleaved stereo PCM
- b"" (or None) at end of stream
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from desktop.local_client import (
    LocalChannel,
    LocalVoiceClient,
    SilentOutput,
    _stereo_to_mono,
)
from desktop.mic_input import MicInput


# --- helpers ---------------------------------------------------------------

def _stereo_frame(left: int, right: int, n_samples: int = 240) -> bytes:
    """Build interleaved stereo s16le bytes with constant L/R values."""
    arr = np.empty(n_samples * 2, dtype=np.int16)
    arr[0::2] = left
    arr[1::2] = right
    return arr.tobytes()


class ScriptSource:
    """Fake discord.AudioSource: canned frames, then EOF."""

    def __init__(self, frames, delay: float = 0.0):
        self._frames = list(frames)
        self._delay = delay
        self.reads = 0
        self._lock = threading.Lock()

    def read(self):
        with self._lock:
            self.reads += 1
            if not self._frames:
                return b""
            if self._delay and len(self._frames) > 1:
                time.sleep(self._delay)
            return self._frames.pop(0)


# --- stereo->mono conversion ------------------------------------------------

def test_stereo_to_mono_values():
    frame = _stereo_frame(left=100, right=300, n_samples=4)
    mono = _stereo_to_mono(frame)
    assert mono is not None
    assert mono.dtype == np.float32
    assert mono.shape == (4,)
    expected = np.float32(200) / 32768.0
    assert np.allclose(mono, expected)


def test_stereo_to_mono_empty():
    assert _stereo_to_mono(b"") is None
    assert _stereo_to_mono(b"\x01") is None


# --- full playback flow ------------------------------------------------------

def _frame_resume(fut: asyncio.Future, e, calls):
    calls.append(e)
    if not fut.done():
        fut.set_result(e)


def test_play_completes_and_fires_after_once():
    out_writes = []

    class RecordingOutput(SilentOutput):
        def put(self, frame):
            out_writes.append(frame)
            super().put(frame)

    # Sizable source with per-read delay so play() is still mid-stream
    # when we sample is_playing() right after starting it.
    frames = [_stereo_frame(1000, 2000) for _ in range(5)]
    src = ScriptSource(frames, delay=0.05)

    async def _run():
        loop = asyncio.get_running_loop()
        client = LocalVoiceClient(guild_id=0, output_factory=RecordingOutput)
        fut = loop.create_future()
        calls = []

        client.play(src, after=lambda e: _frame_resume(fut, e, calls))
        # While the source has data, the client is "playing".
        assert client.is_playing()
        assert await asyncio.wait_for(fut, timeout=5.0) is None

        return calls

    calls = asyncio.run(_run())
    # after(None) fired exactly once, playback no longer flagged running,
    # and every frame reached the output.
    assert calls == [None]
    assert out_writes == frames
    assert not src._frames


def test_play_double_raises_client_exception():
    # A client with ongoing playback must reject play() like discord.py does.
    client = LocalVoiceClient(guild_id=0, output_factory=SilentOutput)
    client._out = SilentOutput()  # playing == True
    with pytest.raises(Exception) as info:
        client.play(ScriptSource([_stereo_frame(1, 1)]))
    assert "Already playing" in str(info.value)


def test_stop_midstream_barge():
    out = SilentOutput()
    frames = [_stereo_frame(n, n) for n in range(12)]
    src = ScriptSource(frames, delay=0.02)  # ~0.22s of reading

    async def _run():
        loop = asyncio.get_running_loop()
        client = LocalVoiceClient(guild_id=0, output_factory=lambda: out)
        fut = loop.create_future()
        calls = []
        client.play(src, after=lambda e: _frame_resume(fut, e, calls))
        await asyncio.sleep(0.05)  # let it start producing
        client.stop()
        assert await asyncio.wait_for(fut, timeout=5.0) is None
        return out.frames_played

    played = asyncio.run(_run())
    # Interrupted: some but not all frames played.
    total = sum(len(f) for f in frames) // 4
    assert 0 < played < total
    assert not out.playing


def test_local_channel_members():
    owner = SimpleNamespace(id=1, bot=False)
    ch = LocalChannel(members=[owner])
    others = any(not m.bot and m.id != owner.id for m in ch.members)
    assert not others  # private-call semantics


def test_simple_output_frames_counted():
    out = SilentOutput()
    out.put(_stereo_frame(1, 1, n_samples=960))  # 3840 bytes
    assert out.frames_played == 960  # 3840 // 4


# --- MicInput gating ----------------------------------------------------------

class _RecordingSink:
    def __init__(self):
        self.written = []

    def write(self, user, data):
        self.written.append((user, data.pcm))


def _make_mic(playing: bool | None, barge_rms=0.12, barge_blocks=3,
              echo_gate: bool = True, agc: bool = False,
              agc_target: float = 0.12, agc_max_gain: float = 32.0,
              apm=None, suppress_fn=None):
    sink = _RecordingSink()
    owner = SimpleNamespace(id=42)
    if playing is None:
        playing_fn = lambda: False
    else:
        playing_fn = lambda: playing

    class _MicNoStream(MicInput):
        # Skip the real OutputStream (no devices in CI); logic is in _handle_block.
        def __init__(self):
            self.sink = sink
            self.owner = owner
            self._playing_fn = playing_fn
            self._device = None
            self.echo_gate = echo_gate
            self.muted = False
            self.barge_rms = barge_rms
            self.barge_blocks = barge_blocks
            self.on_barge = None
            self.agc = agc
            self._agc_target = agc_target
            self._agc_max_gain = agc_max_gain
            self._agc_floor = 0.002
            self._agc_gain = 1.0
            self._agc_logged = False
            self._apm = apm
            self._suppress_fn = suppress_fn
            self._suppress_active = False
            self._suppress_logged = False
            self._burst = 0
            self._last_loud = 0.0
            self.stats = {"blocks": 0, "fed": 0, "gated": 0, "barge_fed": 0,
                          "muted": 0, "suppressed": 0}
            self._owner_stub_key = owner.id

    return _MicNoStream(), sink


def test_mic_feeds_when_not_playing():
    mic, sink = _make_mic(playing=None)
    pcm = _stereo_frame(1, 1)
    for _ in range(5):
        mic._handle_block(rms=0.01, pcm_bytes=pcm)
    assert len(sink.written) == 5
    assert mic.stats["fed"] == 5
    assert mic.stats["gated"] == 0


def test_mic_gated_while_alice_playing():
    mic, sink = _make_mic(playing=True)
    pcm = _stereo_frame(1, 1)
    for _ in range(5):
        mic._handle_block(rms=0.01, pcm_bytes=pcm)
    assert sink.written == []  # speaker echo dropped
    assert mic.stats["gated"] == 5


def test_mic_barge_requires_sustained_loudness():
    mic, sink = _make_mic(playing=True, barge_blocks=3)
    pcm = _stereo_frame(1, 1)
    barges = []
    mic.on_barge = lambda: barges.append(1)

    # 2 consecutive loud blocks: below the burst threshold -> still gated.
    mic._handle_block(rms=0.2, pcm_bytes=pcm)
    mic._handle_block(rms=0.2, pcm_bytes=pcm)
    assert sink.written == []
    assert mic._burst == 2

    # A quiet block resets the burst counter.
    mic._handle_block(rms=0.01, pcm_bytes=pcm)
    assert mic._burst == 0

    # 3 consecutive loud blocks -> the interrupt feed opens (one feed).
    mic._handle_block(rms=0.2, pcm_bytes=pcm)
    mic._handle_block(rms=0.2, pcm_bytes=pcm)
    mic._handle_block(rms=0.2, pcm_bytes=pcm)
    assert len(sink.written) == 1
    assert len(barges) == 1  # fired once at the open moment
    assert mic.stats["barge_fed"] == 1

    # Sustained shouting keeps feeding (no re-fire of on_barge).
    mic._handle_block(rms=0.2, pcm_bytes=pcm)
    mic._handle_block(rms=0.2, pcm_bytes=pcm)
    assert len(sink.written) == 3
    assert len(barges) == 1
    assert mic.stats["barge_fed"] == 3


def test_mic_ungated_headphones_feeds_during_playback():
    # Headphone setups (user's default): the mic physically can't hear
    # Alice's speakers, so gating is unnecessary -- normal-volume speech
    # mid-sentence reaches the VAD and powers real barge-in.
    mic, sink = _make_mic(playing=True, echo_gate=False)
    pcm = _stereo_frame(1, 1)
    for _ in range(5):
        mic._handle_block(rms=0.03, pcm_bytes=pcm)
    assert len(sink.written) == 5
    assert mic.stats["fed"] == 5
    assert mic.stats["gated"] == 0
    assert mic.stats["barge_fed"] == 0


# --- MicInput AGC ---------------------------------------------------------------

def _amp_frame(rms: float, n_samples: int = 960) -> tuple[bytes, int]:
    """Constant-amplitude int16 stereo frame at the given mono RMS."""
    amp = max(1, int(round(rms * 32768.0)))
    return _stereo_frame(amp, amp, n_samples=n_samples), amp


def _written_rms(sink) -> float:
    pcm = np.frombuffer(sink.written[-1][1], dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean((pcm / 32768.0) ** 2)))


def test_mic_agc_boosts_faint_speech_toward_target():
    mic, sink = _make_mic(playing=None, agc=True, agc_target=0.12,
                          agc_max_gain=32.0)
    pcm, _ = _amp_frame(0.005)  # faint far-field speech
    for _ in range(40):
        mic._handle_block(rms=0.005, pcm_bytes=pcm)
    assert mic.stats["fed"] == 40
    boosted = _written_rms(sink)
    assert 0.08 <= boosted <= 0.2, f"AGC did not normalize: rms={boosted}"


def test_mic_agc_never_attenuates_or_clips_loud_speech():
    mic, sink = _make_mic(playing=None, agc=True, agc_target=0.12,
                          agc_max_gain=32.0)
    pcm, amp = _amp_frame(0.2)  # healthy close-up level: leave untouched
    mic._handle_block(rms=0.2, pcm_bytes=pcm)
    assert sink.written[0][1] == pcm  # gain floor 1.0 -> byte-identical
    assert mic._agc_gain == 1.0


def test_mic_agc_silence_gain_capped():
    mic, sink = _make_mic(playing=None, agc=True, agc_target=0.12,
                          agc_max_gain=32.0)
    pcm, _ = _amp_frame(0.00001)  # digital near-silence
    for _ in range(400):
        mic._handle_block(rms=0.00001, pcm_bytes=pcm)
    assert mic._agc_gain <= 32.0 + 1e-6
    boosted = _written_rms(sink)
    assert boosted < 0.01, f"silence hiss amplified into VAD territory: {boosted}"


def test_mic_agc_off_is_passthrough():
    mic, sink = _make_mic(playing=None, agc=False)
    pcm, _ = _amp_frame(0.005)
    for _ in range(10):
        mic._handle_block(rms=0.005, pcm_bytes=pcm)
    assert mic._agc_gain == 1.0
    assert sink.written[-1][1] == pcm


# --- MicInput APM (WebRTC AEC + NS) ---------------------------------------------

class _FakeApm:
    """Duck-typed MicApm: configurable passthrough/replacement."""

    def __init__(self, out_pcm: bytes | None = None):
        self.out_pcm = out_pcm
        self.near_calls = 0

    def process_near(self, pcm_bytes: bytes) -> bytes:
        self.near_calls += 1
        return self.out_pcm if self.out_pcm is not None else pcm_bytes


def test_mic_apm_replaces_block_and_bypasses_gate():
    # With an APM active the app runs the mic ungated (echo_gate=False,
    # as voice_app assembles it): the AEC residual -- here a zero block
    # -- is what feeds the sink, even while Alice plays.
    zeros = _stereo_frame(0, 0, n_samples=960)
    mic, sink = _make_mic(playing=True, echo_gate=False, apm=_FakeApm(zeros))
    pcm = _stereo_frame(1, 1)
    for _ in range(5):
        mic._handle_block(rms=0.05, pcm_bytes=pcm)
    assert mic.stats["fed"] == 5          # ungated: AEC owns the echo
    assert mic.stats["gated"] == 0
    # The PROCESSED block is what reaches the sink (AGC-scaled zeros are
    # still zeros).
    assert sink.written[-1][1] == zeros


def test_mic_apm_gate_decision_uses_processed_level():
    # A loud RAW block whose processed residual is quiet must NOT open
    # the barge feed: the burst check reads the post-APM level.
    quiet = _stereo_frame(0, 0, n_samples=960)
    mic, sink = _make_mic(playing=True, barge_blocks=2, apm=_FakeApm(quiet))
    for _ in range(6):
        mic._handle_block(rms=0.5, pcm_bytes=_stereo_frame(3200, 3200))
    assert sink.written == []
    assert mic._burst == 0                # processed RMS ~0 -> burst resets


def test_mic_apm_failure_feeds_raw_block():
    class _BrokenApm:
        def process_near(self, pcm_bytes):
            raise RuntimeError("dsp exploded")

    mic, sink = _make_mic(playing=None, apm=_BrokenApm())
    pcm = _stereo_frame(500, 500)
    mic._handle_block(rms=0.02, pcm_bytes=pcm)
    assert len(sink.written) == 1         # raw block still reaches the sink
    assert sink.written[0][1] == pcm

# --- MicInput Discord-call stand-down --------------------------------------------

def test_mic_suppressed_during_discord_call():
    # While a real Discord voice session is active, the desktop feed
    # must stand down entirely: no second Alice over the call.
    mic, sink = _make_mic(playing=None, suppress_fn=lambda: True)
    pcm = _stereo_frame(1, 1)
    for _ in range(5):
        mic._handle_block(rms=0.05, pcm_bytes=pcm)
    assert sink.written == []
    assert mic.stats["suppressed"] == 5
    assert mic.stats["fed"] == 0


def test_mic_resumes_after_discord_call_ends():
    state = {"active": True}  # call in progress
    mic, sink = _make_mic(playing=None, suppress_fn=lambda: state["active"])
    pcm = _stereo_frame(1, 1)
    for _ in range(3):
        mic._handle_block(rms=0.05, pcm_bytes=pcm)
    assert mic.stats["suppressed"] == 3
    state["active"] = False  # call ended -> feed resumes
    for _ in range(3):
        mic._handle_block(rms=0.05, pcm_bytes=pcm)
    assert mic.stats["fed"] == 3
    assert not mic._suppress_active


try:
    import aec_audio_processing  # noqa: F401
    _HAVE_APM = True
except Exception:  # pragma: no cover
    _HAVE_APM = False


@pytest.mark.skipif(not _HAVE_APM, reason="aec-audio-processing not installed")
def test_micapm_real_roundtrip_shape():
    from desktop.dsp import MicApm

    apm = MicApm(delay_ms=60)
    near = _stereo_frame(800, 800, n_samples=960)  # 20 ms @ 48 kHz stereo
    out = apm.process_near(near)
    assert len(out) == len(near)          # same shape contract
    # Far-end feeding accepts arbitrary block sizes without raising.
    apm.process_far(np.zeros(17, dtype=np.float32))
    apm.process_far(np.zeros(2000, dtype=np.float32))
