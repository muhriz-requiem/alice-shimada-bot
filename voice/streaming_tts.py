"""
voice/streaming_tts.py
=======================
Sentence-by-sentence TTS streaming. Splits AI response into sentences,
synthesizes each sentence via the active TTS engine (GPT-SoVITS, or
Edge-TTS + RVC as fallback), and feeds them to a StreamingAudioSource
for immediate playback.

Key optimization: first sentence plays while remaining sentences
are still being synthesized. Cuts time-to-first-audio from ~5s to ~2s.
"""

from __future__ import annotations

import asyncio
import re
import time

import numpy as np

import config
from bot.logging_setup import get_logger
from voice import rvc, tts
from voice.streaming_source import StreamingAudioSource

log = get_logger("voice.streaming_tts")

# Sentence boundary patterns (Japanese + English). Ellipses must not
# shatter into bare "." chunks: ascii punctuation only splits when
# followed by whitespace ("alright... I think" splits after the run,
# "...I think" does not), while Japanese 。！？ still splits without a
# space but never in the middle of a "。。。" run.
_SENTENCE_RE = re.compile(
    r'(?:(?<=[。！？])\s*(?![。！？])|(?<=[.!?])\s+)'
)

# A chunk that is only punctuation/whitespace (e.g. a leftover "...")
# carries no speech -- synthesizing it wastes a round-trip and can come
# back as empty audio, so it is dropped instead of played.
_CHUNK_PURE_PUNCT = re.compile(r'[\s.!?。！？…]+')

# Maximum characters per TTS chunk (Edge-TTS handles up to ~500 chars well)
_MAX_CHUNK_CHARS = 200
# Minimum characters to avoid tiny chunks
_MIN_CHUNK_CHARS = 10

# Streamed-sentence merge target: after the first (streamed) sentence,
# the worker batches every sentence that has ALREADY arrived from the
# LLM into one synthesis job. Short one-sentence synthesis jobs leave
# holes between them (the player drains mid-boundary = the fragmented
# sound); a ~110-char job covers enough playback time for the next job
# to finish before the audio runs dry. Never blocks: only what has
# already arrived is merged.
_MERGE_CHARS = 110


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentence-sized chunks for streaming TTS.
    Optimized for Japanese with English fallback.
    """
    if not text.strip():
        return []

    # Split on strong sentence boundaries
    parts = _SENTENCE_RE.split(text.strip())

    # Filter empty parts and punctuation-only fragments
    chunks = [
        p.strip() for p in parts
        if p.strip() and not _CHUNK_PURE_PUNCT.fullmatch(p.strip())
    ]

    # If no splits happened and text is long, force split
    if len(chunks) <= 1 and len(text) > _MAX_CHUNK_CHARS:
        chunks = _force_split(text, _MAX_CHUNK_CHARS)

    return chunks if chunks else [text]


def _force_split(text: str, max_len: int) -> list[str]:
    """Force-split text at natural boundaries."""
    chunks = []
    while len(text) > max_len:
        # Find last punctuation within limit
        split_at = -1
        for sep in ['。', '！', '？', '.', '!', '?', '，', ',', '、', ' ']:
            idx = text.rfind(sep, 0, max_len)
            if idx > split_at:
                split_at = idx + 1

        if split_at <= 0:
            # No good split point, just split at max_len
            split_at = max_len

        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()

    if text:
        chunks.append(text)

    return chunks


class SentenceAccumulator:
    """
    Incremental version of split_sentences() for a streaming reply:
    feed(text deltas) -> completed sentences as soon as their boundary
    exists in the buffer, so TTS synthesis can start before the model
    finishes generating the rest of the reply.

    Boundary rules mirror _SENTENCE_RE exactly:
    - after 。！？, optionally with whitespace, when the next character
      is not another 。？！(a trailing 。 needs one more character to
      decide -- flush() covers the end of the stream);
    - after .!? followed by whitespace.
    A trailing "..." plus whitespace splits after the run, never inside
    it (same as batch mode). feed() + flush() == split_sentences() on
    the concatenated text.
    """

    _JA_END = ("。", "！", "？")
    _WS = " \t\r\n\u3000"

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        """Consume a text delta; return newly completed sentences."""
        if delta:
            self._buffer += delta

        completed: list[str] = []
        while True:
            cut = self._find_boundary()
            if cut is None:
                break
            piece, self._buffer = self._buffer[:cut], self._buffer[cut:]
            piece = piece.strip()
            if piece and not _CHUNK_PURE_PUNCT.fullmatch(piece):
                completed.append(piece)
        return completed

    def flush(self) -> list[str]:
        """End of the stream: emit whatever remains as the last chunk."""
        remaining = self._buffer
        self._buffer = ""
        cleaned = remaining.strip()
        if cleaned and not _CHUNK_PURE_PUNCT.fullmatch(cleaned):
            return [cleaned]
        return []

    def _find_boundary(self) -> int | None:
        """Index where the next sentence starts, or None while the
        boundary is not yet decidable from the buffer."""
        buf = self._buffer
        for i, ch in enumerate(buf):
            if ch in self._JA_END:
                j = i + 1
                while j < len(buf) and buf[j] in self._WS:
                    j += 1
                if j >= len(buf):
                    # Trailing 。！？ -- need one more character before
                    # committing (it could start a 。。。 run).
                    return None
                if buf[j] not in self._JA_END:
                    return j
            elif ch in (".", "!", "?"):
                j = i + 1
                while j < len(buf) and buf[j] in self._WS:
                    j += 1
                if j > i + 1:
                    # At least one whitespace character follows the
                    # punctuation -- split here.
                    return j
        return None


async def stream_response_deltas(
    deltas,
    guild_id: int,
    source: StreamingAudioSource | None = None,
    mood: str = "neutral",
    track: dict | None = None,
) -> tuple[StreamingAudioSource, str, bool]:
    """
    Live-mode streaming reply: sentence boundaries detected as the AI's
    tokens arrive; each completed sentence is synthesized into the
    shared StreamingAudioSource while later text is still generating.
    The FIRST sentence additionally uses the GPT-SoVITS streaming API
    when enabled, so its very first PCM chunk plays in a few hundred ms.

    deltas is an async iterator of raw reply-text chunks (e.g. from
    GroqClient.generate_reply_stream). Each completed sentence runs
    through postprocess.clean_stream_sentence (same cleaning as the
    batch path, per sentence); when a raw-completion artifact
    (<|special|>, stray URL) shows up in the raw deltas, further text
    is discarded -- everything before the artifact is reliably the real
    reply.

    track, when given, is filled with the cleaned sentences as they are
    QUEUED (track["parts"] list, track["text"] joined); it keeps
    updating while the caller plays, arriving at the full spoken text
    by the time the source is done.

    Returns (source, spoken_text, settled):
    - abandoned OR settled: awaits the first completed sentence (or the
      stream ending with none) before returning, so spoken_text at
      return time means something real. settled=False only when no
      sentence arrived within a small grace window -- tokens may still
      be on the way, so the caller shouldn't conclude "silent" unless
      the source ends empty.
    - "" with settled=True: the stream truly produced no speakable
      sentence (reasoning ate the budget / everything filtered).
    """
    if source is None:
        source = StreamingAudioSource()

    from voice import readiness

    if not readiness.is_ready():
        log.info("TTS not ready yet; dropping streamed reply")
        source.mark_done()
        return source, "", True

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    from ai.postprocess import clean_stream_sentence, find_hard_cut

    # Shared completion state between reader and caller.
    state: dict = {"cut": False}
    first_ready = asyncio.Event()
    parts = track.setdefault("parts", []) if track is not None else []
    spoken_parts: list[str] = parts

    async def _reader() -> None:
        """Feed cleaned completed sentences into the queue as they arrive."""
        acc = SentenceAccumulator()
        raw_tail = ""  # rolling window for cross-delta hard-cut detection
        try:
            async for delta in deltas:
                if delta:
                    raw_tail = (raw_tail + delta)[-256:]
                    if find_hard_cut(raw_tail) is not None:
                        state["cut"] = True
                        log.info("Reply stream hard-cut at completion artifact")
                        break
                    for piece in acc.feed(delta):
                        cleaned = clean_stream_sentence(piece, config.CHARACTER_NAME)
                        if not cleaned:
                            continue
                        spoken_parts.append(piece)
                        if track is not None:
                            track["text"] = "".join(spoken_parts).strip()
                        await queue.put(cleaned)
                        first_ready.set()
            if not state["cut"]:
                for piece in acc.flush():
                    cleaned = clean_stream_sentence(piece, config.CHARACTER_NAME)
                    if not cleaned:
                        continue
                    spoken_parts.append(piece)
                    if track is not None:
                        track["text"] = "".join(spoken_parts).strip()
                    await queue.put(cleaned)
                    first_ready.set()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - never wedge the queue
            log.error(f"Reply stream read failed: {e}")
        finally:
            await queue.put(None)
            first_ready.set()

    asyncio.create_task(_reader())

    async def _worker() -> None:
        first = True
        stream_done = False
        try:
            while True:
                piece = await queue.get()
                if piece is None:
                    break

                # Batch what has ALREADY arrived behind this piece (never
                # blocks): one synthesis job spanning several short
                # sentences removes the inter-sentence synthesis gap --
                # the fragmented sound when the player drains at every
                # sentence boundary. The FIRST synthesis keeps the pure
                # sentence + SoVITS streaming path (the TTFA lever).
                use_stream = first
                if not first:
                    while len(piece) < _MERGE_CHARS:
                        try:
                            nxt = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if nxt is None:
                            stream_done = True
                            break
                        piece = f"{piece} {nxt}"

                try:
                    await _synthesize_and_add(
                        piece, guild_id, source, mood,
                        use_sovits_stream=use_stream,
                    )
                except Exception as e:  # noqa: BLE001 - skip a failed sentence
                    log.error(f"TTS failed for streamed sentence: {e}")
                    continue
                finally:
                    first = False
                if stream_done:
                    break
        finally:
            source.mark_done()
            log.info("Streaming (delta) complete")

    asyncio.create_task(_worker())

    # Await the first sentence BEFORE returning, so the returned text
    # reflects reality (early-empty = concretely nothing has spoken).
    settled = True
    try:
        await asyncio.wait_for(first_ready.wait(), timeout=3.0)
    except TimeoutError:
        settled = False
        log.warning(
            "Reply stream: no sentence within 3s (tokens still arriving?)"
        )

    return source, "".join(spoken_parts).strip(), settled


async def stream_response(
    text: str,
    guild_id: int,
    source: StreamingAudioSource | None = None,
    mood: str = "neutral",
) -> StreamingAudioSource:
    """
    Stream an AI response sentence-by-sentence.

    1. Split text into sentences
    2. Start TTS for first sentence immediately
    3. While first plays, start TTS for second sentence
    4. Continue until all sentences are streamed

    Args:
        text: AI response text
        guild_id: Guild ID for RVC
        source: Existing source to add to (creates new if None)
        mood: Character mood for TTS

    Returns:
        The StreamingAudioSource (for tracking/completion)
    """
    if source is None:
        source = StreamingAudioSource()

    sentences = split_sentences(text)
    if not sentences:
        source.mark_done()
        return source

    from voice import readiness

    if not readiness.is_ready():
        log.info("TTS not ready yet; dropping streamed reply")
        source.mark_done()
        return source

    log.info(f"Streaming {len(sentences)} sentences: '{text[:80]}...'")

    # Start streaming in background
    asyncio.create_task(
        _stream_sentences(sentences, guild_id, source, mood)
    )

    return source


async def _stream_sentences(
    sentences: list[str],
    guild_id: int,
    source: StreamingAudioSource,
    mood: str,
) -> None:
    """Background task that synthesizes and streams sentences."""
    try:
        for i, sentence in enumerate(sentences):
            try:
                t_total = await _synthesize_and_add(
                    sentence, guild_id, source, mood
                )

                # First sentence + slow stages at INFO so live logs show cost
                if i == 0 or t_total > 2.0:
                    log.info(
                        f"Sentence {i+1}/{len(sentences)}: total="
                        f"{t_total*1000:.0f}ms "
                        f"({len(sentence)} chars, {source.buffer_size} frames)"
                    )
                else:
                    log.debug(
                        f"Sentence {i+1}/{len(sentences)}: total="
                        f"{t_total*1000:.0f}ms"
                    )

            except Exception as e:
                log.error(f"TTS failed for sentence {i+1}: {e}")
                continue

        source.mark_done()
        log.info(f"Streaming complete: {len(sentences)} sentences")

    except Exception as e:
        log.error(f"Streaming TTS failed: {e}", exc_info=True)
        source.abort()


async def _synthesize_and_add(
    sentence: str,
    guild_id: int,
    source: StreamingAudioSource,
    mood: str,
    use_sovits_stream: bool = False,
) -> float:
    """
    Synthesize one sentence and append it to the streaming source.

    With use_sovits_stream (live mode's first sentence), synthesis goes
    through the GPT-SoVITS streaming API and chunks play as they are
    generated; any failure falls back to the legacy whole-sentence path
    on top of the same source. Returns the elapsed synthesis time (s).
    """
    t0 = time.perf_counter()

    streamed_chunks: list[np.ndarray] = []
    if (
        use_sovits_stream
        and config.VOICE_SOVITS_STREAMING
        and config.VOICE_TTS_ENGINE == "sovits"
    ):
        from voice import sovits

        got_chunks = False
        try:
            async for chunk in sovits.generate_stream(sentence):
                got_chunks = True
                streamed_chunks.append(chunk)
                source.add_pcm(chunk, source_sr=32000)
            if got_chunks:
                t_total = time.perf_counter() - t0
                log.debug(
                    f"sovits streamed sentence ({len(sentence)} chars) "
                    f"total={t_total*1000:.0f}ms"
                )
                # The streaming path returns before the legacy publish below,
                # so publish here too -- otherwise the avatar mouth never
                # moves for streamed (first) sentences. Published at stream
                # completion: the subscriber animates over whatever playback
                # remains, which for a short streamed opener is most of it.
                _publish_lip_sync(
                    sentence,
                    tts.array_to_wav_bytes(
                        np.concatenate(streamed_chunks), sample_rate=32000
                    ),
                    32000,
                    "sovits",
                )
                return t_total
            log.warning("sovits streaming returned no audio; falling back")
        except Exception as e:  # noqa: BLE001 - fallback must take over
            log.warning(f"sovits streaming failed, falling back: {e}")

    # Legacy whole-sentence path (also the streaming fallback).
    wav_bytes, sr, engine = await tts.generate_tts(sentence)

    t_rvc = t_soft = 0.0
    # Apply RVC only for the edge engine (sovits is already Alice)
    if engine != "sovits" and rvc.is_available():
        audio = tts.wav_bytes_to_array(wav_bytes, target_sr=16000)
        t1 = time.perf_counter()
        audio = await asyncio.to_thread(rvc.convert, audio, 16000)
        t_rvc = time.perf_counter() - t1
        t1 = time.perf_counter()
        audio = await asyncio.to_thread(rvc.soften_human, audio, 16000)
        t_soft = time.perf_counter() - t1
        wav_bytes = tts.array_to_wav_bytes(audio, sample_rate=16000)
        sr = 16000

    t_total = time.perf_counter() - t0

    # Add to stream
    if engine == "sovits":
        # SoVITS outputs 32k WAV — resample for the 24k stream
        audio_float = tts.wav_bytes_to_array(wav_bytes, target_sr=24000)
        source.add_pcm(audio_float, source_sr=24000)
    elif sr == 16000:
        # RVC output — need to resample to 24k for streaming
        audio_float = tts.wav_bytes_to_array(wav_bytes, target_sr=24000)
        source.add_pcm(audio_float, source_sr=24000)
    else:
        source.add_wav_bytes(wav_bytes)

    # Publish to lip-sync subscribers (neutral: no desktop import here).
    if getattr(config, "VOICE_LIP_SYNC", True):
        _publish_lip_sync(sentence, wav_bytes, sr, engine)
    else:
        log.debug(
            f"Sentence [{engine}]: total={t_total*1000:.0f}ms "
            f"rvc={t_rvc*1000:.0f}ms soften={t_soft*1000:.0f}ms"
        )
    return t_total


def _publish_lip_sync(
    sentence: str, wav_bytes: bytes, sr: int, engine: str
) -> None:
    """One synthesized sentence -> lip-sync subscribers (never raises).

    Neutral: no desktop imports, so voice/ stays headless/Discord-usable.
    """
    try:
        import io
        import wave

        from voice import lip_sync

        _dur = 0.0
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as _wf:
                _dur = _wf.getnframes() / float(_wf.getframerate() or sr or 1)
        except Exception:  # noqa: BLE001
            _dur = 0.0
        lip_sync.publish({
            "text": sentence,
            "wav_bytes": wav_bytes,
            "sr": sr,
            "engine": engine,
            "duration": _dur,
        })
    except Exception as exc:  # noqa: BLE001
        log.debug("lip_sync publish failed: %s", exc)


def get_streaming_stats(source: StreamingAudioSource) -> dict:
    """Get stats about a streaming source."""
    return {
        "buffer_size": source.buffer_size,
        "is_done": source.is_done,
        "bytes_played": source._bytes_played,
    }
