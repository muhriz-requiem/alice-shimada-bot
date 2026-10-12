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
from typing import Optional

import numpy as np

from bot.logging_setup import get_logger
import config
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


async def stream_response(
    text: str,
    guild_id: int,
    source: Optional[StreamingAudioSource] = None,
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
                t0 = time.perf_counter()

                # Generate TTS
                wav_bytes, sr, engine = await tts.generate_tts(sentence)
                t_tts = time.perf_counter() - t0

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

                # First sentence + slow stages at INFO so live logs show cost
                if i == 0 or t_total > 2.0:
                    log.info(
                        f"Sentence {i+1}/{len(sentences)} [{engine}]: "
                        f"tts={t_tts*1000:.0f}ms rvc={t_rvc*1000:.0f}ms "
                        f"soften={t_soft*1000:.0f}ms total={t_total*1000:.0f}ms "
                        f"({len(sentence)} chars, {source.buffer_size} frames)"
                    )
                else:
                    log.debug(
                        f"Sentence {i+1}/{len(sentences)}: "
                        f"tts={t_tts*1000:.0f}ms rvc={t_rvc*1000:.0f}ms "
                        f"soften={t_soft*1000:.0f}ms total={t_total*1000:.0f}ms"
                    )

            except Exception as e:
                log.error(f"TTS failed for sentence {i+1}: {e}")
                continue

        source.mark_done()
        log.info(f"Streaming complete: {len(sentences)} sentences")

    except Exception as e:
        log.error(f"Streaming TTS failed: {e}", exc_info=True)
        source.abort()


def get_streaming_stats(source: StreamingAudioSource) -> dict:
    """Get stats about a streaming source."""
    return {
        "buffer_size": source.buffer_size,
        "is_done": source.is_done,
        "bytes_played": source._bytes_played,
    }
