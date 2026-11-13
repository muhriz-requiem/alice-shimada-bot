"""
voice/pipeline.py
=================
Main voice processing pipeline. Coordinates STT, diarization, AI response,
TTS (GPT-SoVITS engine with Edge-TTS/RVC fallback), and the
conversation-aware response decision engine.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import scipy.signal

import config
from bot.logging_setup import get_logger
from character import prompt_builder
from memory import store as memory_store
from metrics import (
    record_glance,
    record_voice_predictive,
    record_voice_reply,
    record_voice_stage,
    record_voice_stt_filtered,
)
from social.identities import get_identity_async
from voice import rvc, stt, tts
from voice.backchannel import BackchannelBank, BackchannelGenerator

# NOTE (cold-start): the pyannote import used to be module-level here,
# which made `import voice.pipeline` pay pyannote.audio's whole import
# chain (speechbrain/lightning) in EVERY construction -- the single
# biggest "Companion: booting..." cost, paid even when diarization is
# off (desktop default). It is lazy now: see _get_diarize_fn().
from voice.conversation import ConversationTracker
from voice.mixer import VoiceMixer
from voice.response import ResponseDecision, ResponseDecisionEngine
from voice.sink import AliceVoiceSink
from voice.state import TurnState as StateTurnState
from voice.state import voice_state
from voice.streaming_source import StreamingAudioSource
from voice.streaming_tts import (
    stream_response,
    stream_response_deltas,
)
from voice.turn_detector import TurnDetector, TurnState

log = get_logger("voice.pipeline")

# Whisper can hallucinate these on short/noisy clips — drop if the whole
# transcription is just one of these (or starts with one and is very short).
_STT_HALLUCINATIONS = {
    "localities", "locality", "loathing", "loath", "you", "thank you",
    "thanks for watching", "subscribe", "like and subscribe",
    "the", "and", "a", "uh", "um", "mm", "hmm",
}

# Sentences that the old STT prompt biased the model toward. Transcripts that
# are loops of these sentences, or saturated with one of them on short audio
# (Whisper re-emits such prompts verbatim on noisy/overlapping clips), are
# treated as STT hallucinations and dropped before being recorded or sent to
# the AI. Optional pipe-delimited override via config.VOICE_STT_HALLUCINATION_PHRASES.
_STT_HALLUCINATION_PHRASES_DEFAULT = [
    "Hello Alice.",
    "Hi Alice.",
    "Alice, how are you doing?",
    "What are you doing, Alice?",
]


def _is_clean_direct_address(text: str) -> bool:
    """
    Return True if the utterance looks like a genuine direct address to Alice
    (a mention keyword vocatively placed at the start, e.g. "Alice, can you
    hear me?"). A real one-off direct address on a short clip is allowed to
    pass the hallucination filter rather than being dropped.
    """
    text_l = text.lower()
    mention_kws = [kw.lower() for kw in getattr(
        config, "VOICE_MENTION_KEYWORDS", ["alice"]) if kw and kw.strip()]
    return any(
        text_l.startswith(kw + ",") or text_l.startswith(kw + " ")
        for kw in mention_kws
    )


def _is_stt_hallucination(text: str, audio_dur: float) -> bool:
    """
    Reject transcripts that look like STT hallucinations instead of real
    speech. A rejected utterance is dropped BEFORE being recorded in the
    conversation or sent to the AI, so it can't seed a response loop.

    Checks (all must pass to accept):
    - not a verbatim/loop of known hallucination sentences,
    - not a self-address loop (keyword+comma+short phrase repeated 2+ times),
    - not a silence impostor (few content words on long-ish audio).
    """
    text = text.strip()
    if not text:
        return True

    # --- Rule 1: known hallucination sentences (verbatim + loops + saturation) ---
    phrases = getattr(config, "VOICE_STT_HALLUCINATION_PHRASES", None)
    if not phrases:
        phrases = "|".join(_STT_HALLUCINATION_PHRASES_DEFAULT)
    sentences = [p.strip() for p in phrases.split("|") if p.strip()]

    normalized = re.sub(r"[^\w]", " ", text).lower().strip()
    words = normalized.split()
    if not words:
        return True

    hit_total = 0
    for sent in sentences:
        sent_norm = re.sub(r"[^\w]", " ", sent).lower().strip()
        if not sent_norm:
            continue
        cnt = normalized.count(sent_norm)
        hit_total += cnt
        # Same sentence repeated = a loop, definitely a hallucination.
        if cnt >= 2:
            return True
        # Question-type hallucination sentences only make sense as real
        # direct questions (a clear "Alice, how are you doing?" needs ~2.5s);
        # on short audio they are almost certainly prompt echoes. Greetings
        # ("Hello Alice.", "Hi Alice.") are common real speech and are never
        # rejected by duration/saturation alone.
        is_question = sent_norm.startswith(("alice, how are", "what are you"))
        if is_question and cnt == 1 and audio_dur < 2.5:
            if not _is_clean_direct_address(text):
                return True

    if hit_total >= 2:
        return True

    # --- Rule 2: self-address loop ---
    # e.g. "Alice, how are you doing? Alice, ..." or "Alice? Alice? Alice,"
    mention_kws = [kw.lower() for kw in getattr(
        config, "VOICE_MENTION_KEYWORDS", ["alice"]) if kw and kw.strip()]
    addr_re = re.compile(
        rf"\b(?:{'|'.join(re.escape(k) for k in mention_kws)})\b(?:,|\?)"
    )
    addr_hits = 0
    for m in addr_re.finditer(text.lower()):
        end = m.end()
        rest = text.lower()[end:end + 45]
        phrase = rest.split(".")[0].strip()
        # An empty phrase means the keyword was the last thing in the text
        # (a standalone vocative, e.g. "Alice? Alice? Alice,"), count it.
        if (phrase and len(phrase.split()) <= 6) or (
                phrase == "" and rest.strip() == ""):
            addr_hits += 1
    if addr_hits >= 2:
        return True

    # --- Rule 3: silence impostor ---
    # Remove filler words and count remaining content words.
    cleaned = re.sub(
        r"(?:^|\s)(?:uh|um|mm|hmm|ah|ahh)\s*(?:$|\s)", " ", normalized
    )
    content_words = [w for w in cleaned.split() if w and len(w) > 2]
    return bool(len(content_words) < 2 and audio_dur >= 2.5)


_capture_counter = 0


def _capture_clip(
    guild_id: int,
    user_id: int,
    audio: np.ndarray,
    sample_rate: int,
    result: dict,
) -> None:
    """Save one voice clip + JSON sidecar for the STT eval set.

    Runs on the raw input (before resample/hallucination filtering) so the
    eval tool can replay the exact production audio through candidate
    engines and preprocessing. Self-limits at VOICE_STT_CAPTURE_LIMIT.
    """
    global _capture_counter
    try:
        import scipy.io.wavfile as wavfile

        from voice.preprocess import noise_floor_rms

        out_dir = Path(config.STT_EVAL_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(out_dir.glob("clip_*.wav")))
        if existing >= config.VOICE_STT_CAPTURE_LIMIT:
            return

        _capture_counter += 1
        stem = f"clip_{int(time.time())}_{_capture_counter:04d}"
        wav_path = out_dir / f"{stem}.wav"
        side_path = out_dir / f"{stem}.json"

        pcm = np.asarray(audio, dtype=np.float32)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        peak = np.max(np.abs(pcm)) if pcm.size else 0.0
        if peak > 1.0:
            pcm = pcm / peak
        wavfile.write(
            str(wav_path), int(sample_rate), (pcm * 32767.0).astype(np.int16)
        )

        sidecar = {
            "ts": time.time(),
            "guild_id": guild_id,
            "user_id": user_id,
            "sample_rate": int(sample_rate),
            "dur_s": round(len(pcm) / sample_rate, 3),
            "engine": getattr(config, "VOICE_STT_ENGINE", "moonshine"),
            "text": result.get("text", ""),
            "language": result.get("language", ""),
            "noise_floor_rms": round(noise_floor_rms(pcm, sample_rate), 6),
        }
        side_path.write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.debug(f"STT eval capture: {wav_path.name} ({sidecar['dur_s']}s)")
    except Exception as e:
        log.warning(f"STT eval capture failed: {e}")


_DIARIZE_FN = None


def _get_diarize_fn():
    """Lazy pyannote accessor (module-level import was too expensive).

    pyannote.audio's import chain makes up most of the cold cost of
    `import voice.pipeline`; diarization is an optional stage (off on the
    desktop by default), so pay it only when first used.
    """
    global _DIARIZE_FN
    if _DIARIZE_FN is None:
        from voice.diarize import diarize as _fn
        _DIARIZE_FN = _fn
    return _DIARIZE_FN


class VoicePipeline:
    """
    End-to-end voice processing pipeline with conversation awareness.

    Flow:
    1. Receive audio from Discord (via AliceVoiceSink)
    2. VAD + buffering in sink (sample-based timing)
    3. Turn detection (semantic completion)
    4. Transcribe with Whisper
    5. Response decision (I1/I2/S1/S2 framework)
    6. Generate AI response (with conversation context)
    7. Convert TTS to Alice's voice with RVC
    8. Play audio (with backchannel overlap support)
    """

    def __init__(self, bot):
        self.bot = bot
        self.sinks: dict[int, AliceVoiceSink] = {}
        self._processing: dict[int, bool] = {}
        self._last_reply: dict[int, float] = {}
        # One reply in flight per guild: utterances arriving while
        # Alice is answering are dropped instead of racing into
        # vc.play() (discord raises ClientException "Already playing
        # audio" on the second call).
        self._reply_locks: dict[int, asyncio.Lock] = {}
        self._monitor_tasks: dict[int, asyncio.Task] = {}

        # New conversation-aware components
        self.conversation = ConversationTracker(max_utterances=config.VOICE_CONTEXT_WINDOW)
        self.turn_detector = TurnDetector(sample_rate=config.VOICE_DISCORD_SAMPLE_RATE)
        self.response_engine = ResponseDecisionEngine(
            conversation=self.conversation,
            turn_detector=self.turn_detector,
        )
        self.backchannel = BackchannelGenerator()
        self.mixer = VoiceMixer()
        # Non-blocking diarization stats per guild (speaker count, multi-speaker clips)
        self._diarize_stats: dict[int, dict] = {}

        # --- Screen glance ("Alice looks at my screen") ---------------------
        # The desktop registers a provider here: async () -> GlanceFrame |
        # None (None = permission off / denylisted / capture failed). The
        # voice pipeline only calls it on explicit look intents and deixis
        # ("what do you think of this?"); voice/ stays headless.
        self.screen_look_provider = None
        self._look_cooldown_until: dict[int, float] = {}

        # --- Live mode (VOICE_LIVE_MODE) -----------------------------------
        # Adaptive turn-end, predictive STT, banked address-acks ("Hm?"),
        # streaming Groq replies and (optional) SoVITS streaming synthesis:
        # the rebuilt interaction model that overlaps every stage with
        # waiting so Alice answers like ChatGPT's voice mode.
        self._live = bool(config.VOICE_LIVE_MODE)
        self.ack_bank = BackchannelBank(
            size_per_lang=config.VOICE_BACKCHANNEL_BANK_SIZE
        )
        # (guild_id, user_id) -> {"task": Task | None, "revision": int,
        #   "started": monotonic, "result": STT result dict | None}
        self._pred_stt: dict[tuple[int, int], dict] = {}
        # Serializes whisper runs per guild: the whisper model is not
        # thread-safe, and a predictive decode can still be running when
        # the gate fires and a fresh decode would otherwise start.
        self._stt_locks: dict[int, asyncio.Lock] = {}
        # Guilds with an address-ack ("Hm?") currently playing.
        self._ack_in_flight: set[int] = set()
        self._ack_counts: dict[int, int] = {}
        self._last_ack: dict[int, float] = {}
        # guild -> {stage: deque[maxlen=20] of seconds}; rolling medians
        # feed the address-ack delay prediction and the stats panels.
        self._stage_times: dict[int, dict[str, deque]] = {}
        # Periodic sink-vitals logging (30s) in the monitor loop: packet
        # flow + VAD speech ratio make silent-mic / hot-mic failures
        # visible instead of silent.
        self._last_vitals: float = 0.0

    def _record_stage(
        self, guild_id: int, stage: str, seconds: float | None,
        ttfa: float | None = None,
    ) -> None:
        """Record a pipeline stage duration (metrics + rolling per-guild).

        ttfa, when given, additionally feeds the TTFA histogram and the
        latest-TTFA gauge (recorded via the "ttfa" stage call).
        """
        if seconds is None:
            return
        seconds = max(0.0, seconds)
        record_voice_stage(stage, seconds, ttfa=ttfa)
        per = self._stage_times.setdefault(guild_id, {})
        dq = per.setdefault(stage, deque(maxlen=20))
        dq.append(seconds)

    def _stage_p50(
        self, guild_id: int, stage: str, default: float | None = None
    ) -> float | None:
        """Median of the last observations for one stage (or default)."""
        dq = self._stage_times.get(guild_id, {}).get(stage)
        if not dq:
            return default
        s = sorted(dq)
        return s[len(s) // 2]

    def get_sink(self, guild_id: int) -> AliceVoiceSink:
        """Get or create the audio sink for a guild."""
        if guild_id not in self.sinks:
            self.sinks[guild_id] = AliceVoiceSink(guild_id=guild_id)
        return self.sinks[guild_id]

    async def _process_speech(
        self,
        guild_id: int,
        user_id: int,
        silence: float | None = None,
        audio: np.ndarray | None = None,
    ) -> None:
        """Process completed speech segment with conversation awareness.

        silence is the elapsed silence (seconds) when the monitor's gate
        fired -- recorded as the pre-STT wait the live pipeline overlaps
        work with.
        audio is the pre-pulled audio buffer (already cleared from sink).
        """
        t_speech_end = time.monotonic()

        # Live mode: prefer a predictive decode started during the
        # silence wait (bounded await for a recent in-flight one).
        preset_result, pred_outcome = await self._claim_predictive(guild_id, user_id)
        if self._live and config.VOICE_STT_PREDICTIVE_START > 0:
            record_voice_predictive(pred_outcome)

        if audio is None or len(audio) == 0:
            return

        # Determine if this is the owner
        owner_ids = config.OWNER_IDS
        if isinstance(owner_ids, str):
            owner_ids = {o.strip() for o in owner_ids.split(",") if o.strip()}
        elif not isinstance(owner_ids, (set, frozenset, list, tuple)):
            owner_ids = set()
        is_owner = str(user_id) in {str(o) for o in owner_ids}

        # Process the complete utterance
        t_stt0 = time.monotonic()
        result = await self.process_audio(
            guild_id, user_id, audio, 48000, preset_result=preset_result
        )
        if result:
            self._record_stage(guild_id, "stt", time.monotonic() - t_stt0)

        if result:
            text, language = result

            # Non-blocking diarization when enabled (multi-speaker analysis)
            diarize_result = None
            if config.VOICE_DIARIZE_MODE != "none":
                # Stats-only stage: any failure must degrade to None --
                # letting it raise would kill this utterance's whole
                # reply path (conversation record + response decision).
                try:
                    diarize_result = await asyncio.to_thread(
                        _get_diarize_fn(), audio, 48000
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning(f"Diarization failed (continuing without stats): {e}")
                speaker_count = len({s["speaker"] for s in diarize_result}) if diarize_result else 0
                stats = self._diarize_stats.setdefault(guild_id, {"speaker_count": 0, "multi_speaker_count": 0})
                stats["speaker_count"] = speaker_count
                if speaker_count > 1:
                    stats["multi_speaker_count"] += 1

            # Record in conversation history
            self.conversation.add_utterance(
                guild_id, user_id, text,
                language=language, is_bot=False,
            )
            voice_state.update_speaker(guild_id, user_id)

            # Update turn detector state
            self.turn_detector.update(guild_id, user_id, audio)

            # Shared-call signal for the response governor: another human
            # in the channel means the utterer may be talking to them,
            # not to Alice.
            others_present = self._others_in_channel(guild_id, user_id)

            # Decide whether to respond
            t_decision0 = time.monotonic()
            decision, priority = self.response_engine.should_respond(
                guild_id, user_id, text, is_owner=is_owner,
                diarize_result=diarize_result, others_present=others_present,
            )
            self._record_stage(guild_id, "decision", time.monotonic() - t_decision0)

            log.info(
                f"Response decision: {decision.value} (priority={priority.value}, "
                f"shared={others_present}) "
                f"for '{text[:50]}...' from user {user_id}"
            )

            if decision == ResponseDecision.RESPOND:
                await self._generate_and_respond(
                    guild_id, user_id, text, is_owner, language,
                    t_speech_end=t_speech_end, silence=silence,
                )
            elif decision == ResponseDecision.RESPOND_SOFT:
                record_voice_reply("soft")
                await self._play_soft_phrase(guild_id, user_id, text, language)
            elif decision == ResponseDecision.BACKCHANNEL:
                record_voice_reply("backchannel")
                await self._play_backchannel(guild_id, text, language)
            else:
                record_voice_reply("no_response")
        else:
            record_voice_reply("dropped")

    def _others_in_channel(self, guild_id: int, user_id: int) -> bool:
        """Whether another non-bot human is currently in the voice channel.

        Feeds the shared-call governor in voice/response.py: with someone
        else present, most speech is side conversation between the humans,
        so Alice only answers her name. Unresolvable channel state counts
        as a private call (never over-censor the owner).
        """
        try:
            vc = voice_state.get_voice_client(guild_id)
            channel = getattr(vc, "channel", None)
            if channel is None:
                return False
            return any(not m.bot and m.id != user_id for m in channel.members)
        except Exception:  # noqa: BLE001
            return False

    # --- Live mode: predictive STT + address-ack --------------------------

    def _maybe_spawn_predictive(self, guild_id: int, user_id: int) -> None:
        """Live mode: start an early STT decode for buffered speech.

        Fires partway through the end-of-speech silence window
        (VOICE_STT_PREDICTIVE_START): the GPU turns silence-wait time
        into transcription time, so at the turn-end gate the transcript
        is often already done. One decode in flight per guild (the GPU
        serializes them anyway).
        """
        if self._live is False or config.VOICE_STT_PREDICTIVE_START <= 0:
            return
        if config.VOICE_STT_HOSTED:
            # Hosted STT is the decode path: a predictive local decode
            # would burn a local engine load/decode for a result the
            # hosted call at the gate does better (and the hosted round
            # trip rarely beats even the short silence window). Skip.
            return
        sink = self.sinks.get(guild_id)
        if sink is None or self._processing.get(guild_id, False):
            return

        obs = sink.observation(user_id)
        if not obs or obs["silence_duration"] < config.VOICE_STT_PREDICTIVE_START:
            return

        key = (guild_id, user_id)
        entry = self._pred_stt.get(key)
        if entry is not None:
            task = entry.get("task")
            if task is not None and not task.done():
                return  # already decoding this user's utterance
            # A finished decode from an earlier utterance layer: replaced
            # by a fresh snapshot below.
            self._pred_stt.pop(key, None)

        if any(
            g == guild_id and e.get("task") is not None and not e["task"].done()
            for (g, _u), e in self._pred_stt.items()
        ):
            return  # another user's decode owns the GPU right now

        snapshot = sink.snapshot_audio(user_id)
        if snapshot is None:
            return
        audio, revision = snapshot
        task = asyncio.create_task(
            self._predictive_decode(guild_id, user_id, audio, 48000, revision)
        )
        self._pred_stt[key] = {
            "task": task,
            "revision": revision,
            "started": time.monotonic(),
            "result": None,
        }
        log.debug(
            f"Predictive STT started (user={user_id}, "
            f"buffer={obs['buffer_duration']:.2f}s)"
        )

    def _stt_lock(self, guild_id: int) -> asyncio.Lock:
        """Serialize whisper runs per guild (model is not thread-safe)."""
        return self._stt_locks.setdefault(guild_id, asyncio.Lock())

    async def _predictive_decode(
        self,
        guild_id: int,
        user_id: int,
        audio: np.ndarray,
        sample_rate: int,
        revision: int,
    ) -> None:
        """Background whisper run on a frozen buffer snapshot."""
        key = (guild_id, user_id)
        t0 = time.monotonic()
        try:
            if sample_rate != 16000:
                audio_16k = scipy.signal.resample(
                    audio, int(len(audio) * 16000 / sample_rate)
                ).astype(np.float32)
            else:
                audio_16k = audio.astype(np.float32)
            async with self._stt_lock(guild_id):
                result = await asyncio.wait_for(
                    asyncio.to_thread(stt.transcribe, audio_16k, 16000),
                    timeout=config.VOICE_STT_DECODE_TIMEOUT,
                )
            self._record_stage(guild_id, "pred_decode", time.monotonic() - t0)
            entry = self._pred_stt.get(key)
            if entry is not None and entry.get("revision") == revision:
                entry["result"] = result
        except TimeoutError:
            # Wedged decode (e.g. CUDA hung under VRAM pressure): cancel
            # the awaiter so the STT lock is released and the pipeline
            # keeps responding. The orphaned thread's result is discarded.
            log.warning(
                f"Predictive STT exceeded {config.VOICE_STT_DECODE_TIMEOUT}s; "
                "releasing the STT lock (decode thread abandoned)"
            )
        except Exception as e:  # noqa: BLE001 - predictive is best-effort
            log.warning(f"Predictive STT failed: {e}")

    async def _claim_predictive(
        self, guild_id: int, user_id: int
    ) -> tuple[dict | None, str]:
        """Collect the predictive decode at the turn-end gate.

        Awaiting the in-flight decode is strictly cheaper than decoding
        from scratch: it already burned its first tokens against this
        utterance's silence window. The wait is bounded by the learned
        remaining decode time plus a small grace (VOICE_STT_PREDICTIVE_WAIT,
        hard-capped at 2.5s) so a wedged engine can't stall the reply.

        Returns (result | None, outcome) with outcome one of: none (no
        decode started), hit (decode finished with the buffer unchanged),
        hit_wait (in-flight finished while waiting and matched), stale
        (finished but newer speech arrived since), miss (did not finish
        inside the wait), failed (decode errored/cancelled).
        """
        key = (guild_id, user_id)
        entry = self._pred_stt.pop(key, None)
        if entry is None:
            return None, "none"

        sink = self.sinks.get(guild_id)
        task = entry.get("task")

        if entry.get("result") is None and task is not None and not task.done():
            unchanged = (
                sink is None
                or entry.get("revision") == sink.get_revision(user_id)
            )
            if unchanged:
                elapsed = time.monotonic() - entry.get("started", time.monotonic())
                expected = self._stage_p50(guild_id, "pred_decode", 0.0) or 0.0
                remaining = max(0.0, expected - elapsed)
                timeout = min(remaining + config.VOICE_STT_PREDICTIVE_WAIT, 2.5)
                if timeout > 0.0:
                    await asyncio.wait({task}, timeout=timeout)

        result = entry.get("result")
        if result is None:
            if task is not None and task.cancelled():
                return None, "failed"
            return None, "miss"

        if sink is not None and entry.get("revision") != sink.get_revision(user_id):
            return None, "stale"
        return result, "hit"

    def _maybe_start_ack(self, guild_id: int, language: str) -> asyncio.Task | None:
        """Live mode: speak a banked "Hm?" while the reply generates.

        Fires only when past replay latencies predict the real reply's
        first audio later than VOICE_ACK_MIN_DELAY (no point
        acknowledging if Alice would start talking anyway), and never
        while audio is already playing. Returns the playback task (the
        caller awaits its completion, bounded, before vc.play).
        """
        if not (self._live and config.VOICE_ACK_ENABLED):
            return None
        if self.mixer.is_playing(guild_id) or guild_id in self._ack_in_flight:
            return None
        if not self.ack_bank.ready:
            return None

        # Cooldown: don't ack more than once per VOICE_ACK_COOLDOWN seconds
        now = time.monotonic()
        last_ack = self._last_ack.get(guild_id, 0.0)
        if now - last_ack < config.VOICE_ACK_COOLDOWN:
            return None

        predicted = (
            (self._stage_p50(guild_id, "stt", 1.2) or 0.0)
            + (self._stage_p50(guild_id, "ai_first", 0.6) or 0.0)
            + (self._stage_p50(guild_id, "tts_first", 0.8) or 0.0)
        )
        if predicted <= config.VOICE_ACK_MIN_DELAY:
            return None

        lang = "ja" if (language or "").startswith("ja") else "en"
        pick = self.ack_bank.pick_ack(lang)
        if pick is None:
            return None
        _audio, _sr, wav_bytes = pick

        self._ack_in_flight.add(guild_id)
        self._ack_counts[guild_id] = self._ack_counts.get(guild_id, 0) + 1
        self._last_ack[guild_id] = now
        return asyncio.create_task(self._play_banked_ack(guild_id, wav_bytes))

    async def _play_banked_ack(self, guild_id: int, wav_bytes: bytes) -> None:
        """Play one pre-synthesized address-ack (instant, no TTS call)."""
        try:
            await self.mixer.play_backchannel(guild_id, wav_bytes)
        except Exception as e:  # noqa: BLE001 - an ack must never kill the reply
            log.error(f"Address-ack playback failed: {e}")
        finally:
            self._ack_in_flight.discard(guild_id)

    async def _screen_look_reply(
        self,
        *,
        guild_id: int,
        user_id_str: str,
        channel_id: str,
        text: str,
        look_kind: str,
        monitor: str | None = None,
        messages: list[dict],
        identity: dict,
        relationship: str | None,
        history: list[dict],
        voice_extra: str,
    ) -> str | None:
        """Utterance-driven glance: capture -> vision -> one in-character
        line (or a fixed refusal when gated/denylisted). Returns None to
        fall through to the normal text reply (e.g. deixis with the
        feature off, or cooldown)."""
        from ai import vision as _vision
        from voice import screen_look as sl

        now = time.monotonic()
        if now < self._look_cooldown_until.get(guild_id, 0.0):
            return None
        self._look_cooldown_until[guild_id] = now + max(1.0, config.SCREEN_LOOK_COOLDOWN_S)

        if not config.SCREEN_LOOK_ENABLED:
            # Feature off: explicit asks get the honest in-character
            # refusal; deixis falls through to the normal reply.
            if look_kind == sl.LOOK_INTENT_EXPLICIT:
                record_glance("refused")
                return sl.REFUSE_PERMISSION
            return None

        frame = None
        try:
            frame = await self.screen_look_provider(monitor)
        except Exception as e:  # noqa: BLE001 - provider is external
            log.warning(f"screen_look provider failed: {e}")
        if frame is None:
            # Tray permission off, denylisted window, or capture failed:
            # nothing left the machine.
            record_glance("refused")
            if look_kind == sl.LOOK_INTENT_EXPLICIT:
                return sl.REFUSE_PERMISSION
            return None

        t_glance0 = time.monotonic()
        reply = await _vision.glance_reply(
            self.bot.ai_client,
            messages,
            frame.jpeg,
            frame.title,
            user_id=user_id_str,
            channel_id=channel_id,
            deixis=(look_kind == sl.LOOK_INTENT_DEIXIS),
            question=text,
        )
        if reply is None:
            record_glance("failed")
            if look_kind == sl.LOOK_INTENT_EXPLICIT:
                return "Hmm... I can't look right now. Sorry, Father."
            return None
        record_glance("ok", time.monotonic() - t_glance0)

        if config.SCREEN_LOOK_MEMORY:
            try:
                note = (
                    f"(glance) Father's screen [{(frame.title or 'unknown')[:60]}]: "
                    f"{reply[:80]}"
                )
                self.bot.memory_graph.add(
                    user_id_str, note, self.bot.character.identity_keywords
                )
            except Exception as e:  # noqa: BLE001 - memory is best-effort
                log.debug(f"glance memory add failed: {e}")

        try:
            cb = getattr(self, "screen_look_on_spoken", None)
            if cb is not None:
                cb(reply)
        except Exception:  # noqa: BLE001 - expression touch is cosmetic
            pass
        return reply

    async def speak_glance(self, guild_id: int, reply_text: str) -> None:
        """Speak an unprompted ambient-glance comment through the normal
        streaming TTS path (no utterance behind it). Used by the desktop
        ambient governor; no-ops when a reply is already in flight."""
        from voice import readiness

        if not readiness.is_ready():
            return
        lock = self._reply_locks.setdefault(guild_id, asyncio.Lock())
        if lock.locked():
            return
        async with lock:
            vc = voice_state.get_voice_client(guild_id)
            source = StreamingAudioSource()
            await stream_response(reply_text, guild_id, source)
            if not vc:
                return
            t_tts0 = time.monotonic()
            while (
                source.first_pcm_at == 0.0
                and not source.is_done
                and time.monotonic() < t_tts0 + 15.0
            ):
                await asyncio.sleep(0.02)
            voice_state.set_bot_playing(guild_id, True)
            self.turn_detector.mark_bot_playing(guild_id, True)
            vc.play(source, after=lambda e: None)
            try:
                deadline = time.time() + 45.0
                while not source.is_done and time.time() < deadline:
                    await asyncio.sleep(0.1)
                if not source.is_done:
                    source.abort()
            finally:
                voice_state.set_bot_playing(guild_id, False)
                self.turn_detector.mark_bot_playing(guild_id, False)

    async def _generate_and_respond(
        self, guild_id: int, user_id: int, text: str, is_owner: bool,
        user_language: str = "en",
        t_speech_end: float | None = None,
        silence: float | None = None,
    ) -> None:
        """Generate AI response and stream it sentence-by-sentence.

        t_speech_end (monotonic) is the moment the turn-end gate fired;
        silence is the elapsed silence at that moment. Both feed the
        stage-timing line and the TTFA metric.
        """
        # Rate limiting. The cooldown stamp moves to the START of the
        # reply (it used to be set only after playback finished), so
        # utterances arriving during generation are gated too.
        now = time.time()
        last = self._last_reply.get(guild_id, 0)
        if now - last < config.VOICE_REPLY_COOLDOWN:
            log.debug(f"Rate limited for guild {guild_id}")
            return

        lock = self._reply_locks.setdefault(guild_id, asyncio.Lock())
        if lock.locked():
            # Already answering a previous utterance -- drop this one
            # rather than queue a stale reply behind it.
            log.debug(f"Reply in flight for guild {guild_id}; dropping utterance")
            return

        from voice import readiness

        if not readiness.is_ready():
            log.info("TTS not ready yet; dropping reply")
            return

        # Serialize reply generation + playback per guild.
        await lock.acquire()
        self._last_reply[guild_id] = time.time()
        try:
            # Mark bot as responding
            voice_state.set_bot_playing(guild_id, True)
            self.turn_detector.mark_bot_playing(guild_id, True)
            voice_state.set_turn_state(guild_id, StateTurnState.BOT_RESPONDING)

            # Determine response language
            response_lang = config.VOICE_RESPONSE_LANGUAGE
            if response_lang == "auto":
                response_lang = user_language

            lang_instruction = (
                f"[Language instruction] Respond in {response_lang}. "
                f"Do NOT use any other language.\n"
            )

            # Voice call context: keep replies short and conversational.
            voice_extra = (
                "You are in a live Discord voice call. Keep replies short "
                "(1-3 sentences), natural, and in-character. Do not use "
                "markdown, lists, or action/Narrator stage directions — "
                "only spoken words.\n"
                f"{lang_instruction}"
            )

            # Conversation history from the voice tracker (exclude the
            # current utterance -- it was already recorded and is passed
            # as user_message). Kept short: completion tokens count
            # against Groq's TPM budget and spoken turns are fast.
            recent = self.conversation.get_recent(guild_id, count=5)
            history: list[dict] = []
            for u in recent[:-1]:
                history.append({
                    "role": "assistant" if u.is_bot else "user",
                    "content": u.text,
                })

            # Identity/relationship for personalization (same as cogs.chat).
            user_id_str = str(user_id)
            identity: dict = {}
            relationship: str | None = None
            try:
                identity = await get_identity_async(user_id_str)
                relationship = await asyncio.to_thread(
                    memory_store.get_relationship, user_id_str
                )
            except Exception as e:
                log.debug(f"Identity/relationship lookup failed (continuing): {e}")

            character = self.bot.character
            ai_client = self.bot.ai_client

            messages = prompt_builder.build_voice_prompt(
                character,
                identity=identity if identity.get("is_authoritative") else None,
                relationship_key=relationship,
                history=history,
                user_message=text,
                extra_context=voice_extra,
            )

            session = voice_state.get_session(guild_id)
            channel_id = str(session.channel_id if session else guild_id)

            tt_speech_end = t_speech_end if t_speech_end is not None else None
            wait_silence = silence if silence is not None else 0.0

            def _fmt(v: float | None) -> str:
                return f"{v:.2f}" if v is not None else "-"

            # --- Screen glance (explicit look / deixis) --------------------
            # When the utterance asks her to look (or refers to "this"),
            # the reply comes from the vision path instead of the text
            # model. Refusal/gate cases return a fixed in-character line;
            # failures fall through to the normal text reply.
            look_reply: str | None = None
            from voice import screen_look as _sl

            look_kind = _sl.match_look_intent(text)
            if look_kind != _sl.LOOK_INTENT_NONE:
                monitor_hint = _sl.match_monitor_hint(text)
                if self.screen_look_provider is None:
                    # No capture source (Discord bot on a remote host, or
                    # the desktop session is off): explicit asks get an
                    # honest line instead of a confused normal reply;
                    # deixis falls through to the normal reply.
                    if look_kind == _sl.LOOK_INTENT_EXPLICIT:
                        look_reply = _sl.REFUSE_NO_PROVIDER
                else:
                    look_reply = await self._screen_look_reply(
                        guild_id=guild_id,
                        user_id_str=user_id_str,
                        channel_id=channel_id,
                        text=text,
                        look_kind=look_kind,
                        monitor=monitor_hint,
                        messages=messages,
                        identity=identity,
                        relationship=relationship,
                        history=history,
                        voice_extra=voice_extra,
                    )

            response: str | None = None
            stream = None
            ack_task: asyncio.Task | None = None
            track: dict = {}

            if look_reply is not None:
                # Vision reply already final; ride the batch TTS path.
                response = look_reply
                ai_first = None
                stream = None
            else:
                t_ai0 = time.monotonic()
                if (
                    self._live and config.VOICE_STREAM_GROQ
                    and hasattr(ai_client, "generate_reply_stream")
                ):
                    stream = await ai_client.generate_reply_stream(
                        messages, user_id=user_id_str, channel_id=channel_id,
                        max_tokens=config.VOICE_MAX_TOKENS,
                    )

                if stream is not None:
                    # Live streaming reply: the first sentence starts
                    # synthesizing while the model is still generating. The
                    # banked address-ack fires NOW so it plays concurrently.
                    ack_task = self._maybe_start_ack(guild_id, response_lang)
                    source, response, settled = await stream_response_deltas(
                        stream.iter_chunks(), guild_id, source=None,
                        track=track,
                    )
                    ai_first = (
                        stream.first_token_at - t_ai0
                        if stream.first_token_at is not None else None
                    )
                    self._record_stage(guild_id, "ai_first", ai_first)
                    if settled and not response:
                        # Reasoning consumed the budget / every sentence was
                        # filtered: same silent-on-fallback rule as batch.
                        log.info("AI stream produced no speech; staying silent in voice")
                        record_voice_reply("silent")
                        return
                    if settled:
                        log.info(f"AI response (streamed): '{response[:100]}...'")
                else:
                    response = await ai_client.generate_reply(
                        messages, user_id=user_id_str, channel_id=channel_id,
                        max_tokens=config.VOICE_MAX_TOKENS,
                    )
                    ai_total = time.monotonic() - t_ai0
                    ai_first = ai_total
                    self._record_stage(guild_id, "ai_first", ai_total)
                    self._record_stage(guild_id, "ai", ai_total)

                if not response or response == config.FALLBACK_REPLY:
                    # Backend failed (rate limit, parse error, timeout, or
                    # empty content -- reasoning ate the token budget) and
                    # returned the fallback. Text chat still shows it; on a
                    # live call, silence beats repeating "I don't know what
                    # to say." every time Groq hiccups.
                    log.info("AI reply unavailable (fallback); staying silent in voice")
                    record_voice_reply("silent")
                    return

            # Batch path: stream the finished reply sentence-by-sentence.
            # (The streamed path's source is already being fed.)
            t_tts0 = time.monotonic()
            if stream is None:
                source = StreamingAudioSource()
                await stream_response(response, guild_id, source)

            # Play the streaming source
            vc = voice_state.get_voice_client(guild_id)
            t_play: float | None = None
            audible: float | None = None
            if vc and source:
                # Give the address-ack room to finish (bounded): both use
                # the same voice client; overlapping play() calls raise.
                if ack_task is not None and not ack_task.done():
                    await asyncio.wait({ack_task}, timeout=1.5)
                    ack_deadline = time.monotonic() + 1.0
                    while self.mixer.is_playing(guild_id) and time.monotonic() < ack_deadline:
                        await asyncio.sleep(0.05)

                # Wait until the first PCM exists (so play() starts with
                # audio, not an empty buffer) -- bounded like synthesis.
                # (Was `time.time() < t_tts0 + 30` -- t_tts0 is
                # time.monotonic(), so the comparison was always False and
                # this wait never ran; the source then reported silence.
                while (
                    source.first_pcm_at == 0.0
                    and not source.is_done
                    and time.monotonic() < t_tts0 + 30.0
                ):
                    await asyncio.sleep(0.02)

                vc.play(source, after=lambda e: None)
                t_play = time.monotonic()
                # Wait for TTS synthesis (capped so a hung synth can't hang forever)
                deadline = time.time() + 60.0
                while not source.is_done and time.time() < deadline:
                    await asyncio.sleep(0.1)
                if not source.is_done:
                    log.warning("Streaming TTS synthesis timeout; aborting source")
                    source.abort()
                # Wait for playback to drain trailing silence
                while vc.is_playing() and time.time() < deadline:
                    await asyncio.sleep(0.1)
                if vc.is_playing():
                    vc.stop()
                audible = max(t_play, source.first_pcm_at) if source.first_pcm_at else t_play
            else:
                # No voice client (harness/off-playback runs): first PCM
                # is the first audio that would have been heard. Same
                # monotonic() fix as the vc branch above.
                while (
                    source.first_pcm_at == 0.0
                    and not source.is_done
                    and time.monotonic() < t_tts0 + 30.0
                ):
                    await asyncio.sleep(0.02)
                audible = source.first_pcm_at or None

            if audible is None:
                # Nothing was ever synthesized (all sentences failed to
                # produce audio): count it explicitly instead of ok.
                record_voice_reply("empty")
            else:
                # Record what was actually spoken: for the streamed path
                # the queued text now includes sentences that arrived
                # while we were already playing.
                if stream is not None:
                    spoken_final = str(track.get("text") or "").strip() or response
                else:
                    spoken_final = response
                self.conversation.add_utterance(
                    guild_id, self.bot.user.id, spoken_final,
                    language="unknown", is_bot=True,
                )

                tts_first = audible - t_tts0
                self._record_stage(guild_id, "tts_first", tts_first)
                if tt_speech_end is not None:
                    ttfa = max(0.0, audible - tt_speech_end)
                    self._record_stage(guild_id, "ttfa", ttfa, ttfa=ttfa)
                    log.info(
                        "Voice stage timing: "
                        f"silence={wait_silence:.2f} "
                        f"stt={_fmt(self._stage_p50(guild_id, 'stt'))} "
                        f"decision={_fmt(self._stage_p50(guild_id, 'decision'))} "
                        f"ai_first={_fmt(ai_first)} "
                        f"tts_first={tts_first:.2f} "
                        f"ttfa={ttfa:.2f}"
                    )

            self.response_engine.record_response(guild_id)
            voice_state.update_spoke(guild_id)
            voice_state.increment_messages(guild_id)
            record_voice_reply("ok")

        except Exception as e:
            log.error(f"Voice message handling failed: {e}", exc_info=True)
        finally:
            lock.release()
            voice_state.set_bot_playing(guild_id, False)
            self.turn_detector.mark_bot_playing(guild_id, False)
            voice_state.set_turn_state(guild_id, StateTurnState.IDLE)

    async def _play_backchannel(self, guild_id: int, text: str, language: str = "en") -> None:
        """Play a brief backchannel acknowledgment."""
        from voice import readiness

        if not readiness.is_ready():
            log.debug("TTS not ready yet; dropping backchannel")
            return

        try:
            # Pick a backchannel in the detected language
            ack_lang = language if language in ("ja", "en") else "en"
            ack_text = self.backchannel.pick_backchannel(guild_id, language=ack_lang)

            # Generate TTS for the backchannel
            wav_bytes, tts_sr, engine = await tts.generate_tts(ack_text)
            audio = tts.wav_bytes_to_array(wav_bytes, target_sr=16000)

            # Apply RVC only for the edge engine (sovits is already Alice)
            if engine != "sovits" and rvc.is_available():
                audio = await asyncio.to_thread(rvc.convert, audio, 16000)
                audio = await asyncio.to_thread(rvc.soften_human, audio, 16000)

            # Convert to WAV bytes
            ack_audio = tts.array_to_wav_bytes(audio, sample_rate=16000)

            # Play (non-interruptible)
            await self.mixer.play_backchannel(guild_id, ack_audio)

        except Exception as e:
            log.error(f"Backchannel failed: {e}", exc_info=True)

    async def _play_soft_phrase(self, guild_id: int, user_id: int, context_text: str, language: str = "en") -> None:
        """Play a short clarification phrase when Alice's name is mentioned but not directly addressed."""
        from voice import readiness, rvc, soft_phrases, tts

        if not readiness.is_ready():
            log.debug("TTS not ready; dropping soft phrase")
            return

        # Rate limit: same cooldown as full responses
        now = time.time()
        last = self._last_reply.get(guild_id, 0)
        if now - last < config.VOICE_REPLY_COOLDOWN:
            log.debug(f"Soft phrase rate limited for guild {guild_id}")
            return
        self._last_reply[guild_id] = now

        try:
            # Pick a phrase appropriate for the detected/response language
            response_lang = config.VOICE_RESPONSE_LANGUAGE
            if response_lang == "auto":
                response_lang = language or "en"
            lang_code = "ja" if response_lang == "ja" else "en"

            # In `address` mode, multi-speaker clips warrant a more tentative
            # clarification ("Hmm?" instead of "did I hear my name?")
            is_multi = (
                config.VOICE_DIARIZE_MODE == "address"
                and self._diarize_stats.get(guild_id, {}).get("speaker_count", 1) > 1
            )
            phrase = soft_phrases.pick_soft_phrase(lang_code, tentative=is_multi)
            log.info(f"Soft phrase: '{phrase[:80]}...' (lang={lang_code})")

            # Generate TTS
            wav_bytes, tts_sr, engine = await tts.generate_tts(phrase)
            audio = tts.wav_bytes_to_array(wav_bytes, target_sr=16000)

            # Apply RVC only for the edge engine (sovits is already Alice)
            if engine != "sovits" and rvc.is_available():
                audio = await asyncio.to_thread(rvc.convert, audio, 16000)
                audio = await asyncio.to_thread(rvc.soften_human, audio, 16000)

            # Convert to WAV bytes
            ack_audio = tts.array_to_wav_bytes(audio, sample_rate=16000)

            # Play (non-interruptible, like a backchannel)
            await self.mixer.play_backchannel(guild_id, ack_audio)

            # Record the utterance + stats
            self.conversation.add_utterance(
                guild_id, self.bot.user.id, phrase,
                language="unknown", is_bot=True,
            )
            self.response_engine.record_soft_response(guild_id)
            voice_state.update_spoke(guild_id)
            voice_state.increment_messages(guild_id)

        except Exception as e:
            log.error(f"Soft phrase failed: {e}", exc_info=True)

    def start_monitor(self, guild_id: int) -> None:
        """Start monitoring a guild for completed speech."""
        if guild_id in self._monitor_tasks:
            return

        self._monitor_tasks[guild_id] = asyncio.create_task(
            self._monitor_loop(guild_id)
        )

    def stop_monitor(self, guild_id: int) -> None:
        """Stop monitoring a guild."""
        task = self._monitor_tasks.pop(guild_id, None)
        if task:
            task.cancel()

    async def _monitor_loop(self, guild_id: int) -> None:
        """Monitor loop that checks for completed speech segments."""
        log.info(f"Starting speech monitor for guild {guild_id}")

        while True:
            try:
                await asyncio.sleep(0.1)  # 100ms check interval (sample-based timing)

                sink = self.sinks.get(guild_id)
                if not sink:
                    break

                # Check each user for completed speech
                for user_id in list(sink.buffers.keys()):
                    # Live mode adaptive turn-end: private calls get the
                    # snappier threshold; shared/unknown calls keep the
                    # conservative one (the utterer may be talking to
                    # someone else -- mid-sentence cuts are worse).
                    private = not self._others_in_channel(guild_id, user_id)
                    threshold = (
                        config.VOICE_SILENCE_THRESHOLD_PRIVATE
                        if self._live and private else None
                    )

                    # Predictive STT: overlap an early decode with the
                    # remaining silence before the gate fires.
                    self._maybe_spawn_predictive(guild_id, user_id)

                    if sink.check_speech_complete(user_id, silence_threshold=threshold):
                        # Silence elapsed at the gate (for the timing line)
                        obs = sink.observation(user_id)
                        gate_silence = (
                            obs["silence_duration"] if obs else None
                        )
                        # Pull audio NOW to clear the buffer atomically, preventing
                        # re-triggering while the async task runs.
                        audio = sink.get_audio(user_id)
                        if audio is not None and len(audio) > 0:
                            asyncio.create_task(
                                self._process_speech(
                                    guild_id, user_id, silence=gate_silence, audio=audio
                                )
                            )

                # Periodic vitals (30s): mic packet flow + VAD speech ratio.
                # A dead-quiet mic shows up as pkts rising with speech=0;
                # a hot/noisy mic as a speech percentage near 100.
                now_v = time.monotonic()
                if now_v - self._last_vitals >= 30.0:
                    self._last_vitals = now_v
                    for uid, st in sink._write_stats.items():
                        pct = (
                            100.0 * st["speech"] / st["total"]
                            if st["total"] else 0.0
                        )
                        log.info(
                            "sink vitals: user=%s pkts=%d speech=%d (%.1f%%) open_buffers=%d",
                            uid, st["total"], st["speech"], pct, len(sink.buffers),
                        )

                # Update turn state from the monitor loop, which is the
                # authoritative real-time activity detector. Set it in both
                # voice_state and the decision engine's turn_detector so they
                # stay in sync.
                session = voice_state.get_session(guild_id)
                if self.mixer.is_playing(guild_id):
                    # Alice is playing TTS - that's the turn state. Silero
                    # would also see Alice's voice as an active speaker, so we
                    # short-circuit here instead of letting the monitor mask
                    # the responding state with SINGLE_SPEAKER.
                    self.turn_detector.set_activity_state(
                        guild_id, TurnState.BOT_RESPONDING
                    )
                    voice_state.set_turn_state(guild_id, StateTurnState.BOT_RESPONDING)
                elif session:
                    active = sink._speaking.copy()
                    if len(active) == 0:
                        # Check lull vs idle
                        last_spoke = max(
                            (sink.buffers[uid].silence_duration for uid in sink.buffers),
                            default=float("inf")
                        )
                        if last_spoke >= config.VOICE_LULL_THRESHOLD:
                            state = StateTurnState.LULL
                        elif last_spoke >= config.VOICE_TURN_PAUSE_THRESHOLD:
                            state = StateTurnState.PAUSE
                        else:
                            state = StateTurnState.IDLE
                    elif len(active) == 1:
                        state = StateTurnState.SINGLE_SPEAKER
                    else:
                        state = StateTurnState.MULTIPLE_SPEAKERS

                    voice_state.set_turn_state(guild_id, state)
                    self.turn_detector.set_activity_state(
                        guild_id, TurnState(state.value)
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Monitor loop error: {e}", exc_info=True)
                await asyncio.sleep(1)

        log.info(f"Speech monitor ended for guild {guild_id}")

    async def process_audio(
        self,
        guild_id: int,
        user_id: int,
        audio: np.ndarray,
        sample_rate: int = 48000,
        preset_result: dict | None = None,
    ) -> tuple[str, str] | None:
        """
        Process received audio through the full pipeline.
        Returns (text, language) tuple, or None if processing failed.

        preset_result, when given, is a predictive decode's {text,
        language, ...} for this utterance (voice.pipeline takes care of
        freshness); the resample + local transcribe stages are skipped.
        """
        if self._processing.get(guild_id, False):
            log.debug(f"Already processing for guild {guild_id}, skipping")
            return None

        self._processing[guild_id] = True

        try:
            if preset_result is not None:
                # Predictive decode finished in time: reuse it. The raw
                # clip still goes through capture; the per-utterance
                # hallucination filters below run exactly as after a
                # local decode.
                audio_16k = None
                result = preset_result
            else:
                # Resample to 16kHz for STT
                if sample_rate != 16000:
                    audio_16k = scipy.signal.resample(
                        audio, int(len(audio) * 16000 / sample_rate)
                    ).astype(np.float32)
                else:
                    audio_16k = audio.astype(np.float32)

                # Hosted STT (Phase 3, VOICE_STT_HOSTED, default off):
                # Groq's Whisper endpoint first -- it freezes no local
                # GPU and is usually faster end-to-end; the local engine
                # is the fallback when unset/unreachable.
                result = None
                if config.VOICE_STT_HOSTED:
                    from voice import stt_hosted

                    result = await stt_hosted.transcribe(audio_16k, 16000)
                    if result is None:
                        log.debug(
                            "Hosted STT unavailable; falling back to local engine"
                        )
                        # The local engine lazy-loads INSIDE transcribe; a
                        # cold whisper-medium load (~15-30 s on CPU) would
                        # stall THIS turn's reply. Preload in the background
                        # so the next fallback turn is fast instead.
                        _schedule_local_stt_prewarm()

                if result is None:
                    # Serialize with any in-flight predictive decode --
                    # faster-whisper's model is not thread-safe. Bounded:
                    # a wedged decode (VRAM-starved CUDA hang) must never
                    # stall the pipeline forever.
                    async with self._stt_lock(guild_id):
                        try:
                            result = await asyncio.wait_for(
                                asyncio.to_thread(
                                    stt.transcribe, audio_16k, 16000
                                ),
                                timeout=config.VOICE_STT_DECODE_TIMEOUT,
                            )
                        except TimeoutError:
                            log.error(
                                f"Local STT exceeded "
                                f"{config.VOICE_STT_DECODE_TIMEOUT}s (wedged "
                                "decode; STT lock released) -- dropping turn"
                            )
                            return None

            # STT eval capture: save the raw clip + sidecar for offline
            # scoring (manage.py eval-stt). Before the hallucination filter
            # so accent-caused empty/garbage transcripts are kept too.
            if config.VOICE_STT_CAPTURE:
                await asyncio.to_thread(
                    _capture_clip, guild_id, user_id, audio, sample_rate, result
                )

            # At this point result is guaranteed to be a dict with "text" and "language"
            # keys because:
            # - preset_result, if provided, comes from _predictive_decode which returns
            #   {"text": ..., "language": ...}
            # - stt_hosted.transcribe returns dict or None (None handled above)
            # - stt.transcribe returns dict
            result_dict: dict[str, str] = result  # type: ignore[assignment]

            text = result_dict["text"].strip()
            if not text:
                log.debug("Empty transcription, skipping")
                return None

            # Drop single-word Whisper hallucinations on short/noise clips.
            if audio_16k is not None:
                audio_dur = len(audio_16k) / 16000.0
            else:
                audio_dur = len(audio) / float(sample_rate)
            text_l = text.lower().strip(".,!?;:\"'")
            if audio_dur < 2.0 and text_l in _STT_HALLUCINATIONS:
                record_voice_stt_filtered("single_word")
                log.debug(
                    f"Filtered STT hallucination {text!r} "
                    f"({audio_dur:.2f}s audio)"
                )
                return None

            # Additional hallucination checks (prompt echoes, self-address
            # loops, silence impostors) - applies to both whisper and
            # moonshine; drops the utterance before it can seed a response.
            if _is_stt_hallucination(text, audio_dur):
                record_voice_stt_filtered("hallucination")
                log.debug(
                    f"Filtered STT hallucination {text!r} "
                    f"({audio_dur:.2f}s audio)"
                )
                return None

            language = result_dict["language"]
            log.info(f"Transcribed: '{text}' (lang={language})")
            # Clear the user's buffer so we don't reprocess the same audio
            if guild_id in self.sinks:
                self.sinks[guild_id].clear_user(user_id)
            return text, language

        except Exception as e:
            log.error(f"Audio processing failed: {e}", exc_info=True)
            return None
        finally:
            self._processing[guild_id] = False

    async def generate_speech(
        self,
        text: str,
        guild_id: int,
        mood: str = "neutral",
    ) -> bytes | None:
        """
        Generate speech audio from text.
        1. Generate TTS audio
        2. Apply RVC voice conversion
        3. Soften for natural sound
        4. Return WAV bytes for discord.py playback
        """
        try:
            # Generate TTS
            wav_bytes, tts_sr, engine = await tts.generate_tts(text)

            # Convert to numpy
            audio = tts.wav_bytes_to_array(wav_bytes, target_sr=16000)

            # Apply RVC only for the edge engine (sovits is already Alice)
            if engine != "sovits" and rvc.is_available():
                audio = await asyncio.to_thread(
                    rvc.convert, audio, 16000
                )
                # Post-process for natural sound
                audio = await asyncio.to_thread(
                    rvc.soften_human, audio, 16000
                )

            # Convert back to WAV bytes (discord.py expects WAV/PCM)
            wav_bytes = tts.array_to_wav_bytes(audio, sample_rate=16000)

            return wav_bytes

        except Exception as e:
            log.error(f"Speech generation failed: {e}", exc_info=True)
            return None

    async def send_audio(
        self,
        guild_id: int,
        audio_data: bytes,
    ) -> bool:
        """Send audio data to the voice channel."""
        return await self.mixer.play_tts(guild_id, audio_data)

    def get_stats(self, guild_id: int) -> dict:
        """Get comprehensive pipeline statistics."""
        live_stages: dict[str, float] = {}
        for stage, dq in self._stage_times.get(guild_id, {}).items():
            if dq:
                s = sorted(dq)
                live_stages[stage] = round(s[len(s) // 2], 3)
        pred_guild = [
            e for (g, _u), e in self._pred_stt.items() if g == guild_id
        ]
        return {
            "conversation": {
                "utterances": len(self.conversation.get_recent(guild_id)),
                "speakers": self.conversation.get_speaker_count(guild_id),
                "alice_silence": self.conversation.get_alice_silence_duration(guild_id),
            },
            "turn": {
                "state": self.turn_detector.get_state(guild_id).value,
                "duration": self.turn_detector.get_turn_duration(guild_id),
            },
            "response": self.response_engine.get_stats(guild_id),
            "backchannel": self.backchannel.get_stats(guild_id),
            "mixer": self.mixer.get_stats(guild_id),
            "diarization": {
                "speakers_last": self._diarize_stats.get(guild_id, {}).get("speaker_count", 0),
                "multi_speaker_clips": self._diarize_stats.get(guild_id, {}).get("multi_speaker_count", 0),
            },
            "live": {
                "enabled": self._live,
                "ack_bank": self.ack_bank.stats(),
                "acks_spoken": self._ack_counts.get(guild_id, 0),
                "predictive": {
                    "in_flight": sum(
                        1 for e in pred_guild
                        if e.get("task") is not None and not e["task"].done()
                    ),
                    "cached": sum(
                        1 for e in pred_guild if e.get("result") is not None
                    ),
                },
                "stage_p50": live_stages,
            },
        }

    def cleanup(self, guild_id: int) -> None:
        """Clean up resources for a guild."""
        self.stop_monitor(guild_id)
        if guild_id in self.sinks:
            self.sinks[guild_id].cleanup()
            del self.sinks[guild_id]
        self._processing.pop(guild_id, None)
        self._last_reply.pop(guild_id, None)
        self._diarize_stats.pop(guild_id, None)
        self._ack_counts.pop(guild_id, None)
        self._ack_in_flight.discard(guild_id)
        self._stage_times.pop(guild_id, None)
        for key in [k for k in self._pred_stt if k[0] == guild_id]:
            entry = self._pred_stt.pop(key, None)
            if entry and entry.get("task") is not None and not entry["task"].done():
                entry["task"].cancel()
        self._stt_locks.pop(guild_id, None)
        self.conversation.cleanup(guild_id)
        self.turn_detector.cleanup(guild_id)
        self.response_engine.cleanup(guild_id)
        self.backchannel.cleanup(guild_id)
        self.mixer.cleanup(guild_id)


# Global singleton
voice_pipeline = None


_stt_prewarm_scheduled = False
_stt_prewarm_lock = threading.Lock()


def _schedule_local_stt_prewarm() -> None:
    """Preload the local STT engine once, in the background, after the
    hosted endpoint has failed at least once (rate limit / outage).

    The first post-failure turn still pays the load inline (racy), but
    every turn after that is warm -- and with hosted back up, the
    preload simply sits unused in RAM until `stt.unload()` at shutdown.
    """
    global _stt_prewarm_scheduled
    with _stt_prewarm_lock:
        if _stt_prewarm_scheduled:
            return
        _stt_prewarm_scheduled = True
    log.info("Hosted STT failed once -- preloading the local engine in the background")
    try:
        threading.Thread(target=stt.preload, name="stt-fallback-preload", daemon=True).start()
    except Exception as e:  # noqa: BLE001
        log.warning(f"Local STT fallback preload failed to schedule: {e}")


def init_pipeline(bot) -> VoicePipeline:
    """Initialize the global voice pipeline and preload heavy models."""
    global voice_pipeline
    voice_pipeline = VoicePipeline(bot)

    # Preload VAD (fast, ~3s) synchronously so first packet never blocks.
    try:
        from voice import vad as _vad
        if not _vad.is_model_loaded():
            t0 = time.time()
            _vad.preload()
            log.info(f"VAD preloaded in {time.time()-t0:.1f}s")
    except Exception as e:
        log.warning(f"VAD preload failed (will RMS-fallback): {e}")

    # Preload STT (slow, ~15-35s first CUDA call) in background so bot
    # startup isn't delayed; first utterance may still warm up if it
    # arrives before preload finishes. Hosted STT decodes nothing locally
    # -- keep its VRAM free for TTS instead.
    if config.VOICE_STT_HOSTED:
        log.info("STT preload skipped (VOICE_STT_HOSTED=true; local engine stays lazy)")
    else:
        try:
            from voice import stt as _stt
            if not _stt.is_loaded():
                import threading
                threading.Thread(target=_stt.preload, name="stt-preload", daemon=True).start()
                log.info("STT preload started in background")
        except Exception as e:
            log.warning(f"STT preload schedule failed: {e}")

    # Defer the diarization preload until a TTS engine is confirmed
    # usable: pyannote holds ~1GB of VRAM once loaded, and loading it
    # alongside whisper while the GPT-SoVITS server boots contributed
    # to the 2026-10-25 overnight stall (the server's first boot
    # crawled for the whole startup budget while the GPU was contended;
    # a retry after everything else had settled came up in 35s). Until
    # TTS settles, diarization stays lazy-loaded on first use.
    try:
        import threading as _threading

        from voice import diarize as _diarize
        from voice import readiness
        if config.VOICE_DIARIZE_MODE != "none":
            def _deferred_diarize_warmup() -> None:
                _threading.Thread(
                    target=_diarize.warmup, name="diarize-preload", daemon=True
                ).start()

            readiness.kick(on_ready=_deferred_diarize_warmup)
            log.info("Diarization preload deferred until TTS is ready")
    except Exception as e:
        log.warning(f"Diarization preload schedule failed: {e}")

    # Preload RVC so the first spoken reply already has Alice's voice.
    # Skipped for the sovits engine (already Alice; loads lazily on fallback).
    try:
        from voice import rvc as _rvc
        if config.VOICE_TTS_ENGINE == "sovits":
            log.info("RVC preload skipped (sovits engine)")
        elif _rvc.is_available() and not _rvc.is_loaded():
            import threading
            threading.Thread(target=_rvc.preload, name="rvc-preload", daemon=True).start()
            log.info("RVC preload started in background")
        elif not _rvc.is_available():
            log.info("RVC not available (disabled or model missing)")
    except Exception as e:
        log.warning(f"RVC preload schedule failed: {e}")

    # Live mode: build the banked address-ack ("Hm?") cache once a TTS
    # engine is confirmed -- it defers to the same readiness hook that
    # delays the diarization warmup so neither fights the GPU.
    try:
        from voice import readiness as _readiness

        if config.VOICE_LIVE_MODE and config.VOICE_ACK_ENABLED:
            def _deferred_ack_bank_build() -> None:
                async def _build() -> None:
                    if voice_pipeline is not None:
                        try:
                            await voice_pipeline.ack_bank.build()
                        except Exception as e:
                            log.warning(f"Backchannel bank build failed: {e}")

                asyncio.get_running_loop().create_task(_build())

            # kick() queues the hook for the next readiness transition;
            # if the engine already settled during init, fire now.
            if _readiness.is_ready():
                _deferred_ack_bank_build()
            else:
                _readiness.kick(on_ready=_deferred_ack_bank_build)
            log.info("Backchannel bank build deferred until TTS is ready")
    except Exception as e:
        log.warning(f"Backchannel bank build schedule failed: {e}")

    # Start the TTS readiness loop (bring up GPT-SoVITS / decide fallback).
    try:
        from voice import readiness as _readiness
        _readiness.kick()
    except Exception as e:
        log.warning(f"Readiness kick failed: {e}")

    return voice_pipeline
