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
import time
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal

from bot.logging_setup import get_logger
import config
from character import prompt_builder
from memory import store as memory_store
from social.identities import get_identity_async
from voice import rvc, stt, tts
from voice.diarize import diarize as _diarize_fn
from voice.conversation import ConversationTracker
from voice.turn_detector import TurnDetector, TurnState
from voice.response import ResponseDecisionEngine, ResponseDecision, ResponsePriority
from voice.backchannel import BackchannelGenerator
from voice.mixer import VoiceMixer
from voice.sink import AliceVoiceSink
from voice.state import voice_state, TurnState as StateTurnState
from voice.streaming_source import StreamingAudioSource
from voice.streaming_tts import stream_response, split_sentences

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
    for kw in mention_kws:
        if text_l.startswith(kw + ",") or text_l.startswith(kw + " "):
            return True
    return False


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
        if phrase and len(phrase.split()) <= 6:
            addr_hits += 1
        elif phrase == "" and rest.strip() == "":
            addr_hits += 1
    if addr_hits >= 2:
        return True

    # --- Rule 3: silence impostor ---
    # Remove filler words and count remaining content words.
    cleaned = re.sub(
        r"(?:^|\s)(?:uh|um|mm|hmm|ah|ahh)\s*(?:$|\s)", " ", normalized
    )
    content_words = [w for w in cleaned.split() if w and len(w) > 2]
    if len(content_words) < 2 and audio_dur >= 2.5:
        return True

    return False


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

    def get_sink(self, guild_id: int) -> AliceVoiceSink:
        """Get or create the audio sink for a guild."""
        if guild_id not in self.sinks:
            self.sinks[guild_id] = AliceVoiceSink(guild_id=guild_id)
        return self.sinks[guild_id]

    async def _process_speech(self, guild_id: int, user_id: int) -> None:
        """Process completed speech segment with conversation awareness."""
        sink = self.sinks.get(guild_id)
        if not sink:
            return

        audio = sink.get_audio(user_id)
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
        result = await self.process_audio(guild_id, user_id, audio, 48000)

        if result:
            text, language = result

            # Non-blocking diarization when enabled (multi-speaker analysis)
            diarize_result = None
            if config.VOICE_DIARIZE_MODE != "none":
                diarize_result = await asyncio.to_thread(_diarize_fn, audio, 48000)
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

            # Decide whether to respond
            decision, priority = self.response_engine.should_respond(
                guild_id, user_id, text, is_owner=is_owner, diarize_result=diarize_result,
            )

            log.info(
                f"Response decision: {decision.value} (priority={priority.value}) "
                f"for '{text[:50]}...' from user {user_id}"
            )

            if decision == ResponseDecision.RESPOND:
                await self._generate_and_respond(guild_id, user_id, text, is_owner, language)
            elif decision == ResponseDecision.RESPOND_SOFT:
                await self._play_soft_phrase(guild_id, user_id, text, language)
            elif decision == ResponseDecision.BACKCHANNEL:
                await self._play_backchannel(guild_id, text, language)

    async def _generate_and_respond(
        self, guild_id: int, user_id: int, text: str, is_owner: bool,
        user_language: str = "en",
    ) -> None:
        """Generate AI response and stream it sentence-by-sentence."""
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

            messages = prompt_builder.build_prompt(
                character,
                identity=identity if identity.get("is_authoritative") else None,
                relationship_key=relationship,
                memory_snippets=[],
                history=history,
                user_message=text,
                fast_path=False,
                extra_context=voice_extra,
            )

            session = voice_state.get_session(guild_id)
            channel_id = str(session.channel_id if session else guild_id)

            response = await ai_client.generate_reply(
                messages, user_id=user_id_str, channel_id=channel_id,
                max_tokens=config.VOICE_MAX_TOKENS,
            )

            if not response or response == config.FALLBACK_REPLY:
                # Backend failed (rate limit, parse error, timeout, or
                # empty content -- reasoning ate the token budget) and
                # returned the fallback. Text chat still shows it; on a
                # live call, silence beats repeating "I don't know what
                # to say." every time Groq hiccups.
                log.info("AI reply unavailable (fallback); staying silent in voice")
                return

            log.info(f"AI response: '{response[:100]}...'")
            self.conversation.add_utterance(
                guild_id, self.bot.user.id, response,
                language="unknown", is_bot=True,
            )

            # Stream response sentence-by-sentence
            from voice.streaming_tts import stream_response
            from voice.streaming_source import StreamingAudioSource

            source = StreamingAudioSource()
            await stream_response(response, guild_id, source)

            # Play the streaming source
            vc = voice_state.get_voice_client(guild_id)
            if vc and source:
                vc.play(source, after=lambda e: None)
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

            self.response_engine.record_response(guild_id)
            voice_state.update_spoke(guild_id)
            voice_state.increment_messages(guild_id)

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
        from voice import readiness, tts, rvc, soft_phrases

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
                    if sink.check_speech_complete(user_id):
                        # Process in background to not block monitor
                        asyncio.create_task(self._process_speech(guild_id, user_id))

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
    ) -> Optional[tuple[str, str]]:
        """
        Process received audio through the full pipeline.
        Returns (text, language) tuple, or None if processing failed.
        """
        if self._processing.get(guild_id, False):
            log.debug(f"Already processing for guild {guild_id}, skipping")
            return None

        self._processing[guild_id] = True

        try:
            # Resample to 16kHz for STT
            if sample_rate != 16000:
                audio_16k = scipy.signal.resample(
                    audio, int(len(audio) * 16000 / sample_rate)
                ).astype(np.float32)
            else:
                audio_16k = audio.astype(np.float32)

            # Transcribe
            result = await asyncio.to_thread(
                stt.transcribe, audio_16k, 16000
            )

            # STT eval capture: save the raw clip + sidecar for offline
            # scoring (manage.py eval-stt). Before the hallucination filter
            # so accent-caused empty/garbage transcripts are kept too.
            if config.VOICE_STT_CAPTURE:
                await asyncio.to_thread(
                    _capture_clip, guild_id, user_id, audio, sample_rate, result
                )

            text = result["text"].strip()
            if not text:
                log.debug("Empty transcription, skipping")
                return None

            # Drop single-word Whisper hallucinations on short/noise clips.
            audio_dur = len(audio_16k) / 16000.0
            text_l = text.lower().strip(".,!?;:\"'")
            if audio_dur < 2.0 and text_l in _STT_HALLUCINATIONS:
                log.debug(
                    f"Filtered STT hallucination {text!r} "
                    f"({audio_dur:.2f}s audio)"
                )
                return None

            # Additional hallucination checks (prompt echoes, self-address
            # loops, silence impostors) - applies to both whisper and
            # moonshine; drops the utterance before it can seed a response.
            if _is_stt_hallucination(text, audio_dur):
                log.debug(
                    f"Filtered STT hallucination {text!r} "
                    f"({audio_dur:.2f}s audio)"
                )
                return None

            language = result["language"]
            log.info(f"Transcribed: '{text}' (lang={language})")
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
    ) -> Optional[bytes]:
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
        self.conversation.cleanup(guild_id)
        self.turn_detector.cleanup(guild_id)
        self.response_engine.cleanup(guild_id)
        self.backchannel.cleanup(guild_id)
        self.mixer.cleanup(guild_id)


# Global singleton
voice_pipeline = None


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
    # arrives before preload finishes.
    try:
        from voice import stt as _stt
        if not _stt.is_loaded():
            import threading
            threading.Thread(target=_stt.preload, name="stt-preload", daemon=True).start()
            log.info("STT preload started in background")
    except Exception as e:
        log.warning(f"STT preload schedule failed: {e}")

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

    # Start the TTS readiness loop (bring up GPT-SoVITS / decide fallback).
    try:
        from voice import readiness as _readiness
        _readiness.kick()
    except Exception as e:
        log.warning(f"Readiness kick failed: {e}")

    return voice_pipeline
