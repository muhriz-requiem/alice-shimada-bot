"""
voice/response.py
==================
Response decision engine for voice interactions.
Determines whether Alice should respond based on context,
speaker roles, conversation state, and character personality.
"""

from __future__ import annotations

import random
import time
from enum import Enum
from typing import Optional

from bot.logging_setup import get_logger
import config
from voice.conversation import ConversationTracker
from voice.turn_detector import TurnDetector, TurnState

log = get_logger("voice.response")


class ResponseDecision(Enum):
    """Decision on whether to respond."""
    RESPOND = "respond"           # Generate and play response
    RESPOND_SOFT = "soft"         # Play short clarification (name was mentioned)
    BACKCHANNEL = "backchannel"  # Play brief acknowledgment
    SKIP = "skip"                 # Stay silent
    WAIT = "wait"                 # Wait for more context


class ResponsePriority(Enum):
    """Priority level for responses."""
    CRITICAL = 0   # Owner direct address
    HIGH = 1       # Direct mention
    MEDIUM = 2     # Relevant to conversation
    LOW = 3        # Conversation lull, joining
    MINIMAL = 4    # Passive observation


class ResponseDecisionEngine:
    """
    Decides whether Alice should respond in a voice conversation.
    
    Implements the I1/I2/S1/S2 framework from research:
    - I1 (Explicit Address): Direct mention -> Always respond
    - I2 (Contextual Intervention): Topic shift, lull -> Maybe respond
    - S1 (No Reference): Others chatting -> Stay silent
    - S2 (Referenced, not addressed): Third-party mention -> Stay silent
    
    Alice's personality (shy/quiet) means:
    - Conservative response probability
    - Prefers to be spoken to directly
    - Joins conversations reluctantly
    - Owner always gets priority
    """

    def __init__(
        self,
        conversation: ConversationTracker,
        turn_detector: TurnDetector,
    ):
        self.conversation = conversation
        self.turn_detector = turn_detector
        self._last_response: dict[int, float] = {}  # guild_id -> timestamp
        self._response_count: dict[int, int] = {}   # guild_id -> count
        self._cooldown_until: dict[int, float] = {}  # guild_id -> timestamp
        self._soft_response_count: dict[int, int] = {}  # guild_id -> count

    def should_respond(
        self,
        guild_id: int,
        user_id: int,
        text: str,
        is_owner: bool = False,
        diarize_result: Optional[list[dict]] = None,
    ) -> tuple[ResponseDecision, ResponsePriority]:
        """
        Decide whether Alice should respond to an utterance.
        
        Returns:
            (decision, priority) tuple
        """
        now = time.time()

        # --- Hard BLOCKS ---

        # Cooldown check
        cooldown_until = self._cooldown_until.get(guild_id, 0)
        if now < cooldown_until:
            return ResponseDecision.SKIP, ResponsePriority.MINIMAL

        # Rate limiting
        last = self._last_response.get(guild_id, 0)
        if now - last < config.VOICE_REPLY_COOLDOWN:
            return ResponseDecision.SKIP, ResponsePriority.MINIMAL

        # --- Response mode gate (VOICE_RESPONSE_MODE) ---

        mode = (getattr(config, "VOICE_RESPONSE_MODE", "smart") or "smart").strip().lower()

        if mode == "observer":
            # Passive: never generate a spoken response
            return ResponseDecision.SKIP, ResponsePriority.MINIMAL

        if mode == "mention_only":
            # Only direct addresses get through
            if self._is_direct_address(guild_id, text):
                if is_owner:
                    return ResponseDecision.RESPOND, ResponsePriority.CRITICAL
                return ResponseDecision.RESPOND, ResponsePriority.HIGH
            return ResponseDecision.SKIP, ResponsePriority.MINIMAL

        if mode == "always":
            # Respond to every non-empty utterance.
            # Cooldowns still apply above.
            if is_owner:
                return ResponseDecision.RESPOND, ResponsePriority.CRITICAL
            return ResponseDecision.RESPOND, ResponsePriority.HIGH

        # mode == "smart" (default): fall through to full decision logic.
        # Owner priority: always respond even without direct address.
        if is_owner and config.VOICE_OWNER_PRIORITY:
            return ResponseDecision.RESPOND, ResponsePriority.CRITICAL

        # --- I1: Explicit Address ---

        if self._is_direct_address(guild_id, text):
            if is_owner:
                return ResponseDecision.RESPOND, ResponsePriority.CRITICAL
            return ResponseDecision.RESPOND, ResponsePriority.HIGH

        # --- Turn State Analysis ---

        turn_state = self.turn_detector.get_state(guild_id)

        # Don't interrupt multiple speakers
        if turn_state == TurnState.MULTIPLE_SPEAKERS:
            return ResponseDecision.SKIP, ResponsePriority.MINIMAL

        # Don't interrupt unless barge-in is enabled and it's the owner
        if turn_state == TurnState.BOT_RESPONDING:
            if is_owner and config.VOICE_BARGE_IN_ENABLED:
                return ResponseDecision.RESPOND, ResponsePriority.CRITICAL
            return ResponseDecision.SKIP, ResponsePriority.MINIMAL

        # --- S1/S2: No Reference or Not Addressed ---

        # Check if Alice is being talked about (third person)
        alice_keywords = [kw.lower() for kw in config.VOICE_MENTION_KEYWORDS]
        text_lower = text.lower()
        mentions_alice = any(kw in text_lower for kw in alice_keywords)

        # If mentioned in third person but not directly addressed
        if mentions_alice:
            # Multi-speaker clip: the mention likely came from background
            # chatter and the transcript is unreliable; stay silent.
            if diarize_result and len({s["speaker"] for s in diarize_result}) > 1:
                return ResponseDecision.SKIP, ResponsePriority.MINIMAL
            if config.VOICE_NAME_MENTION_RESPONSE == "soft":
                return ResponseDecision.RESPOND_SOFT, ResponsePriority.MEDIUM
            return ResponseDecision.SKIP, ResponsePriority.MINIMAL

        # --- I2: Contextual Intervention ---

        # Owner gets special treatment - always respond in conversation lull
        if is_owner and turn_state == TurnState.LULL:
            owner_chance = 0.7
            if random.random() < owner_chance:
                return ResponseDecision.RESPOND, ResponsePriority.MEDIUM

        # Conversation lull - Alice might join
        if turn_state == TurnState.LULL:
            alice_silence = self.conversation.get_alice_silence_duration(guild_id)
            # Only join if Alice hasn't spoken recently
            if alice_silence > 10:
                # Alice is shy - low probability of joining
                join_chance = config.VOICE_LULL_JOIN_PROBABILITY
                if random.random() < join_chance:
                    return ResponseDecision.RESPOND, ResponsePriority.LOW

        # --- Backchannel Detection ---

        # Brief utterances often just need acknowledgment
        if self._should_backchannel(guild_id, user_id, text, diarize_result=diarize_result):
            return ResponseDecision.BACKCHANNEL, ResponsePriority.LOW

        # --- Default: Stay Silent ---

        return ResponseDecision.SKIP, ResponsePriority.MINIMAL

    def _should_backchannel(
        self, guild_id: int, user_id: int, text: str,
        diarize_result: Optional[list[dict]] = None,
    ) -> bool:
        """Determine if a brief acknowledgment is appropriate."""
        if not config.VOICE_BACKCHANNEL_ENABLED:
            return False

        # Don't backchannel over multi-speaker clips — the transcript is too
        # unreliable to warrant an acknowledgment.
        if diarize_result and len({s["speaker"] for s in diarize_result}) > 1:
            return False

        # Check turn state — backchannel during speech or natural pauses
        turn_state = self.turn_detector.get_state(guild_id)
        if turn_state not in (TurnState.SINGLE_SPEAKER, TurnState.PAUSE):
            return False

        # Don't backchannel if Alice just spoke
        alice_silence = self.conversation.get_alice_silence_duration(guild_id)
        if alice_silence < 2.0:
            return False

        # Require real content words before acknowledging — this reduces
        # backchannel responses to garbage or filler-only transcripts.
        content_words = [w for w in text.split() if len(w) > 2]
        if len(content_words) < 3:
            return False

        # Questions sometimes need acknowledgment
        if text.rstrip().endswith("?") or text.rstrip().endswith("？"):
            return random.random() < 0.2

        return False

    def _is_direct_address(self, guild_id: int, text: str) -> bool:
        """
        Check if text is a direct address to Alice (I1 category).
        Must be: "alice, ...", "alice [verb]", or "hello alice, ..."
        (keyword in first 3 words) — NOT: "alice is ..." (S2 third-person).
        """
        keywords = [kw.lower() for kw in config.VOICE_MENTION_KEYWORDS]
        text_lower = text.lower().strip()

        # Third-person copula verbs — these indicate S2, not I1
        third_person_verbs = {
            "is", "was", "are", "were", "has", "had", "will", "would",
            "could", "should", "might", "can", "may", "does", "did",
            "looks", "sounds", "seems", "feels", "appears", "became",
        }

        words = [w.strip(".,!?;:\"'()…") for w in text_lower.split()]
        words = [w for w in words if w]

        for kw in keywords:
            # Direct address: keyword followed by comma (strongest signal)
            if text_lower.startswith(kw + ","):
                return True

            # Direct address: keyword at start, followed by space + non-copula word
            if text_lower.startswith(kw + " "):
                after = text_lower[len(kw) + 1:].strip()
                first_word = after.split()[0] if after else ""
                if first_word.rstrip("?？!") not in third_person_verbs:
                    return True

            # Vocative in first 3 words: "hello alice, ...", "hey alice ..."
            if kw in words[:3]:
                idx = words.index(kw)
                if idx > 0:
                    # Named in greeting position — always a direct address,
                    # regardless of what follows.
                    return True

        return False

    def record_response(self, guild_id: int) -> None:
        """Record that Alice responded."""
        now = time.time()
        self._last_response[guild_id] = now
        self._response_count[guild_id] = self._response_count.get(guild_id, 0) + 1
        # Set cooldown
        self._cooldown_until[guild_id] = now + config.VOICE_REPLY_COOLDOWN

    def record_soft_response(self, guild_id: int) -> None:
        """Record that Alice responded with a soft clarification phrase."""
        now = time.time()
        self._last_response[guild_id] = now
        self._soft_response_count[guild_id] = self._soft_response_count.get(guild_id, 0) + 1
        # Set cooldown
        self._cooldown_until[guild_id] = now + config.VOICE_REPLY_COOLDOWN

    def get_stats(self, guild_id: int) -> dict:
        """Get response statistics for a guild."""
        return {
            "total_responses": self._response_count.get(guild_id, 0),
            "total_soft_responses": self._soft_response_count.get(guild_id, 0),
            "last_response": self._last_response.get(guild_id, 0),
            "cooldown_until": self._cooldown_until.get(guild_id, 0),
        }

    def cleanup(self, guild_id: int) -> None:
        """Clean up guild data."""
        self._last_response.pop(guild_id, None)
        self._response_count.pop(guild_id, None)
        self._cooldown_until.pop(guild_id, None)
        self._soft_response_count.pop(guild_id, None)
