"""
voice/__init__.py
==================
Voice processing package for AliceShimadaBot.
Provides STT, TTS, RVC voice conversion, diarization,
and conversation-aware voice interaction.
"""

from __future__ import annotations

from voice import dave_patch
dave_patch.apply()

from voice.pipeline import VoicePipeline, init_pipeline
from voice.state import VoiceSession, VoiceStateManager, TurnState, voice_state
from voice.sink import AliceVoiceSink, AudioBuffer, SimpleSink
from voice.conversation import ConversationTracker, Utterance
from voice.turn_detector import TurnDetector, TurnState as DetectorTurnState
from voice.response import ResponseDecisionEngine, ResponseDecision
from voice.backchannel import BackchannelGenerator
from voice.mixer import VoiceMixer
from voice.streaming_source import StreamingAudioSource
from voice.streaming_tts import stream_response, split_sentences
from voice.vad import SileroVAD, preload as preload_vad, is_model_loaded as vad_loaded


def get_voice_pipeline():
    """Get the global voice pipeline instance (avoids import-time None)."""
    from voice import pipeline
    return pipeline.voice_pipeline


# Lazy property for voice_pipeline
class _VoicePipelineProxy:
    """Proxy that defers to the actual pipeline after init."""
    def __getattr__(self, name):
        from voice import pipeline
        if pipeline.voice_pipeline is None:
            raise RuntimeError("Voice pipeline not initialized. Call init_pipeline() first.")
        return getattr(pipeline.voice_pipeline, name)

voice_pipeline = _VoicePipelineProxy()

__all__ = [
    "VoicePipeline",
    "VoiceSession",
    "VoiceStateManager",
    "TurnState",
    "AliceVoiceSink",
    "AudioBuffer",
    "SimpleSink",
    "ConversationTracker",
    "Utterance",
    "TurnDetector",
    "ResponseDecisionEngine",
    "ResponseDecision",
    "BackchannelGenerator",
    "VoiceMixer",
    "StreamingAudioSource",
    "stream_response",
    "split_sentences",
    "SileroVAD",
    "preload_vad",
    "vad_loaded",
    "init_pipeline",
    "get_voice_pipeline",
    "voice_pipeline",
    "voice_state",
]
