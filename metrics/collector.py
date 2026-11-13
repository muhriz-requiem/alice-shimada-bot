"""
metrics/collector.py
=====================
Prometheus metrics for the bot.
"""

from __future__ import annotations

import logging
from collections import deque

from prometheus_client import Counter, Gauge, Histogram, Info

# Bot info
bot_info = Info("alicebot", "AliceShimadaBot information")

# Message metrics
messages_total = Counter(
    "alicebot_messages_total",
    "Total number of messages processed",
    ["direction", "channel_type"],  # direction: in/out, channel_type: dm/guild
)

# AI request metrics
ai_request_duration = Histogram(
    "alicebot_ai_request_duration_seconds",
    "Time spent on AI requests",
    ["backend", "status"],  # status: success/error/timeout
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

ai_requests_total = Counter(
    "alicebot_ai_requests_total",
    "Total number of AI requests",
    ["backend", "status"],
)

# Memory metrics
memory_graph_size = Gauge(
    "alicebot_memory_graph_nodes",
    "Number of nodes in memory graph per user",
    ["user_id"],
)

memory_graph_cap = Gauge(
    "alicebot_memory_graph_cap",
    "Maximum memory graph capacity per user",
)

# Reddit metrics
reddit_posts_total = Gauge(
    "alicebot_reddit_posts_available",
    "Number of posts available in Reddit pool",
    ["subreddit"],
)

reddit_refresh_duration = Histogram(
    "alicebot_reddit_refresh_duration_seconds",
    "Time spent refreshing Reddit posts",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
)

reddit_refresh_total = Counter(
    "alicebot_reddit_refresh_total",
    "Total number of Reddit refreshes",
    ["status"],  # success/error
)

# Error metrics
errors_total = Counter(
    "alicebot_errors_total",
    "Total number of errors",
    ["component", "error_type"],
)

# GIF metrics
gif_sent_total = Counter(
    "alicebot_gif_sent_total",
    "Total number of GIFs sent",
    ["intent", "emotion"],
)

# Emotion metrics
emotion_state = Gauge(
    "alicebot_emotion_state",
    "Current emotion state values",
    ["user_id", "emotion"],
)

# Voice latency (v2.4 live mode). Each histogram bucket edge is a
# decision-relevant boundary: sub-300ms reads as instant acknowledgment,
# ~1.2s reads as "live", >3s reads as a stitching pipeline.
voice_ttfa_seconds = Histogram(
    "alicebot_voice_ttfa_seconds",
    "Voice time-to-first-audio: end-of-speech decision -> first PCM frame",
    buckets=[0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.6, 2.0, 3.0, 5.0, 8.0, 12.0],
)

voice_stage_seconds = Histogram(
    "alicebot_voice_stage_seconds",
    "Per-stage voice pipeline latency (silence/stt/decision/ai/ai_first/tts_first)",
    ["stage"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0, 10.0],
)

voice_ttfa_latest = Gauge(
    "alicebot_voice_ttfa_latest_seconds",
    "Most recently measured voice time-to-first-audio",
)

voice_replies_total = Counter(
    "alicebot_voice_replies_total",
    "Voice reply outcomes",
    ["status"],  # ok / silent / dropped / no_response / soft / backchannel / empty
)

voice_stt_predictive_total = Counter(
    "alicebot_voice_stt_predictive_total",
    "Predictive STT decode outcomes",
    ["outcome"],  # none / hit / hit_wait / stale / miss / failed
)

# Utterances dropped by the STT hallucination filter, before they can
# reach the AI or the conversation history. reason = single_word (known
# one-word Whisper artifacts on short clips) / hallucination (phrase
# loops, prompt echoes, silence impostors). Watch the fire rate: it
# should be occasional, not constant -- a spike means the filter is
# eating real speech and the eval clips need review.
voice_stt_filtered_total = Counter(
    "alicebot_voice_stt_filtered_total",
    "Utterances dropped by the STT hallucination filter",
    ["reason"],
)

# Screen glances (vision): outcomes + latency. ok = comment spoken,
# refused = gate/denylist/no-capture-source, failed = vision/capture error.
glance_total = Counter(
    "alicebot_glance_total",
    "Screen glance outcomes",
    ["outcome"],  # ok / refused / failed
)

glance_latency = Histogram(
    "alicebot_glance_latency_seconds",
    "Time from capture to spoken comment",
    buckets=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
)

# Rolling TTFA observations for the in-process snapshot/p50 (bounded).
_recent_ttfas: deque[float] = deque(maxlen=50)

# Rate limiting metrics
rate_limit_wait = Histogram(
    "alicebot_rate_limit_wait_seconds",
    "Time spent waiting for rate limits",
    ["backend"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)

# Active users
active_users = Gauge(
    "alicebot_active_users",
    "Number of users currently being processed",
)

# Uptime
uptime_seconds = Gauge(
    "alicebot_uptime_seconds",
    "Bot uptime in seconds",
)

# Gateway reconnects. discord.py has no event for the individual retry
# attempts, so these are counted by watching its own
# "Attempting a reconnect in Xs" log lines (drops, network/DNS
# failures) -- see install_gateway_reconnect_listener().
gateway_reconnects_total = Counter(
    "alicebot_gateway_reconnects_total",
    "Discord gateway reconnect attempts (drops, network/DNS failures)",
)

_gateway_reconnect_count = 0

# Initialize bot info
def init_metrics(bot_name: str = "alice", version: str = "2.0.0") -> None:
    """Initialize bot info metric."""
    bot_info.info({"name": bot_name, "version": version})
    install_gateway_reconnect_listener()


def record_message(direction: str, channel_type: str) -> None:
    """Record a message."""
    messages_total.labels(direction=direction, channel_type=channel_type).inc()


def record_ai_request(backend: str, status: str, duration: float) -> None:
    """Record an AI request."""
    ai_requests_total.labels(backend=backend, status=status).inc()
    ai_request_duration.labels(backend=backend, status=status).observe(duration)


def record_reddit_refresh(status: str, duration: float) -> None:
    """Record a Reddit refresh."""
    reddit_refresh_total.labels(status=status).inc()
    reddit_refresh_duration.observe(duration)


def set_reddit_posts(subreddit: str, count: int) -> None:
    """Set Reddit posts count."""
    reddit_posts_total.labels(subreddit=subreddit).set(count)


def record_error(component: str, error_type: str) -> None:
    """Record an error."""
    errors_total.labels(component=component, error_type=error_type).inc()


def record_gif_sent(intent: str | None, emotion: str | None) -> None:
    """Record a GIF sent."""
    gif_sent_total.labels(
        intent=intent or "none",
        emotion=emotion or "none"
    ).inc()


def set_memory_graph_size(user_id: str, size: int) -> None:
    """Set memory graph size for a user."""
    memory_graph_size.labels(user_id=user_id).set(size)


def set_memory_graph_cap(cap: int) -> None:
    """Set memory graph capacity."""
    memory_graph_cap.set(cap)


def set_emotion_state(user_id: str, emotion: str, value: float) -> None:
    """Set emotion state for a user."""
    emotion_state.labels(user_id=user_id, emotion=emotion).set(value)


def record_rate_limit_wait(backend: str, wait_time: float) -> None:
    """Record rate limit wait time."""
    rate_limit_wait.labels(backend=backend).observe(wait_time)


def set_active_users(count: int) -> None:
    """Set active users count."""
    active_users.set(count)


def set_uptime(seconds: float) -> None:
    """Set uptime."""
    uptime_seconds.set(seconds)


def record_voice_stage(stage: str, seconds: float, ttfa: float | None = None) -> None:
    """Record one voice-pipeline stage timing (and the overall TTFA).

    Called from voice/pipeline.py at the end of every reply; the stage
    label is one of silence/stt/decision/ai/ai_first/tts_first.
    """
    voice_stage_seconds.labels(stage=stage).observe(max(seconds, 0.0))
    if ttfa is not None:
        ttfa_value = max(ttfa, 0.0)
        voice_ttfa_seconds.observe(ttfa_value)
        voice_ttfa_latest.set(ttfa_value)
        _recent_ttfas.append(ttfa_value)


def record_voice_reply(status: str) -> None:
    """Record a voice reply outcome:
    ok / silent / dropped / no_response / soft / backchannel / empty."""
    voice_replies_total.labels(status=status).inc()


def record_voice_predictive(outcome: str) -> None:
    """Record a predictive-STT outcome:
    none / hit / hit_wait / stale / miss / failed."""
    voice_stt_predictive_total.labels(outcome=outcome).inc()


def record_voice_stt_filtered(reason: str) -> None:
    """Record an utterance dropped by the STT hallucination filter:
    single_word / hallucination."""
    voice_stt_filtered_total.labels(reason=reason).inc()


def record_glance(outcome: str, latency: float | None = None) -> None:
    """Record a screen-glance outcome:
    ok / refused / failed (latency = capture-to-comment seconds)."""
    glance_total.labels(outcome=outcome).inc()
    if latency is not None:
        glance_latency.observe(max(0.0, latency))


def voice_stats_snapshot() -> dict:
    """Small in-process snapshot of the latest voice latency numbers,
    for /voice-status (no Prometheus scrape needed)."""
    ttfa_p50 = 0.0
    if _recent_ttfas:
        ordered = sorted(_recent_ttfas)
        ttfa_p50 = ordered[len(ordered) // 2]
    return {
        "ttfa_latest": voice_ttfa_latest._value.get(),
        "ttfa_p50": ttfa_p50,
    }


_RECONNECT_LOG_PREFIX = "Attempting a reconnect"


class _ReconnectCountingFilter(logging.Filter):
    """Counts discord.py's reconnect-attempt log lines as a metric.

    discord.py logs gateway reconnect retries only through its
    "discord.client" logger ("Attempting a reconnect in Xs"); there is
    no event to hook, so the count is extracted here. The filter never
    swallows records -- it just counts them on the way through.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if (
                record.name.startswith("discord.")
                and isinstance(record.msg, str)
                and record.msg.startswith(_RECONNECT_LOG_PREFIX)
            ):
                global _gateway_reconnect_count
                _gateway_reconnect_count += 1
                gateway_reconnects_total.inc()
        except Exception:  # pragma: no cover - never break logging
            pass
        return True


_filter_installed = False


def install_gateway_reconnect_listener() -> None:
    """Attach the reconnect-counting filter to discord.py's loggers.

    Idempotent, called from init_metrics().
    """
    global _filter_installed
    if _filter_installed:
        return
    _filter_installed = True
    logging.getLogger("discord.client").addFilter(_ReconnectCountingFilter())


def record_gateway_reconnect() -> None:
    """Record a reconnect attempt (used when the log filter can't apply)."""
    global _gateway_reconnect_count
    _gateway_reconnect_count += 1
    gateway_reconnects_total.inc()


def gateway_reconnect_count() -> int:
    """Total gateway reconnect attempts since process start."""
    return _gateway_reconnect_count
