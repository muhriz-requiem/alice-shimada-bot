"""
metrics/collector.py
=====================
Prometheus metrics for the bot.
"""

from __future__ import annotations

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

# Initialize bot info
def init_metrics(bot_name: str = "alice", version: str = "2.0.0") -> None:
    """Initialize bot info metric."""
    bot_info.info({"name": bot_name, "version": version})


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
