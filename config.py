"""
config.py
=========
Single source of truth for every setting in the bot. Everything is read
from environment variables (populated from `.env` via python-dotenv in
main.py) with sane defaults, so nothing important is hardcoded in the
application code.

All paths are anchored to BASE_DIR (this project's own folder) using
pathlib, so the bot works the same whether it's run locally on
D:\\AliceShimadaBot (Windows) or deployed on a Linux host such as
Kerit Cloud. Nothing writes outside this project folder, and nothing
touches the system drive — the one exception is the auto-managed
GPT-SoVITS API server (VOICE_TTS_ENGINE=sovits), which lives in
VOICE_SOVITS_SERVER_DIR and writes its pid/boot log there.
"""

from __future__ import annotations

import os
from pathlib import Path


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_id_list(name: str) -> list[str]:
    """Parses a comma-separated list of Discord snowflake IDs from env."""
    raw = os.getenv(name, "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _prefixed(name: str, prefix: str | None = None) -> str:
    """Returns the environment variable name with optional bot prefix."""
    prefix = prefix or os.getenv("BOT_NAME", "").upper()
    if prefix:
        return f"{prefix}_{name}"
    return name


def _strip_inline_comment(val: str) -> str:
    """Strip trailing '# ...' comments from .env values (dotenv misses some)."""
    if not val:
        return val
    # Leading comment-only value (e.g. empty var + spaces + '# note')
    stripped = val.strip()
    if stripped.startswith("#"):
        return ""
    # Trailing inline comment: split on whitespace+# but not mid-token
    for i, ch in enumerate(val):
        if ch == "#" and i > 0 and val[i - 1] in (" ", "\t"):
            return val[:i].rstrip()
    return val


def _get_prefixed(name: str, default: str = "", prefix: str | None = None) -> str:
    """Gets env var with optional bot prefix, falling back to non-prefixed."""
    prefixed_name = _prefixed(name, prefix)
    raw = os.getenv(prefixed_name, os.getenv(name, default))
    return _strip_inline_comment(raw)


def _get_prefixed_bool(name: str, default: bool, prefix: str | None = None) -> bool:
    val = _get_prefixed(name, "", prefix)
    if not val:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_prefixed_int(name: str, default: int, prefix: str | None = None) -> int:
    val = _get_prefixed(name, "", prefix)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_prefixed_float(name: str, default: float, prefix: str | None = None) -> float:
    val = _get_prefixed(name, "", prefix)
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_prefixed_id_list(name: str, prefix: str | None = None) -> list[str]:
    val = _get_prefixed(name, "", prefix)
    if not val:
        return []
    return [x.strip() for x in val.split(",") if x.strip()]


# =========================================================
# BOT NAME (for multi-bot deployments)
# =========================================================
BOT_NAME = os.getenv("BOT_NAME", "").upper()


# =========================================================
# PATHS  (everything lives under the project folder -> D:)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

# Use bot-specific subdirectories if BOT_NAME is set
if BOT_NAME:
    DATA_DIR = BASE_DIR / "data" / BOT_NAME.lower()
else:
    DATA_DIR = BASE_DIR / "data"

REDDIT_DATA_DIR = DATA_DIR / "reddit"
MEMORY_DATA_DIR = DATA_DIR / "memory"
LOG_DIR = DATA_DIR / "logs"

for _dir in (DATA_DIR, REDDIT_DATA_DIR, MEMORY_DATA_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

MEMORY_DB_PATH = MEMORY_DATA_DIR / "memory.db"
REDDIT_ROTATION_STATE_PATH = REDDIT_DATA_DIR / "rotation_state.json"


# =========================================================
# DISCORD
# =========================================================
DISCORD_TOKEN = _get_prefixed("DISCORD_TOKEN")

# Comma-separated Discord user IDs allowed to use admin commands.
OWNER_IDS = set(_get_prefixed_id_list("OWNER_IDS"))

# Optional: set this to a guild ID during development for instant
# slash-command sync (guild syncs are immediate; global syncs can take
# up to an hour to propagate). Leave unset for production/global sync.
DEV_GUILD_ID = _get_prefixed("DEV_GUILD_ID") or None


# =========================================================
# AI BACKEND
# =========================================================
# "shapes" -> cloud inference via shapes.inc -- NOTE: as of mid-2026
#             Shapes.inc has deprecated third-party build/API access,
#             so this backend will fail until they restore it. Kept
#             here so switching back later is a one-line env change.
# "groq"   -> cloud inference via Groq's OpenAI-compatible API (no
#             local GPU needed, fast, has a free tier). Current default.
# "ollama" -> local inference via a running Ollama server (useful for
#             offline dev/testing, or as a no-API-key fallback)
AI_BACKEND = _get_prefixed("AI_BACKEND", "groq").strip().lower()

# --- Shapes.inc ---
SHAPES_API_KEY = _get_prefixed("SHAPESINC_API_KEY")
SHAPES_SHAPE_USERNAME = _get_prefixed("SHAPESINC_SHAPE_USERNAME")
SHAPES_BASE_URL = _get_prefixed("SHAPES_BASE_URL", "https://api.shapes.inc/v1/")
SHAPES_MODEL = f"shapesinc/{SHAPES_SHAPE_USERNAME}" if SHAPES_SHAPE_USERNAME else ""
# Shapes' default rate limit is 5-20 requests/minute depending on plan.
# The client self-throttles to stay under this so bursts queue instead
# of erroring. Raise this if you've requested a higher limit from Shapes.
SHAPES_RATE_LIMIT_RPM = _get_prefixed_int("SHAPES_RATE_LIMIT_RPM", 20)
SHAPES_TIMEOUT_SECONDS = _get_prefixed_float("SHAPES_TIMEOUT_SECONDS", 60.0)
SHAPES_MAX_RETRIES = _get_prefixed_int("SHAPES_MAX_RETRIES", 2)

# --- Groq ---
GROQ_API_KEY = _get_prefixed("GROQ_API_KEY")
GROQ_BASE_URL = _get_prefixed("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
# Any current Groq-hosted chat model works here -- see
# https://console.groq.com/docs/models for the up-to-date list.
# NOTE: llama-3.3-70b-versatile / llama-3.1-8b-instant are now
# Enterprise-only on Groq; openai/gpt-oss-120b is the free-tier default.
GROQ_MODEL = _get_prefixed("GROQ_MODEL", "openai/gpt-oss-120b")
# Groq's free tier is generous but still rate-limited per-minute. The
# client self-throttles to stay under this so bursts queue instead of
# erroring. Raise this if you're on a paid plan with a higher limit.
GROQ_RATE_LIMIT_RPM = _get_prefixed_int("GROQ_RATE_LIMIT_RPM", 30)
GROQ_TIMEOUT_SECONDS = _get_prefixed_float("GROQ_TIMEOUT_SECONDS", 60.0)
GROQ_MAX_RETRIES = _get_prefixed_int("GROQ_MAX_RETRIES", 2)
# Caps how long a single completion can run. Without this, a runaway
# or repetitive completion can exceed Discord's message length limit
# (2000 chars normal, 4000 on boosted servers) and crash message.reply()
# with an uncaught HTTPException -- this keeps replies from ever
# reaching that size in the first place. ~300 tokens is generous for
# Alice's short, quiet style; raise if a character needs longer
# replies.
GROQ_MAX_TOKENS = _get_prefixed_int("GROQ_MAX_TOKENS", 600)
# Rolling tokens-per-minute budget. Groq enforces TPM per organisation
# (free tier for openai/gpt-oss-120b is 8,000 TPM); the client keeps a
# 60s window of actual usage and will not fire a request that would
# cross this limit -- it waits briefly, then fails fast. Raise to match
# a paid plan (Developer tier is 250K+ TPM).
GROQ_TPM_LIMIT = _get_prefixed_int("GROQ_TPM_LIMIT", 8000)
# Longest a single rate-limit wait may be before the client gives up
# and returns the fallback instead of blocking. A voice reply that
# arrives 20s after the utterance is worthless, so this stays small.
GROQ_RETRY_MAX_WAIT = _get_prefixed_float("GROQ_RETRY_MAX_WAIT", 3.0)
# Reasoning effort for Groq's openai/gpt-oss models: low/medium/high
# (empty = don't send the parameter; ignored for non-gpt-oss models --
# Groq only accepts this on gpt-oss 20B/120B). gpt-oss burns hidden
# reasoning tokens BEFORE any visible content, and reasoning counts
# against both max_tokens and TPM. At the default "medium" effort a
# 250-token voice reply could be fully spent on reasoning, returning
# empty content (silent voice, fallback text). "low" keeps replies
# fast, cheap, and actually visible.
GROQ_REASONING_EFFORT = _get_prefixed("GROQ_REASONING_EFFORT", "low")

# --- Ollama (optional local fallback) ---
OLLAMA_HOST = _get_prefixed("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_MODEL = _get_prefixed("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT_SECONDS = _get_prefixed_float("OLLAMA_TIMEOUT_SECONDS", 90.0)


# =========================================================
# CHARACTER
# =========================================================
CHARACTER_NAME = _get_prefixed("CHARACTER_NAME", "alice").strip().lower()

# When True, the character's full lore/identity/style-example block is
# injected into every request. Groq and Ollama both have no personality
# of their own (unlike Shapes' dashboard-configured Shape), so this
# defaults on for both. When False, only a short per-user
# relationship/identity context note is sent, since the personality
# itself is assumed to be configured elsewhere (e.g. Shapes' dashboard).
INJECT_FULL_CHARACTER_PROMPT = _get_prefixed_bool(
    "INJECT_FULL_CHARACTER_PROMPT", AI_BACKEND in ("ollama", "groq")
)


# =========================================================
# CONVERSATION BEHAVIOR
# =========================================================
CHAT_HISTORY_LENGTH = _get_prefixed_int("CHAT_HISTORY_LENGTH", 10)
MEMORY_GRAPH_LIMIT = _get_prefixed_int("MEMORY_GRAPH_LIMIT", 10)
MEMORY_GRAPH_CAP_PER_USER = _get_prefixed_int("MEMORY_GRAPH_CAP_PER_USER", 80)
GIF_COOLDOWN_SECONDS = _get_prefixed_float("GIF_COOLDOWN_SECONDS", 25.0)

# Emotion decay constants (for EmotionEngine)
# These control how quickly emotional states decay over time
EMOTION_DECAY_SHY = _get_prefixed_float("EMOTION_DECAY_SHY", 0.97)
EMOTION_DECAY_CURIOSITY = _get_prefixed_float("EMOTION_DECAY_CURIOSITY", 0.985)
EMOTION_DECAY_HAPPINESS = _get_prefixed_float("EMOTION_DECAY_HAPPINESS", 0.985)
EMOTION_DECAY_SADNESS = _get_prefixed_float("EMOTION_DECAY_SADNESS", 0.985)
EMOTION_DECAY_SURPRISE = _get_prefixed_float("EMOTION_DECAY_SURPRISE", 0.985)
EMOTION_CALM_INCREASE = _get_prefixed_float("EMOTION_CALM_INCREASE", 0.003)

# Emotion persistence boost thresholds
EMOTION_PERSISTENCE_THRESHOLD = _get_prefixed_float("EMOTION_PERSISTENCE_THRESHOLD", 0.45)
EMOTION_PERSISTENCE_BOOST = _get_prefixed_float("EMOTION_PERSISTENCE_BOOST", 0.015)

# Safety cap on outgoing Discord message length. Discord's hard limit
# is 2000 chars normally (4000 on boosted servers) -- this stays under
# both regardless of server boost level, as a last-resort truncation
# net in case GROQ_MAX_TOKENS ever isn't enough to prevent an oversized
# reply (e.g. a very long GIF-selection tail, or a future backend with
# no token cap of its own).
DISCORD_MAX_REPLY_CHARS = _get_prefixed_int("DISCORD_MAX_REPLY_CHARS", 1900)

# Fallback reply used when AI backend returns empty/error response.
# Can be overridden per-character via character data if needed.
FALLBACK_REPLY = _get_prefixed("FALLBACK_REPLY", "I don't know what to say.")


# =========================================================
# REDDIT
# =========================================================
REDDIT_USER_AGENT = _get_prefixed(
    "REDDIT_USER_AGENT",
    "AliceShimadaBot/2.0 (Discord character bot; contact via r/AliceShimada)",
)

# How many raw dump files to keep per subreddit before pruning the
# oldest ones. Keeps disk usage bounded on small hosts (e.g. Kerit
# Cloud's free 2GB tier).
REDDIT_RAW_RETENTION_FILES = _get_prefixed_int("REDDIT_RAW_RETENTION_FILES", 20)

# How many listing pages (~100 posts each) to pull per subreddit per
# fetch cycle, and how often to run a fetch cycle automatically.
REDDIT_FETCH_PAGES = _get_prefixed_int("REDDIT_FETCH_PAGES", 4)
REDDIT_FETCH_INTERVAL_HOURS = _get_prefixed_float("REDDIT_FETCH_INTERVAL_HOURS", 12.0)
REDDIT_FETCH_SORT = _get_prefixed("REDDIT_FETCH_SORT", "top")
REDDIT_FETCH_TIME_RANGE = _get_prefixed("REDDIT_FETCH_TIME_RANGE", "month")
# Enable parallel fetching of subreddits (faster but more concurrent connections)
REDDIT_FETCH_PARALLEL = _get_prefixed_bool("REDDIT_FETCH_PARALLEL", True)
# Where fresh posts come from:
#   "pullpush" - api.pullpush.io community archive. No auth needed.
#                (Reddit ended self-serve API app creation in Nov 2025
#                and 403s unauthenticated www.reddit.com JSON, so new
#                installs cannot get OAuth credentials.)
#   "reddit"   - legacy www.reddit.com JSON listing API. Only works
#                where Reddit still serves it; kept for restored access.
REDDIT_SOURCE = _get_prefixed("REDDIT_SOURCE", "pullpush").strip().lower()

# Subreddit registry: adding a new entry here (plus a matching slash
# command in cogs/reddit_cog.py) is all that's needed to support
# another subreddit.
REDDIT_SUBREDDITS: dict[str, dict] = {
    "AliceShimada": {
        "raw_dir": REDDIT_DATA_DIR / "alice_raw",
        "posts_file": REDDIT_DATA_DIR / "alice_posts.json",
        "display_name": "r/AliceShimada",
        "embed_color": 0x87CEEB,
    },
    "GIRLSundPANZER": {
        "raw_dir": REDDIT_DATA_DIR / "gupp_raw",
        "posts_file": REDDIT_DATA_DIR / "gupp_posts.json",
        "display_name": "r/GIRLSundPANZER",
        "embed_color": 0xFFB347,
    },
}

for _sub in REDDIT_SUBREDDITS.values():
    _sub["raw_dir"].mkdir(parents=True, exist_ok=True)


# =========================================================
# LOGGING
# =========================================================
LOG_LEVEL = _get_prefixed("LOG_LEVEL", "INFO").upper()
LOG_MAX_BYTES = _get_prefixed_int("LOG_MAX_BYTES", 2_000_000)  # 2 MB per file
LOG_BACKUP_COUNT = _get_prefixed_int("LOG_BACKUP_COUNT", 3)
LOG_JSON = _get_prefixed_bool("LOG_JSON", False)


# =========================================================
# VOICE (v2.1)
# =========================================================
VOICE_ENABLED = _get_prefixed_bool("VOICE_ENABLED", False)

# STT
VOICE_STT_ENGINE = _get_prefixed("VOICE_STT_ENGINE", "moonshine")  # moonshine | whisper
VOICE_STT_MODEL = _get_prefixed("VOICE_STT_MODEL", "small")  # whisper only: tiny/base/small/medium/large-v3
VOICE_STT_DEVICE = _get_prefixed("VOICE_STT_DEVICE", "cuda")  # cuda/cpu
VOICE_STT_COMPUTE_TYPE = _get_prefixed("VOICE_STT_COMPUTE_TYPE", "float16")  # float16/int8
# Fixed language = no detection at all (saves time/resources and is the
# only mode the moonshine engine supports); "auto"/empty = detection via
# faster-whisper.
VOICE_STT_LANGUAGE = _get_prefixed("VOICE_STT_LANGUAGE", "en")  # en/ja/auto
VOICE_STT_MOONSHINE_EN = _get_prefixed("VOICE_STT_MOONSHINE_EN", "UsefulSensors/moonshine-base")
VOICE_STT_MOONSHINE_JA = _get_prefixed("VOICE_STT_MOONSHINE_JA", "UsefulSensors/moonshine-tiny-ja")
# STT eval capture: save raw voice clips + sidecars to STT_EVAL_DIR so
# manage.py eval-stt can score candidate engines/preprocessing on the
# user's own accented speech. Off by default; self-limits at the cap.
VOICE_STT_CAPTURE = _get_prefixed_bool("VOICE_STT_CAPTURE", False)
VOICE_STT_CAPTURE_LIMIT = _get_prefixed_int("VOICE_STT_CAPTURE_LIMIT", 200)
STT_EVAL_DIR = _get_prefixed("STT_EVAL_DIR", str(BASE_DIR / "stt_eval" / "clips"))

# TTS (edge / sovits)
VOICE_TTS_ENGINE = _get_prefixed("VOICE_TTS_ENGINE", "edge")  # edge/sovits/rvc/piper
VOICE_TTS_VOICE = _get_prefixed("VOICE_TTS_VOICE", "ja-JP-NanamiNeural")
VOICE_TTS_RATE = _get_prefixed("VOICE_TTS_RATE", "-10%")  # SSML rate for character feel
VOICE_TTS_PITCH = _get_prefixed("VOICE_TTS_PITCH", "+5Hz")

# GPT-SoVITS (VOICE_TTS_ENGINE=sovits)
VOICE_SOVITS_URL = _get_prefixed("VOICE_SOVITS_URL", "http://127.0.0.1:9880/tts")
VOICE_SOVITS_REF_AUDIO = _get_prefixed("VOICE_SOVITS_REF_AUDIO", "data/voice_models/alice_ref.wav")
VOICE_SOVITS_TEXT_LANG = _get_prefixed("VOICE_SOVITS_TEXT_LANG", "auto")  # auto/ja/en
VOICE_SOVITS_TEMPERATURE = _get_prefixed_float("VOICE_SOVITS_TEMPERATURE", 0.5)
VOICE_SOVITS_TOP_P = _get_prefixed_float("VOICE_SOVITS_TOP_P", 0.8)
VOICE_SOVITS_TOP_K = _get_prefixed_int("VOICE_SOVITS_TOP_K", 15)
VOICE_SOVITS_SPEED = _get_prefixed_float("VOICE_SOVITS_SPEED", 1.0)
VOICE_SOVITS_SPLIT = _get_prefixed("VOICE_SOVITS_SPLIT", "cut5")
VOICE_SOVITS_TIMEOUT = _get_prefixed_float("VOICE_SOVITS_TIMEOUT", 30.0)
# Warm-up/probe syntheses tolerate cold weight loading on a fresh server,
# which can exceed VOICE_SOVITS_TIMEOUT; real requests keep the short one.
VOICE_SOVITS_WARMUP_TIMEOUT = _get_prefixed_float("VOICE_SOVITS_WARMUP_TIMEOUT", 120.0)
# Auto-managed API server (spawned only when nothing is listening)
VOICE_SOVITS_SERVER_PY = _get_prefixed("VOICE_SOVITS_SERVER_PY", "D:\\GPT-SoVITS\\venv\\Scripts\\python.exe")
VOICE_SOVITS_SERVER_DIR = _get_prefixed("VOICE_SOVITS_SERVER_DIR", "D:\\GPT-SoVITS")
VOICE_SOVITS_SERVER_ENTRY = _get_prefixed("VOICE_SOVITS_SERVER_ENTRY", "api_v2.py")
VOICE_SOVITS_SERVER_CONFIG = _get_prefixed("VOICE_SOVITS_SERVER_CONFIG", "GPT_SoVITS\\configs\\tts_infer.yaml")
VOICE_SOVITS_FFMPEG_BINS = _get_prefixed(
    "VOICE_SOVITS_FFMPEG_BINS",
    "D:\\ffmpeg-shared\\ffmpeg-9.0.2-full_build-shared\\bin;C:\\ffmpeg\\ffmpeg-9.0.2-essentials_build\\bin",
)
VOICE_SOVITS_STARTUP_TIMEOUT = _get_prefixed_float("VOICE_SOVITS_STARTUP_TIMEOUT", 150.0)

# RVC (Voice Conversion) - Optional
VOICE_RVC_ENABLED = _get_prefixed_bool("VOICE_RVC_ENABLED", False)
VOICE_RVC_MODEL_PATH = _get_prefixed("VOICE_RVC_MODEL_PATH", "")  # Path to .pth model
VOICE_RVC_INDEX_PATH = _get_prefixed("VOICE_RVC_INDEX_PATH", "")  # Path to .index file
VOICE_RVC_PITCH = _get_prefixed_int("VOICE_RVC_PITCH", 0)  # Semitone shift
VOICE_RVC_DEVICE = _get_prefixed("VOICE_RVC_DEVICE", "cuda")
VOICE_RVC_INDEX_RATE = _get_prefixed_float("VOICE_RVC_INDEX_RATE", 0.75)  # 0..1 FAISS retrieval mix

# Diarization (pyannote)
VOICE_DIARIZATION = _get_prefixed_bool("VOICE_DIARIZATION", True)
VOICE_HF_TOKEN = _get_prefixed("VOICE_HF_TOKEN")  # Required for pyannote models

# Audio
VOICE_SAMPLE_RATE = 16000  # Internal (Whisper)
VOICE_DISCORD_SAMPLE_RATE = 48000  # Discord Opus
VOICE_VAD_THRESHOLD = _get_prefixed_float("VOICE_VAD_THRESHOLD", 0.5)
VOICE_MIN_SPEECH_DURATION = _get_prefixed_float("VOICE_MIN_SPEECH_DURATION", 0.5)
VOICE_MAX_SPEECH_DURATION = _get_prefixed_float("VOICE_MAX_SPEECH_DURATION", 30.0)

# Behavior
VOICE_CHANNEL_ID = _get_prefixed("VOICE_CHANNEL_ID")  # Auto-join channel
VOICE_AUTO_JOIN = _get_prefixed_bool("VOICE_AUTO_JOIN", False)
VOICE_REPLY_COOLDOWN = _get_prefixed_float("VOICE_REPLY_COOLDOWN", 2.0)
VOICE_REPLY_PROBABILITY = _get_prefixed_float("VOICE_REPLY_PROBABILITY", 1.0)
VOICE_IGNORE_BOTS = _get_prefixed_bool("VOICE_IGNORE_BOTS", True)
VOICE_IGNORE_SELF = _get_prefixed_bool("VOICE_IGNORE_SELF", True)

# Conversation-aware interaction
VOICE_RESPONSE_MODE = _get_prefixed("VOICE_RESPONSE_MODE", "smart")  # always/smart/mention_only/observer
VOICE_SILENCE_THRESHOLD = _get_prefixed_float("VOICE_SILENCE_THRESHOLD", 0.8)  # Seconds of silence before speech complete
VOICE_TURN_PAUSE_THRESHOLD = _get_prefixed_float("VOICE_TURN_PAUSE_THRESHOLD", 2.0)  # Seconds before Alice can interject
VOICE_LULL_THRESHOLD = _get_prefixed_float("VOICE_LULL_THRESHOLD", 4.0)  # Seconds of silence = conversation lull
VOICE_LULL_JOIN_PROBABILITY = _get_prefixed_float("VOICE_LULL_JOIN_PROBABILITY", 0.3)  # Chance to join in lull
VOICE_MENTION_KEYWORDS = _get_prefixed("VOICE_MENTION_KEYWORDS", "alice,アリス").split(",")  # Keywords that trigger response
VOICE_CONTEXT_WINDOW = _get_prefixed_int("VOICE_CONTEXT_WINDOW", 10)  # Last N utterances to track
VOICE_OWNER_PRIORITY = _get_prefixed_bool("VOICE_OWNER_PRIORITY", True)  # Always respond to owner
VOICE_BACKCHANNEL_ENABLED = _get_prefixed_bool("VOICE_BACKCHANNEL_ENABLED", True)  # "mm", "uh-huh" acknowledgments
VOICE_BARGE_IN_ENABLED = _get_prefixed_bool("VOICE_BARGE_IN_ENABLED", True)  # Allow user to interrupt Alice
VOICE_RESPONSE_LANGUAGE = _get_prefixed("VOICE_RESPONSE_LANGUAGE", "auto")  # auto/en/ja — forces Alice to respond in this language

# STT hallucination filter: optional pipe-delimited list of sentences to
# reject wholesale (e.g. the old Whisper prompt phrases). Leave unset to use
# the built-in defaults ("Hello Alice.", "Hi Alice.", "Alice, how are you doing?", "What are you doing, Alice?").
VOICE_STT_HALLUCINATION_PHRASES = _get_prefixed("VOICE_STT_HALLUCINATION_PHRASES", None)

# --- Name mention response (when Alice is mentioned but not directly addressed) ---
# soft = respond with a short clarification phrase ("Hmm? ... did I hear my name?")
# skip = stay silent (original behavior)
VOICE_NAME_MENTION_RESPONSE = _get_prefixed("VOICE_NAME_MENTION_RESPONSE", "soft").strip().lower()
if VOICE_NAME_MENTION_RESPONSE not in ("soft", "skip"):
    VOICE_NAME_MENTION_RESPONSE = "soft"

# --- Diarization mode ---
# Speaker analysis of audio clips (runs non-blocking in a worker thread;
# useful in party channels with 2+ speakers).
# none   = disabled (default, zero overhead)
# detect = analysis + silence when a mixed clip only mentions her
# address= detect (kept for compatibility — same behavior)
VOICE_DIARIZE_MODE = _get_prefixed("VOICE_DIARIZE_MODE", "none").strip().lower()
if VOICE_DIARIZE_MODE not in ("none", "detect", "address"):
    VOICE_DIARIZE_MODE = "none"

# Completion cap for voice replies only. Spoken replies are 1-3
# sentences, so ~400 tokens is generous -- but on gpt-oss this is a
# TOTAL budget: hidden reasoning tokens are spent first, so too small
# a cap (250) let reasoning eat everything and leave empty content.
# Completion tokens still count against Groq's TPM budget, so this
# doubles as rate-limit protection.
VOICE_MAX_TOKENS = _get_prefixed_int("VOICE_MAX_TOKENS", 400)


# =========================================================
# REDIS (for horizontal scaling)
# =========================================================
REDIS_HOST = _get_prefixed("REDIS_HOST", "localhost")
REDIS_PORT = _get_prefixed_int("REDIS_PORT", 6379)
REDIS_DB = _get_prefixed_int("REDIS_DB", 0)
REDIS_PASSWORD = _get_prefixed("REDIS_PASSWORD") or None


# =========================================================
# HEALTH CHECK SERVER
# =========================================================
HEALTH_HOST = _get_prefixed("HEALTH_HOST", "0.0.0.0")
HEALTH_PORT = _get_prefixed_int("HEALTH_PORT", 8080)


# =========================================================
# STARTUP VALIDATION
# =========================================================
class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def validate() -> None:
    """
    Fails fast with a clear message instead of failing deep inside a
    Discord event handler or an AI request. Call this once at startup.
    """
    problems = []

    if not DISCORD_TOKEN:
        problems.append("DISCORD_TOKEN is not set.")

    if AI_BACKEND not in ("shapes", "groq", "ollama"):
        problems.append(
            f"AI_BACKEND={AI_BACKEND!r} is invalid. Use 'shapes', 'groq', or 'ollama'."
        )

    if AI_BACKEND == "shapes":
        if not SHAPES_API_KEY:
            problems.append("SHAPESINC_API_KEY is not set (required for AI_BACKEND=shapes).")
        if not SHAPES_SHAPE_USERNAME:
            problems.append("SHAPESINC_SHAPE_USERNAME is not set (required for AI_BACKEND=shapes).")

    if AI_BACKEND == "groq":
        if not GROQ_API_KEY:
            problems.append("GROQ_API_KEY is not set (required for AI_BACKEND=groq).")

    if not OWNER_IDS:
        problems.append(
            "OWNER_IDS is empty -> admin commands will be unusable. "
            "Set OWNER_IDS to your Discord user ID(s), comma-separated."
        )

    # Voice validation (only if enabled)
    if VOICE_ENABLED:
        if VOICE_RVC_ENABLED:
            if not VOICE_RVC_MODEL_PATH:
                problems.append("VOICE_RVC_MODEL_PATH is required when VOICE_RVC_ENABLED=true.")
            elif not VOICE_RVC_MODEL_PATH.lower().endswith(".pth"):
                problems.append("VOICE_RVC_MODEL_PATH must be a .pth file.")
            if not 0.0 <= VOICE_RVC_INDEX_RATE <= 1.0:
                problems.append("VOICE_RVC_INDEX_RATE must be between 0.0 and 1.0.")
        if VOICE_DIARIZATION and not VOICE_HF_TOKEN:
            problems.append("VOICE_HF_TOKEN is required when VOICE_DIARIZATION=true (pyannote models).")

    if problems:
        details = "\n  - ".join(problems)
        raise ConfigError(f"Configuration problems found:\n  - {details}")
