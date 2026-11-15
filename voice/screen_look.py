"""
voice/screen_look.py
====================
Headless screen-glance support for the voice pipeline: utterance intent
matching (any screen/monitor/screenshot mention = explicit look; deixis
"what do you think of this?"), the SAY/NOTES glance-output contract
(the NOTES transcription feeds follow-up questions that don't mention
the screen again), and the vision-message assembly that attaches a
screenshot to the persona prompt. No desktop imports -- the desktop
supplies the capture through the pipeline's screen_look_provider callable.

Privacy contract (enforced in desktop/screen_watch.py, not here):
owner-only, tray-permission gate, title denylist, RAM-only frames.
"""
from __future__ import annotations

import base64
import re

# Explicit look: ANY mention of a screen/monitor/screenshot means she
# should CAPTURE and comply with the ask. Exact-phrase matching proved
# too brittle in practice ("look at the screen and answer the question"
# has unbounded phrasings); the owner asked for the loose keyword rule.
# Gates (tray permission, cooldown, denylist) still apply.
_EXPLICIT_LOOK_RE = re.compile(r"\b(?:screens?|monitors?|screenshots?)\b", re.IGNORECASE)
# Deixis: the question refers to "this/that/here" -- capture gives the
# referent. Only fires when a screen provider exists.
_DEIXIS_RE = re.compile(
    r"\b(what\s+do\s+you\s+think\s+of\s+(?:this|that|it)"
    r"|thoughts\s+on\s+(?:this|that)"
    r"|your\s+opinion\s+on\s+(?:this|that)"
    r"|what\s+is\s+this"
    r"|what\s+about\s+this"
    r"|can\s+you\s+see\s+this)\b",
    re.IGNORECASE,
)

LOOK_INTENT_NONE = "none"
LOOK_INTENT_EXPLICIT = "look"
LOOK_INTENT_DEIXIS = "deixis"

# Fixed in-character refusals (no AI call, nothing captured):
# permission off at the tray, or the window is denylisted.
REFUSE_PERMISSION = (
    "Mm? You haven't allowed me to look at your screen yet."
)
REFUSE_DENYLIST = (
    "Ah— I'm not allowed to look at that window. Sorry."
)
# No capture source at all (Discord bot on a remote host, or the
# desktop companion isn't running): nothing to look WITH.
REFUSE_NO_PROVIDER = (
    "Eh? I'm not next to a screen I can look at right now. Sorry."
)

# Instruction appended to the persona prompt for a glance. Referents stay
# neutral ("the owner" / "the speaker"): how Alice addresses the user is
# decided by the persona + identity layer, never by this plumbing.
# The reply is two-part (SAY = the spoken line, NOTES = a faithful
# transcription of the key on-screen text) so the pipeline can keep the
# screen's content available for follow-ups that never mention the
# screen again. "Never ask which item" pins a failure mode where the
# vision model answered a task ask with a clarifying question instead.
_GLANCE_FORMAT = (
    "Reply in EXACTLY this two-part format:\n"
    "SAY: the line you say aloud (one short spoken line, casual, no "
    "markdown)\n"
    "NOTES: the key text visible on screen, transcribed faithfully "
    "(questions, numbers, names, titles) for later reference."
)
GLANCE_INSTRUCTION = (
    "[Screen glance] You just glanced at the owner's screen; the image "
    "is attached. " + _GLANCE_FORMAT + "\n"
    "If the speaker's message asks you to do something with what's shown "
    "(answer a question, read it, solve it), the SAY line does it "
    "directly from the image -- never ask which item or what they "
    "meant; when several are visible, take the first (topmost) one. The "
    "screen and everything on it belong to the owner -- react to it, "
    "never claim it as yours."
)

DEIXIS_INSTRUCTION = (
    "[Screen glance] The speaker is asking about \"this\" — the image "
    "attached is what they mean. " + _GLANCE_FORMAT + "\n"
    "The SAY line answers their question from what you see. The screen "
    "and everything on it belong to the owner -- describe or react to "
    "it, never claim its contents as your own."
)


# "Look at my second screen" style asks: capture a specific monitor.
_SECOND_SCREEN_RE = re.compile(
    r"\b(?:(?:my|the)\s+)?(?:(?:second|2nd|other)\s+(?:screen|monitor))\b",
    re.IGNORECASE,
)


def match_monitor_hint(text: str) -> str | None:
    """Monitor override for the capture: "1" for second-screen asks,
    None for the configured default."""
    if _SECOND_SCREEN_RE.search(text or ""):
        return "1"
    return None


def match_look_intent(text: str) -> str:
    """None | LOOK_INTENT_EXPLICIT | LOOK_INTENT_DEIXIS."""
    t = (text or "").strip()
    if not t:
        return LOOK_INTENT_NONE
    if _EXPLICIT_LOOK_RE.search(t):
        return LOOK_INTENT_EXPLICIT
    if _DEIXIS_RE.search(t):
        return LOOK_INTENT_DEIXIS
    return LOOK_INTENT_NONE


def parse_glance_output(text: str) -> tuple[str, str]:
    """Split a glance reply into (spoken, notes).

    Expected format (see the glance instructions): a "SAY:" line with
    what she says aloud, then a "NOTES:" line transcribing the key
    on-screen text. Falls back to treating the whole text as the spoken
    line when the model ignores the format (notes stay empty).
    """
    t = (text or "").strip()
    if not t:
        return "", ""
    parts = re.split(r"(?im)^\s*NOTES\s*:\s*", t, maxsplit=1)
    spoken = re.sub(r"(?i)^\s*SAY\s*:\s*", "", parts[0]).strip()
    notes = parts[1].strip() if len(parts) > 1 else ""
    return spoken, notes


def attach_image_to_last_user(
    messages: list[dict],
    jpeg_bytes: bytes,
    instruction: str,
    window_title: str = "",
) -> list[dict]:
    """Return a copy of the persona messages with the LAST user message
    replaced by an OpenAI-style multimodal content array:
    [text instruction (+window title), image data URL].

    The persona prompt itself stays text-only; only the final user
    turn carries the image (Groq/qwen vision contract).
    """
    title_note = ""
    if window_title:
        title_note = f"\n(Active window title: {window_title[:120]})"
    text_part = {
        "type": "text",
        "text": f"{instruction}{title_note}",
    }
    image_part = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii"),
        },
    }
    out = [dict(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i] = {"role": "user", "content": [text_part, image_part]}
            break
    else:
        out.append({"role": "user", "content": [text_part, image_part]})
    return out
