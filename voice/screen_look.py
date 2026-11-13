"""
voice/screen_look.py
====================
Headless screen-glance support for the voice pipeline: utterance intent
matching (explicit "look at my screen" + deixis "what do you think of
this?") and the vision-message assembly that attaches a screenshot to
the persona prompt. No desktop imports -- the desktop supplies the
capture through the pipeline's screen_look_provider callable.

Privacy contract (enforced in desktop/screen_watch.py, not here):
owner-only, tray-permission gate, title denylist, RAM-only frames.
"""
from __future__ import annotations

import base64
import re

# Explicit look: she should CAPTURE and comment.
_EXPLICIT_LOOK_RE = re.compile(
    r"\b(look\s+(?:at|at\s+my|at\s+the)\s+(?:my\s+|the\s+)?screen"
    r"|what.?s?\s+(?:on\s+)?my\s+screen"
    r"|what\s+am\s+i\s+looking\s+at"
    r"|can\s+you\s+see\s+my\s+screen"
    r"|check\s+my\s+screen)\b",
    re.IGNORECASE,
)
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
    "Mm? Father, you haven't allowed me to look at your screen yet."
)
REFUSE_DENYLIST = (
    "Ah— I'm not allowed to look at that window. Sorry."
)
# No capture source at all (Discord bot on a remote host, or the
# desktop companion isn't running): nothing to look WITH.
REFUSE_NO_PROVIDER = (
    "Eh? I'm not next to a screen I can look at right now. Sorry."
)

# Instruction appended to the persona prompt for a glance. One short
# in-character line, as decided with the owner.
GLANCE_INSTRUCTION = (
    "[Screen glance] You just glanced at Father's screen. The image is "
    "attached. React in ONE short spoken line (max ~15 words), in "
    "character, casual and a little curious. Do not describe the screen "
    "in detail, do not list items, no markdown. If the screen shows "
    "something you recognize, react to THAT. The screen and everything "
    "on it belong to Father -- react to it, never claim it as yours."
)

DEIXIS_INSTRUCTION = (
    "[Screen glance] Father is asking about \"this\" — the image attached "
    "is what he means. Answer his question in ONE short spoken line "
    "(max ~20 words), in character, based on what you see. The screen "
    "belongs to Father -- describe/react to it, never claim its contents "
    "as your own."
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
