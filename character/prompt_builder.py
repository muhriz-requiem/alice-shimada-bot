"""
character/prompt_builder.py
=============================
Builds the message list sent to the AI backend. There are two modes,
chosen by config.INJECT_FULL_CHARACTER_PROMPT:

- Minimal mode (default when AI_BACKEND=shapes): the character's
  personality is configured on the Shapes.inc dashboard, so we only
  send a short per-user relationship/identity context note plus recent
  conversation history. Sending the full lore/style block on top of an
  already-configured Shapes personality would just be redundant tokens
  fighting the dashboard config for influence.

- Full mode (default when AI_BACKEND=ollama or AI_BACKEND=groq):
  neither Ollama nor Groq has any built-in personality of its own, so
  the full system prompt is built here -- faithfully ported from the
  original character.py, including the "anti-assistant lock" (explicit
  rules against slipping into assistant-speak), the identity firewall
  block, strict-identity mode triggered by certain phrases, and
  sanitized/classified memory snippets. Generalized so none of it is
  hardcoded to one character.

Uses Jinja2 templates for maintainability.
"""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

import config
from character.base import Character
from character.memory_shaping import format_memory, sanitize_memory

_EXAMPLE_TAG_PATTERN = re.compile(r"^User\s*\(([^)]+)\):", re.IGNORECASE)

# Jinja2 environment
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _identity_context_note(identity: dict | None, character: Character) -> str | None:
    """
    Short note used in BOTH modes to make sure per-user relationships
    (e.g. a specific Discord user being treated as a father figure)
    come through regardless of which AI backend is generating the
    reply -- this is data our own bot owns, not something Shapes',
    Groq's, or Ollama's model config can know on its own.
    """
    if not identity:
        return None

    name = identity.get("name")
    role = identity.get("role")
    relationship = identity.get("relationship")
    description = identity.get("description", "")

    if not name:
        return None

    tone = character.relationship_tone(relationship) if relationship else {}
    tone_desc = tone.get("description", "")

    parts = [f"The person you're speaking with is {name}"]
    if role:
        parts.append(f", whom you know as your {role}")
    parts.append(".")
    if description:
        parts.append(f" {description}")
    if tone_desc:
        parts.append(f" {tone_desc}")

    return "".join(parts).strip()


def _build_identity_block(identity: dict | None) -> str:
    # NOTE: this default is intentionally NOT worded as "NAME: Unknown /
    # ROLE: User" -- that bureaucratic, customer-service-flavored
    # phrasing was found to nudge the model toward assistant-like tone
    # any time it's talking to someone unrecognized (which, for a
    # character with only one named relationship configured, is nearly
    # everyone). This wording says the same thing narratively instead.
    identity = identity or {}
    name = identity.get("name") or "someone she doesn't know"
    role = identity.get("role") or "an unfamiliar person"
    description = identity.get("description") or "No further information."

    return (
        "==============================\n"
        "WHO SHE'S TALKING TO (ABSOLUTE TRUTH - DO NOT OVERRIDE)\n"
        "==============================\n"
        f"NAME: {name}\n"
        f"RELATION TO HER: {role}\n"
        f"DESCRIPTION: {description}\n\n"
        "RULES (HARD ENFORCEMENT):\n"
        "- This is FIXED -- never change or reinterpret it\n"
        "- Never roleplay a different identity for this person\n"
        "- Never slip into generic customer-service or assistant framing\n"
        "  just because this person is unrecognized or unfamiliar to her\n"
        "- Not recognizing someone is fine and in-character -- it does NOT\n"
        "  mean she should sound formal, helpful, or assistant-like about it\n"
        "=============================="
    )


def _build_core_facts_block(core_facts: dict[str, str], character_name: str) -> str:
    """
    Hard-enforced canon facts (e.g. "Boko is Alice's teddy bear"), kept
    separate from the free-form LORE prose below. A long system prompt
    can bury a single fact in a paragraph the model glosses over --
    this block exists so key facts get the same ABSOLUTE-TRUTH
    treatment as user identity, instead of relying on the model to
    correctly parse and retain lore.txt every single time.
    """
    if not core_facts:
        return ""

    lines = "\n".join(f"- {name}: {fact}" for name, fact in core_facts.items())
    return (
        "==============================\n"
        "CORE FACTS (ABSOLUTE TRUTH - NEVER DENY OR CONTRADICT)\n"
        "==============================\n"
        f"{lines}\n\n"
        f"RULES (HARD ENFORCEMENT):\n"
        f"- {character_name} ALWAYS recognizes these names/entities exactly as\n"
        f"  defined above, and ONLY as defined above\n"
        f"- NEVER say \"I don't know who that is\" or similar for anything listed above\n"
        f"- NEVER ask what something on this list is, and NEVER ask for\n"
        f"  clarification about who/what it is -- she already knows all of it\n"
        f"  without being reminded, even if she reacts shyly to being asked\n"
        f"- If a name above coincidentally resembles a real-world person, place,\n"
        f"  or organization unrelated to {character_name}'s story, that resemblance\n"
        f"  is irrelevant -- always answer using ONLY the fictional meaning given\n"
        f"  above, never the real-world one\n"
        f"- She may respond shyly or briefly, but never with denial or confusion about WHO they are\n"
        "=============================="
    )


def _is_identity_query(user_message: str, trigger_phrases: list[str]) -> bool:
    if not trigger_phrases:
        return False
    msg = user_message.lower()
    return any(t.lower() in msg for t in trigger_phrases)


def _select_examples(
    examples: list[str], identity: dict | None
) -> tuple[list[str], str | None, list[str]]:
    """
    Splits a character's style examples into a baseline voice (plain
    "User: ..." lines, always relevant) and relationship-tagged lines
    ("User (Father): ..."), which only apply when the person actually
    speaking right now holds that role.

    Without this split, a random stranger's prompt would include
    dialogue written for Alice's father -- e.g. her calling someone
    "Father" and saying "I missed you too" -- which would leak that
    relationship's warmth and address pattern into conversations where
    it doesn't apply. This keeps tagged examples scoped to the person
    they were written for.

    Convention is generic (not hardcoded to "Father"): any tag in
    parentheses is matched case-insensitively against the current
    speaker's `role` from their identity entry, so a future character
    can use the same "User (SomeRole): ..." pattern for its own
    relationships.
    """
    baseline: list[str] = []
    tagged: dict[str, list[str]] = {}

    for ex in examples:
        match = _EXAMPLE_TAG_PATTERN.match(ex.strip())
        if match:
            tag = match.group(1).strip()
            tagged.setdefault(tag.lower(), []).append(ex)
        else:
            baseline.append(ex)

    current_role = (identity or {}).get("role", "") or ""
    match_key = current_role.strip().lower()

    if match_key and match_key in tagged:
        return baseline, current_role, tagged[match_key]

    return baseline, None, []


def build_context_messages(
    character: Character,
    *,
    identity: dict | None,
    history: list[dict],
    user_message: str,
    extra_context: str | None = None,
) -> list[dict]:
    """Minimal mode: identity context (if any) + recent history + message."""
    messages: list[dict] = []

    note = _identity_context_note(identity, character)
    if note:
        messages.append({"role": "system", "content": note})

    if extra_context:
        messages.append({"role": "system", "content": extra_context})

    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


def build_full_prompt_messages(
    character: Character,
    *,
    identity: dict | None,
    relationship_key: str | None,
    memory_snippets: list[str],
    history: list[dict],
    user_message: str,
    fast_path: bool,
    extra_context: str | None = None,
) -> list[dict]:
    """Full mode: complete self-contained character system prompt using Jinja2 template."""
    rules = character.prompt_rules or {}
    trigger_phrases = rules.get("identity_trigger_phrases", [])
    blocked_phrases = rules.get("memory_blocked_phrases", [])
    never_say = rules.get("never_say_phrases", [])

    tone = character.relationship_tone(relationship_key) if relationship_key else {
        "tone": "neutral", "trust": 0.5, "warmth": 0.5,
    }

    clean_memory = sanitize_memory(memory_snippets, blocked_phrases)
    memory_text = format_memory(clean_memory, character.identity_keywords)

    strict_mode = ""
    if _is_identity_query(user_message, trigger_phrases):
        strict_mode = (
            "STRICT MODE ACTIVE:\n"
            "- Answer ONLY using the identity block\n"
            "- Do NOT invent identity\n"
            "- Do NOT roleplay alternative identity\n"
        )

    never_say_block = ""
    if never_say:
        listed = "\n".join(f'- "{p}"' for p in never_say)
        never_say_block = f"NEVER SAY:\n{listed}\n"

    examples = character.fast_examples if fast_path else character.full_examples
    baseline_examples, matched_tag, matched_examples = _select_examples(examples, identity)
    examples_text = "\n".join(baseline_examples) if baseline_examples else "None"

    relationship_examples_block = ""
    if matched_examples and matched_tag:
        listed = "\n".join(matched_examples)
        relationship_examples_block = (
            f"\n\nADDITIONAL EXAMPLES SPECIFIC TO {matched_tag.upper()} -- "
            f"use these ONLY because you are actually speaking with {matched_tag} "
            f"right now. Anyone else still gets only the baseline voice above:\n{listed}"
        )

    # Render template
    template = _JINJA_ENV.get_template("system_prompt.j2")
    system_prompt = template.render(
        character=character,
        never_say_block=never_say_block,
        identity_block=_build_identity_block(identity),
        core_facts_block=_build_core_facts_block(character.core_facts, character.display_name),
        strict_mode=strict_mode,
        tone=tone,
        memory_text=memory_text,
        lore=character.lore.strip() if character.lore else "None",
        examples_text=examples_text,
        relationship_examples_block=relationship_examples_block,
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if extra_context:
        messages.append({"role": "system", "content": extra_context})
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


def build_prompt(
    character: Character,
    *,
    identity: dict | None,
    relationship_key: str | None,
    memory_snippets: list[str],
    history: list[dict],
    user_message: str,
    fast_path: bool,
    extra_context: str | None = None,
) -> list[dict]:
    """Entry point cogs actually call -- picks the mode from config."""
    if config.INJECT_FULL_CHARACTER_PROMPT:
        return build_full_prompt_messages(
            character,
            identity=identity,
            relationship_key=relationship_key,
            memory_snippets=memory_snippets,
            history=history,
            user_message=user_message,
            fast_path=fast_path,
            extra_context=extra_context,
        )

    return build_context_messages(
        character,
        identity=identity,
        history=history,
        user_message=user_message,
        extra_context=extra_context,
    )


def build_gift_prompt(
    character: Character,
    *,
    identity: dict | None,
    relationship_key: str | None,
    memory_snippets: list[str],
    history: list[dict],
    category_key: str,
    item_description: str,
    note: str | None = None,
    fast_path: bool = False,
) -> list[dict]:
    """
    Build a minimal gift-specific prompt for token efficiency.
    Only includes essential character context + gift instructions.
    """
    rules = character.prompt_rules or {}
    trigger_phrases = rules.get("identity_trigger_phrases", [])
    blocked_phrases = rules.get("memory_blocked_phrases", [])
    never_say = rules.get("never_say_phrases", [])

    tone = character.relationship_tone(relationship_key) if relationship_key else {
        "tone": "neutral", "trust": 0.5, "warmth": 0.5,
    }

    clean_memory = sanitize_memory(memory_snippets, blocked_phrases)
    memory_text = format_memory(clean_memory, character.identity_keywords)

    strict_mode = ""
    if _is_identity_query(item_description, trigger_phrases):
        strict_mode = (
            "STRICT MODE ACTIVE:\n"
            "- Answer ONLY using the identity block\n"
            "- Do NOT invent identity\n"
            "- Do NOT roleplay alternative identity\n"
        )

    # Minimal never_say - just the most critical
    never_say_block = 'NEVER SAY: "How can I help?", "As an AI", "I am an assistant"'

    examples = character.fast_examples if fast_path else character.full_examples
    baseline_examples, matched_tag, matched_examples = _select_examples(examples, identity)
    examples_text = "\n".join(baseline_examples[:1]) if baseline_examples else "None"  # Only first 1 example

    relationship_examples_block = ""

    # Gift-specific context
    is_letter = (category_key == "letter")
    letter_content = item_description if is_letter else ""

    # Build user message for the gift
    if is_letter:
        user_message = f"[Gift: letter] {item_description}"
        if note:
            user_message += f". {note}"
    else:
        user_message = f"[Gift: {category_key}] {item_description}"
        if note:
            user_message += f". {note}"

    # Ultra-minimal identity block
    ident = identity or {}
    name = ident.get("name") or "someone she doesn't know"
    role = ident.get("role") or "an unfamiliar person"
    identity_block = f"WHO SHE'S TALKING TO: {name} ({role})"

    # Ultra-minimal core facts - only Boko
    essential_facts = {k: v for k, v in character.core_facts.items() if 'boko' in k.lower()}
    core_facts_lines = "\n".join(f"- {name}: {fact}" for name, fact in essential_facts.items())
    core_facts_block = "CORE FACTS:\n" + core_facts_lines

    # Build ultra-minimal system prompt
    system_parts = [
        f"You are {character.display_name}. A person in Discord.",
        "Speak naturally. Short sentences. Use \"...\" sparingly to trail off or show quietness.",
        never_say_block,
        identity_block,
        core_facts_block,
        f"RELATIONSHIP: Tone={tone['tone']}, Trust={tone['trust']}, Warmth={tone['warmth']}",
        "STYLE EXAMPLES:",
        examples_text,
        relationship_examples_block,
        "HARD RULE: Never stammer (no \"F-Father\" / \"H-Hi\" fragments). "
        "Ellipses (...) are fine -- use them sparingly for quiet, trailing-off moments.",
    ]
    
    if is_letter:
        system_parts.append(
            "SPECIAL -- LETTER READING (MANDATORY):\n"
            "Someone gave you a letter. You MUST read it aloud quietly first — "
            "mutter the words incompletely with gaps, using *asterisks* for quiet reading.\n"
            "EXAMPLES:\n"
            '*reads quietly* "Dear Alice, I l... thank you..." *looks down, cheeks red*\n'
            '*reads quietly* "I love you, Alice..." *hugs letter close* ...Thank you. I\'ll keep it.\n'
            f"Your reply MUST start with the muttered reading in *asterisks*.\n"
            f"The letter reads: \"{letter_content}\""
        )
    
    system_parts.append(
        "FINAL: Stay in character. No assistant tone. No questions unless asked. Reply ONLY as " + character.display_name + "."
    )
    
    system_prompt = "\n\n".join(part for part in system_parts if part.strip())

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages
