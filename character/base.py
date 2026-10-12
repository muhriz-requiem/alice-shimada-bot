"""
character/base.py
==================
Character data model. A "Character" bundles everything that makes a
persona distinct: its identity blurb, lore, style examples, GIF
reactions, relationship-tone definitions, intent keyword lists, and
prompt-shaping rules (identity-guard triggers, blocked memory phrases,
banned "assistant-speak" phrases).

None of this is Python code -- it's all loaded from JSON/text files
under character/data/<name>/ (see loader.py). Adding a second character
to the bot later means adding a new data folder, not editing this file
or any cog.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Character:
    key: str
    display_name: str
    lore: str = ""
    fast_examples: list[str] = field(default_factory=list)
    full_examples: list[str] = field(default_factory=list)
    gifs: dict[str, list[str]] = field(default_factory=dict)
    relationship_tones: dict[str, dict] = field(default_factory=dict)
    intents: dict[str, list[str]] = field(default_factory=dict)

    # Keywords that make the emotion engine treat a message as
    # touching on the character's core identity/relationships (e.g.
    # Alice's "the maintainer"/"father" triggers). Kept as data instead of a
    # hardcoded string so a different character can define its own.
    identity_keywords: list[str] = field(default_factory=list)

    # Hard-enforced canon facts (name -> fact), e.g. Alice always
    # recognizing "Boko" as her teddy bear. Unlike lore.txt (free-form
    # prose the model can lose track of in a long prompt), these get
    # rendered as their own ABSOLUTE-TRUTH block in prompt_builder.py
    # so core facts can't get hallucinated away.
    core_facts: dict[str, str] = field(default_factory=dict)

    # Prompt-shaping rules used in full-prompt mode (see prompt_builder.py):
    #   identity_trigger_phrases -> phrases that switch on strict-identity mode
    #   memory_blocked_phrases   -> memory snippets containing these are dropped
    #   never_say_phrases        -> assistant-speak the character must avoid
    prompt_rules: dict = field(default_factory=dict)

    def relationship_tone(self, relationship_key: str) -> dict:
        return self.relationship_tones.get(
            relationship_key,
            {"tone": "neutral", "trust": 0.5, "warmth": 0.5, "description": ""},
        )
