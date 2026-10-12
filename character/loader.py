"""
character/loader.py
=====================
Loads a Character from character/data/<name>/. Every file is optional
except identity is implied by the folder itself existing -- missing
files just mean that piece of the character is empty, logged as a
warning rather than crashing startup, since a character bot should
still boot even with an incomplete data set.
"""

from __future__ import annotations

import json
from pathlib import Path

from bot.logging_setup import get_logger
from character.base import Character

log = get_logger("character.loader")

_DATA_ROOT = Path(__file__).resolve().parent / "data"


def _read_json(path: Path, default):
    if not path.exists():
        log.warning(f"Character data file missing, using default: {path}")
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.error(f"Failed to read {path}: {e}. Using default.")
        return default


def _read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        log.warning(f"Character data file missing, using default: {path}")
        return default
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        log.error(f"Failed to read {path}: {e}. Using default.")
        return default


def load_character(name: str) -> Character:
    folder = _DATA_ROOT / name
    if not folder.exists():
        raise FileNotFoundError(
            f"No character data folder found for '{name}' at {folder}"
        )

    examples = _read_json(folder / "examples.json", {"fast": [], "full": []})
    identity = _read_json(folder / "identity.json", {"display_name": name.title()})
    prompt_rules = _read_json(
        folder / "prompt_rules.json",
        {
            "identity_trigger_phrases": [],
            "memory_blocked_phrases": [],
            "never_say_phrases": [],
        },
    )

    character = Character(
        key=name,
        display_name=identity.get("display_name", name.title()),
        lore=_read_text(folder / "lore.txt"),
        fast_examples=examples.get("fast", []),
        full_examples=examples.get("full", []),
        gifs=_read_json(folder / "gifs.json", {}),
        relationship_tones=_read_json(folder / "relationship_tones.json", {}),
        intents=_read_json(folder / "intents.json", {}),
        identity_keywords=identity.get("identity_keywords", []),
        core_facts=identity.get("core_facts", {}),
        prompt_rules=prompt_rules,
    )

    log.info(f"Loaded character '{character.display_name}' ({name})")
    return character
