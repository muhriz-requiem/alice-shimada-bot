"""
character/memory_shaping.py
=============================
Ported from the original character.py, generalized to not be
Alice-specific. This shapes raw memory-graph snippets before they go
into a prompt:

- classify_memory: tags a snippet as FACT / EMOTION / IDENTITY / GENERAL
  so the model can weigh it appropriately.
- sanitize_memory: drops snippets matching a character's configured
  "blocked phrases" list (e.g. non-canon roleplay concepts that would
  contradict the character's lore if they leaked back in as "memory").
- format_memory: renders the final tagged, sanitized list as text.
"""

from __future__ import annotations


def classify_memory(text: str, identity_keywords: list[str] | None = None) -> str:
    msg = text.lower()

    if any(x in msg for x in ["my name is", "i am", "i like", "my favorite"]):
        return "FACT"

    if any(x in msg for x in ["feel", "frustrated", "happy", "sad", "trust"]):
        return "EMOTION"

    if identity_keywords and any(k.lower() in msg for k in identity_keywords):
        return "IDENTITY"

    return "GENERAL"


def sanitize_memory(snippets: list[str], blocked_phrases: list[str]) -> list[str]:
    if not snippets:
        return []

    clean = []
    for m in snippets:
        low = m.lower()
        if any(p.lower() in low for p in blocked_phrases):
            continue
        clean.append(m)

    return clean[-12:]


def format_memory(snippets: list[str], identity_keywords: list[str] | None = None) -> str:
    if not snippets:
        return "None"

    lines = [f"[{classify_memory(m, identity_keywords)}] {m}" for m in snippets[-10:]]
    return "\n".join(lines)
