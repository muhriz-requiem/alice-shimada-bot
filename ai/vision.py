"""
ai/vision.py
============
Screen-glance reply generation: persona prompt + screenshot ->
(spoken line, screen notes). Sits on top of the configured AIClient's
``vision_chat`` (GroqClient only today; other backends raise).
"""
from __future__ import annotations

import config
from bot.logging_setup import get_logger

log = get_logger("ai.vision")


async def glance_reply(
    ai_client,
    persona_messages: list[dict],
    jpeg_bytes: bytes,
    window_title: str,
    *,
    user_id: str,
    channel_id: str,
    deixis: bool,
    question: str = "",
) -> tuple[str, str] | None:
    """(spoken, notes) about a screenshot, or None on failure.

    persona_messages: the compact voice prompt (from
    prompt_builder.build_voice_prompt) -- mutated here so the FINAL user
    message carries the image + glance instruction.
    deixis: False = she volunteered a comment on what she saw;
            True = she is answering a question that refers to the screen.
    question: the user's actual utterance (injected into the instruction
              so the vision model knows what was asked of it).
    spoken: the in-character line to speak aloud.
    notes: faithful transcription of the key on-screen text (the
           SAY/NOTES contract); the pipeline keeps it briefly so
           follow-up questions without the word "screen" still have
           the screen's content.
    """
    from voice import screen_look

    if not hasattr(ai_client, "vision_chat"):
        log.warning("AI backend has no vision_chat; glance unavailable")
        return None

    ask_note = f'\n(The speaker asked: "{question[:200]}")' if question else ""
    if deixis:
        instruction = screen_look.DEIXIS_INSTRUCTION + ask_note
    else:
        # Explicit looks carry the speaker's request too: without it the
        # model only sees "react casually" and would ignore e.g. "answer
        # the question on the screen". Tray/ambient glances have no
        # question and get the plain instruction.
        instruction = screen_look.GLANCE_INSTRUCTION + ask_note

    messages = screen_look.attach_image_to_last_user(
        persona_messages, jpeg_bytes, instruction, window_title
    )
    try:
        text = await ai_client.vision_chat(
            messages,
            user_id=user_id,
            channel_id=channel_id,
            model=config.SCREEN_LOOK_MODEL,
            max_tokens=200,  # spoken line + NOTES transcription
        )
    except Exception as e:
        log.warning(f"Vision glance failed: {e}")
        return None

    from ai.postprocess import clean_response

    cleaned = clean_response(text)
    if not cleaned or cleaned == config.FALLBACK_REPLY:
        return None
    spoken, notes = screen_look.parse_glance_output(cleaned)
    if not spoken:
        return None
    return spoken, notes
