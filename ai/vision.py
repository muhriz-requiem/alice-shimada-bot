"""
ai/vision.py
============
Screen-glance reply generation: persona prompt + screenshot -> one
in-character line. Sits on top of the configured AIClient's
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
) -> str | None:
    """One in-character comment about a screenshot, or None on failure.

    persona_messages: the compact voice prompt (from
    prompt_builder.build_voice_prompt) -- mutated here so the FINAL user
    message carries the image + glance instruction.
    deixis: False = she volunteered a comment on what she saw;
            True = she is answering a question that refers to the screen.
    question: the user's actual utterance for the deixis case.
    """
    from voice import screen_look

    if not hasattr(ai_client, "vision_chat"):
        log.warning("AI backend has no vision_chat; glance unavailable")
        return None

    if deixis:
        instruction = (
            f'{screen_look.DEIXIS_INSTRUCTION}\n(Father asked: "{question[:200]}")'
        )
    else:
        instruction = screen_look.GLANCE_INSTRUCTION

    messages = screen_look.attach_image_to_last_user(
        persona_messages, jpeg_bytes, instruction, window_title
    )
    try:
        text = await ai_client.vision_chat(
            messages,
            user_id=user_id,
            channel_id=channel_id,
            model=config.SCREEN_LOOK_MODEL,
            max_tokens=120,
        )
    except Exception as e:
        log.warning(f"Vision glance failed: {e}")
        return None

    from ai.postprocess import clean_response

    cleaned = clean_response(text)
    if not cleaned or cleaned == config.FALLBACK_REPLY:
        return None
    return cleaned.strip()
