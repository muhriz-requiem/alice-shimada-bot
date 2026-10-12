"""
tests/test_groq_warm_import.py
==============================
Covers the event-loop freeze fix: GroqClient.__init__ touches
client.chat so the SDK's lazy openai.resources import (plus the jiter
native module) happens at boot instead of on the first generate_reply.

Left lazy, the first voice reply paid the whole import synchronously on
the event loop (~55s under Defender/disk contention), blocking voice
heartbeats long enough to drop the voice connection.
"""

from __future__ import annotations

import sys

from ai.groq_client import GroqClient


def test_openai_resources_warm_after_init():
    GroqClient()
    assert "openai.resources.chat" in sys.modules
