"""
desktop/
========
Desktop companion mode: the same Alice voice pipeline (STT -> decision ->
Groq -> GPT-SoVITS) running against local audio devices instead of Discord
voice channels.

- desktop.local_client : duck-typed voice client that plays TTS to speakers
- desktop.mic_input    : microphone feed into the existing AliceVoiceSink
- desktop.cli          : headless runner (`python -m desktop.cli`)

Discord stays untouched; the discord bot entrypoint (main.py) keeps working
exactly as before.
"""
