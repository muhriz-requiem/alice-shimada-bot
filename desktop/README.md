# Alice — Desktop Companion (deep dive)

Standalone voice companion running the same pipeline as the Discord bot
(STT → decision → AI → TTS) against local mic/speakers, with the avatar
rendered by the external **MateEngine** engine. The root `README.md` is
the product brochure and quick start; this file is the desktop-specific
reference.

```
desktop/
├── app.py                    # Tray entry point (pystray): Companion toggle, Mute,
│                             #   STT hosted↔local, engine restart, Discord toggle,
│                             #   debug console; single-instance via CreateMutexW
├── cli.py                    # Headless runner: mic mode / --inject WAV self-test /
│                             #   --no-speaker / --list-devices
├── voice_app.py              # THE shared bootstrap: CompanionSession + boot_companion
│                             #   (tray and CLI drive the same code path)
├── prewarm.py                # Launch-time warm-up: VAD → GPT-SoVITS → STT (GPU-safe order)
├── boot_timeline.py          # Cold-start stage recorder ("Desktop boot (...)" lines)
├── engine_manager.py         # MateEngineX lifecycle: spawn/attach, ACTIVE bridge polling
│                             #   (BOOT_WAIT is a ceiling, not a sleep), face takeover
│                             #   with retries, 10 s health loop, terminate on quit
├── mate_bridge.py            # MtBridge client (TCP 127.0.0.1:8495, env MT_BRIDGE_PORT):
│                             #   ping/sys/probe/avinfo, expressions + face takeover,
│                             #   clear_mouth / speak_visemes / speak_timeline
├── lip_sync_subscriber.py    # Subscribes to voice.lip_sync; per-sentence WAV
│                             #   → viseme timeline → bridge.speak_timeline() worker thread
├── mouth.py / phonemes.py    # Weighted viseme timelines (kana path for JA; persistent
│                             #   phoneme worker inside the SoVITS venv for EN)
├── local_client.py / mic_input.py  # Discord-transport-compatible 20 ms/48 k stereo,
│                             #   echo gate (VOICE_DESKTOP_ECHO_GATE), mic barge-in
├── discord_service.py        # In-process AliceBot toggle (shares pipeline/memory)
├── convert/                  # Reproducible VRM build: patch_vrm.py (GLB weight patcher,
│                             #   env-parameterized), check_patched.py, build_vrm.py;
│                             #   documented deploy cycle in its README
├── tools/                    # Dev probes (not runtime): make_icon.py, phoneme_worker.py,
│                             #   composition_bench.py, alpha_test.py, overlay_cycle_test.py,
│                             #   probe_live_overlay.py, verify_live_overlay.py
└── __init__.py
```

## Cold start (how to measure it)

Every launch emits one line per boot stage plus a summary:

```
INFO  desktop.boot   boot: app_start                 @    0.00s
INFO  desktop.boot   boot: engine_managing           @    0.21s (+0.21)
...
INFO  desktop.boot   Desktop boot (tray): 2.4s total | ...
INFO  desktop.boot   Desktop boot (companion): 41.8s total | ... stt_ready @16.2s ... session_ready @41.8s
```

- `boot: sovits_up` — the GPT-SoVITS server answered probe (prewarm.py
  ran `ensure_server` + `probe` at launch, not on your first sentence).
- `boot: stt_ready` — the whisper model finished preloading.
- `boot: session_ready` — mic is feeding; you can talk.

Stages arrive in parallel (engine bridge vs. prewarm thread), so read the
summary, not just the last line. Env switches: `DESKTOP_PREWARM=0`,
`DESKTOP_BOOT_TIMELINE=0`.

## Avatar

- The companion never renders the avatar — `MateEngineX.exe` does
  (default path `D:\Temp\opencode\MateEngine\MateEngineX.exe`, override
  with `MATE_ENGINE_EXE`). This app starts it on launch, polls the
  bridge, takes over the face, and stops it on quit.
- The deployed `Alice.vrm` **is generated output**: rebuild with
  `desktop/convert/build_vrm.py` (pure GLB weight editing, output
  hash-verified). Never hand-edit the deployed file.
- After the engine restarts, `engine_manager` re-takes-over the face
  automatically; the health loop restores status/over-takeover if the
  bridge drops.
- The PMX → VRM conversion route is retired (dropped blend shapes → no
  expressions; see root `OVERLAY_FAILURES.md`). Original MMD source is
  preserved at `assets/models/Garupan Models/...`.

## What did NOT happen here (do not resurrect)

The earlier in-app overlay attempts are documented as dead ends:

1. **Qt WebEngine + PyQt6** — DLL chain never resolved (entitlement
   issues); all related files were deleted.
2. **pywebview / WinForms-WebView2 (Edge Chromium)** — compositor paints
   an opaque white background behind the content; color-keyed layered
   windows cannot win (`desktop/tools/alpha_test.py`,
   `composition_bench.py` keep the evidence). Note: with the overlay
   retirement, `pywebview`/`pythonnet`/`PyQt6` are no longer in
   `requirements.txt` — those probes need a manual `pip install` to run.

The MateEngine external-renderer route is the answer to all of the above.

## Desktop semantics worth remembering

- One microphone → one speaker: diarization is opt-in
  (`--diarize`); pyannote (~1 GB VRAM) stays off otherwise.
- STT hosted↔local flips live from the tray (`VOICE_STT_HOSTED`); hosted
  sends mic audio off-machine, so it stays an explicit choice. Local
  whisper warm-up is scheduled independently, so the flip is instant.
- `build_pipeline()` forces TTS readiness (`readiness._state = READY`)
  — the desktop does not run the Discord readiness loop; prewarm.py
  covers what that loop did (VAD/SoVITS/STT warm-up).
- Mic barge-in: sustained loud speech interrupts playback even with the
  echo gate on (`mic_input.py`); with headphones (`echo gate false`),
  normal-volume mid-sentence interruption works.
- Teardown drains speech tasks + mixer so the loop closes cleanly
  (no `Event loop is closed`, exit 0).
