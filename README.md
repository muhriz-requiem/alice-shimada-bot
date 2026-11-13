# Alice — Desktop Companion

**Alice** (Shimada Arisu, *Girls und Panzer*) as a standalone, always-with-you
AI voice companion. Talk to her from the taskbar; she listens through your
microphone, thinks on Groq, speaks in her own fine-tuned voice
(GPT-SoVITS), and appears as a real 3D VRM avatar rendered by the external
MateEngine engine. Her short-term memory, emotion, and relationship state
make her remember you across sessions.

The same codebase can also run as a **Discord bot** (text + voice) — that is
now the secondary deployment mode, documented mid-file under
*Running the Discord bot instead*.

---

## What the desktop companion does

```
Microphone ──▶ AliceVoiceSink (Silero VAD) ──▶ STT (faster-whisper, local)
             or Groq Whisper (hosted, tray-switchable)
           ──▶ Response decision engine (I1/I2/S1/S2, shared-call rules)
           ──▶ Groq streaming reply  ──▶ GPT-SoVITS sentence streaming
           ──▶ Speakers (with barge-in)  ──▶ MateEngine viseme lip-sync
```

- **Tray app** (`desktop/app.py`, pystray): Companion on/off, Mute, STT
  engine hosted↔local runtime flip, avatar-engine restart, Discord bot
  toggle, debug console. Single-instance guarded (`CreateMutexW`).
- **One shared bootstrap** (`desktop/voice_app.py::CompanionSession`) is
  driven identically by the tray and the CLI, so the two cannot drift.
- **Cold start is warm**: `desktop/prewarm.py` brings the voice stack up at
  app launch — VAD, then the GPT-SoVITS server, then the STT model — while
  the tray appears instantly and the avatar engine boots in parallel.
- **Boot timeline**: every launch logs a `Desktop boot (...)` line with
  per-stage timings (`desktop/boot_timeline.py`; `DESKTOP_BOOT_TIMELINE=0`
  silences it).
- **Avatar**: MateEngineX (Unity) renders the SFW-patched VRM; this app
  never renders it. `desktop/mate_bridge.py` speaks the MtBridge text
  protocol over TCP `127.0.0.1:8495` (expressions, face takeover, and the
  fire-and-forget `speak_timeline()` viseme streaming fed from
  `voice/lip_sync` + `desktop/lip_sync_subscriber.py`).
- **Barge-in**: raise your voice while she talks and she stops.
  Echo-gating behavior follows `VOICE_DESKTOP_ECHO_GATE` (speakers: mic
  drops while she talks; headphones: always-on listening — the everyday
  setup here).

## Quick start

```powershell
python -m venv venv
pip install -r requirements.txt                          # CPU base
# GPU (GTX 1650 Ti class): use the CUDA wheels instead:
#   pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126
copy .env.example .env    # fill in: GROQ_API_KEY (required), OWNER_IDS
python -m desktop.app     # tray app (pythonw for console-free daily use)
```

| Run | What it is |
|---|---|
| `venv\Scripts\pythonw.exe -m desktop.app` | Daily driver; tray icon, no console |
| `python -m desktop.app` | Same, with console logs |
| `python -m desktop.cli` | Headless mic → Alice loop (debug) |
| `python -m desktop.cli --inject turn.wav` | Self-test: replay a WAV through the real pipeline |
| `python -m desktop.cli --no-speaker` | Silent self-test (CI-safe) |
| `python -m desktop.cli --list-devices` | Enumerate audio devices |

`.env` essentials beyond the AI keys:

| Variable | Meaning |
|---|---|
| `VOICE_TTS_ENGINE` | `sovits` (her own fine-tuned voice, auto-starts the local server) or `edge` (cloud fallback + RVC) |
| `VOICE_STT_ENGINE` / `VOICE_STT_MODEL` / `VOICE_STT_DEVICE` | STT (default: `whisper medium` on CUDA — picked via `manage.py eval-stt` on the owner's accented speech) |
| `VOICE_STT_HOSTED` | transcribe via Groq Whisper instead (audio leaves the machine; off by default) |
| `VOICE_DESKTOP_ECHO_GATE` | `true` for speakers, `false` for headphones |
| `DESKTOP_PREWARM` | `0` disables the launch-time warm-up |
| `DESKTOP_BOOT_TIMELINE` | `0` silences cold-start timing logs |
| `MATE_ENGINE_EXE`, `MT_BRIDGE_PORT` | avatar engine path / bridge port overrides |
| `VOICE_LIP_SYNC` | `false` disables the avatar mouth (avatar still renders) |

First launch: the tray is up in seconds; the voice stack finishes warming
in the background (watch for `boot: stt_ready` / `boot: sovits_up`). Click
**Companion: off** to start listening. GPT-SoVITS needs its own install at
`D:\GPT-SoVITS` (the bot auto-starts `api_v2.py` and health-probes it; see
the *Voice engine reference* below).

## Architecture

```
main.py / manage.py        Discord entrypoint + maintenance CLI
config.py                  All settings, env-driven (BOT_NAME prefixes for multi-bot)

desktop/                   THE PRODUCT — tray companion
  app.py                   Tray: menu, single instance, engine ownership
  voice_app.py             CompanionSession: pipeline → sink → client → mic
  prewarm.py               Launch-time warm-up: VAD → SoVITS → STT
  boot_timeline.py         Cold-start stage timing
  engine_manager.py        MateEngineX lifecycle (spawns, polls bridge, face takeover, health)
  mate_bridge.py           MtBridge client (expressions, speak_timeline)
  lip_sync_subscriber.py   Subscribes to voice/lip_sync; drives the mouth
  mouth.py / phonemes.py   Viseme timelines + persistent phoneme worker
  local_client.py / mic_input.py   Discord-transport-compatible local audio
  discord_service.py       In-process Discord bot toggle
  screen_watch.py          Glance capture + privacy denylist (RAM-only frames)
  look_service.py          Glance provider, ambient governor, expression touch
  convert/                 Reproducible VRM GLB patcher (build_vrm.py)

voice/                     Shared pipeline (identical for Discord + desktop)
  pipeline.py              Orchestrator: STT → decision → AI → TTS → play
  sink.py                  Per-user buffering + Silero VAD
  response.py / turn_detector.py / conversation.py
  screen_look.py           Glance intents + vision-message assembly (headless)
  streaming_tts.py         Sentence split → parallel synthesis → streaming source
  stt.py / stt_hosted.py   Moonshine/Whisper local; Groq hosted (opt-in)
  tts.py / sovits.py / rvc.py   GPT-SoVITS client+auto-server; RVC fallback
  readiness.py             Mute/silent-until-ready gate
  diarize.py / vad.py      pyannote speakers; Silero VAD
  lip_sync.py              Neutral avatar-lip-sync publisher
  dave_patch.py            discord-ext-voice-recv DAVE-decrypt fixes

ai/                        Groq (default), Ollama, Shapes (legacy)
character/                 Data-driven persona → character/data/alice/
memory/ state/             SQLite memory + per-user graph; Redis cross-instance state
emotion/ social/           8-axis per-user emotion; identity registry
bot/ cogs/                 Discord wiring + slash commands (secondary mode)
metrics/ health/           Prometheus + /healthz /ready /metrics (:8080)
templates/                 Jinja2 prompts
tests/                     pytest suite; .github CI (lint → typecheck → test)
data/                      runtime: sqlite, reddit pools, voice models, logs
_legacy/                   Pre-rewrite reference code (not maintained)
```

## Voice engine reference

- **GPT-SoVITS (primary)**: external install at `D:\GPT-SoVITS`, API on
  `127.0.0.1:9880` (`api_v2.py`, `is_half: false` on this 4 GB GPU). The
  companion/Discord bot auto-starts it, health-probes at boot, and falls
  back silently to **Edge-TTS + RVC** if it can't come up. Reference
  audio: `data/voice_models/alice_ref.wav` (keep `prompt_text` empty —
  non-empty references leak into output on this build). The install is
  additionally patched by `desktop/sovits_patch/apply.py` (lazy-BERT:
  ~1.35 GB of Chinese-BERT VRAM that en/ja synthesis never uses stays
  unloaded; see *Recent changes*).
- **Boot watchdog**: while the API port is down, the server's own
  `api_boot.log` is tailed into the console; if it stops growing for
  `VOICE_SOVITS_BOOT_STALL_TIMEOUT` (180 s) the hung child is killed and
  respawned, up to `VOICE_SOVITS_BOOT_ATTEMPTS` inside the
  `VOICE_SOVITS_STARTUP_TIMEOUT` budget.
- **Latency profile** (from `manage.py latency`): TTFA p50 ≈ 2.5-3 s with
  live mode off; ≈ 1.2-1.8 s with `VOICE_LIVE_MODE=true` (predictive STT
  + Groq streaming + SoVITS chunked-streaming first sentence
  (`VOICE_SOVITS_STREAM_MODE`, see *Recent changes*) + banked
  "Hm?" acks). Stage gauges on `/voice-status` and `/metrics`.
- **Device note**: `onnxruntime-gpu` sees only CPU execution providers
  here (CUDA 13.2 > ONNX Runtime's bundled CUDA), so the edge-engine RVC
  fallback runs on CPU; STT/diarization use CUDA via torch normally.
  **Do not install `torchcodec`** in this venv (broken DLLs on this torch
  build; pyannote's harmless one-line warning mentions it).

## Voice behavior rules

- **Shared-call governor** (Discord): when others are present she only
  speaks when addressed, when her name comes up, or when barged in.
- **Desktop**: one mic = everything is for her (owner priority), diarization
  is opt-in (`--diarize` / `VOICE_DIARIZE_MODE`).
- **Name mention** (S2): soft "did I hear my name?" line by default.
- **Proactive ac_knowledgment**: backchannels (`うん`, `mm`) and live-mode
  banked address-acks; per-call cooldowns everywhere.

## The avatar (VRM)

The deployed model **is generated output** — `desktop/convert/` rebuilds it
byte-identically from a source VRM by patching skin weights directly in
the GLB (offline, no Blender round-trip). Deploy cycle:

```powershell
# Stop MateEngineX first; then:
venv\Scripts\python.exe desktop\convert\build_vrm.py     # rebuild & verify
# -> copy patched vrm to the engine path, relaunch MateEngineX,
#    engine_manager reconnects and re-takes-over the face automatically
```

Verification comes built-in (`desktop/convert/check_patched.py`); the
patcher's history (thigh split bands, chest copy, v11.1 metrics) lives in
the git history of those files. The **PMX → VRM conversion route is
retired** — it dropped blend shapes and materials; see
`OVERLAY_FAILURES.md` (postmortem) if curious. The original MMD source
(`Shimada Arisu.pmx` + textures) is preserved under `assets/models/`.

## Persona, memory, data

- Persona lives in `character/data/alice/` (lore, examples, identity/core
  facts, gift catalogue, GIFs, prompts). Adding a character = a folder.
- Per-user memory: SQLite log + importance-weighted in-RAM graph
  (hydrated from SQLite at boot), per-user relationship tier + gift
  counts, per-user 8-axis emotion. Redis optionally shares this across
  bot instances.
- Identity registry: `social/data/identities.json` (Discord ID → named
  role/relationship notes, currently the owner's father-role entry).
- `/alice-gift` increments gifts and upgrades her relationship tier with
  an in-character reaction (letters get read aloud).

## Running the Discord bot instead

Everything above targets the companion; the Discord mode is the same
pipeline behind `bot/` + `cogs/`:

- Entrypoint `python main.py`; slash commands: `/chat`, `/alicepost`,
  `/guppost`, `/aliceconfess`, `/alice-gift`, `/join /leave
  /voice-status`, and owner-only `/admin-*`.
- Full setup → **SETUP_NEW_BOT.md**; `.env.example` documents every
  variable. `AI_BACKEND=groq` default (`GROQ_API_KEY` required,
  `GROQ_RATE_LIMIT_RPM` / `GROQ_TPM_LIMIT` / `GROQ_REASONING_EFFORT` self-
  throttling for the free tier; text `/chat` sends the full persona prompt,
  voice calls send a compact ~430-token one to fit the TPM window).
- Voice in Discord uses literally the same modules as the companion;
  `/join` stays self-muted until TTS readiness confirms (or falls back).
- Multi-bot / hosting: `BOT_NAME` prefixing isolates data dirs; Redis
  (`REDIS_*`) shares emotion/state; `LOG_JSON=true` for aggregation;
  health endpoints on `:8080`; small free-tier hosts are viable since
  inference is off-host.

## Operations

```
python manage.py stats                    # backend, memory size, reddit pools
python manage.py refresh-reddit           # pullpush fetch + pool rebuild
python manage.py reset-memory --user X    # (or --all)
python manage.py latency --wav turn.wav   # real-call TTFA/stage benchmark
python manage.py eval-stt                 # WER matrix on YOUR captured clips
```

- **STT tuning loop (owner-accent realism)**: enable
  `VOICE_STT_CAPTURE=true`, talk to her (clips → `stt_eval/clips/`), correct
  the machine-created `refs.csv` by listening, then re-run `eval-stt` to
  score engines × preprocessing (WER + p50/p95 latency + VRAM), and set the
  winning `VOICE_STT_ENGINE`/`VOICE_STT_MODEL`. Current winner here:
  `whisper-medium` (`int8_float16`), WER 0.32 vs 0.59 for moonshine-base.
- **Health/metrics**: `/healthz` (liveness), `/ready` (Discord + Redis),
  `/metrics` (Prometheus; grafana dashboard file at the repo root).
- **Cleanup policy**: `REDDIT_RAW_RETENTION_FILES` prunes raw dumps.

## Testing & quality

```
venv\Scripts\python -m pytest tests -q        # full suite
venv\Scripts\python -m pytest tests -q --no-cov    # quick, no coverage gate
```

Ruff + mypy configured in `pyproject.toml`; CI (`.github/`) runs
lint → typecheck → tests. Voice/desktop hot paths have dedicated tests
(predictive STT, streaming split/cleaning, SoVITS stream parsing, shared-call
governor, boot watchdogs, desktop app logic).

## Recent changes (Nov 2026)

- **2026-11-12 STT-garbage policy — "ask, don't invent"**: garbled or
  misheard voice transcripts (accents, STT errors) are now handled at the
  *response* layer instead of by more filtering: the voice prompt tells
  Alice that transcripts can be garbled, and to ask for a repeat
  (in character) rather than inventing meaning — and to treat garbled
  past turns as "you didn't hear it well", never as facts. History keeps
  the honest record; memory stays clean (voice turns were never
  persisted to the graph/SQLite — only glance notes and text chat are).
  The existing hallucination filter (phrase loops, prompt echoes,
  silence impostors; deliberately conservative with direct-address
  exemptions) is unchanged and now observable:
  `alicebot_voice_stt_filtered_total{reason=single_word|hallucination}`.
  Eval clips are captured pre-filter, so `manage.py eval-stt` can audit
  whether real speech ever gets eaten. Memory was also reset to a clean
  slate (36 stale test messages wiped); the owner's relationship tier
  self-heals to `guardian-figure` from the identity registry on the
  next message (`auto_relationship` — auto-assigns whenever the tier is
  "stranger").
- **2026-11-11 SCREEN GLANCE — Alice can see your screen** (tray-permission
  mode): ask "Alice, look at my screen" / "What's on my screen?" and she
  captures (RAM-only), checks the window-title denylist, and reacts in one
  in-character line via Groq's vision model (`qwen/qwen3.8-27b`, ~850
  tokens per glance — free-tier friendly). Deixis works too: "What do you
  think of THIS?" captures first, then answers; multi-monitor asks
  ("look at my second screen") capture the other monitor. Tray items:
  **Look at my screen now** and **Screen glances allowed** — the checkbox
  is THE privacy gate (resets each boot; no capture while it's off;
  granting it runtime-enables the feature). Ambient mode
  (`SCREEN_LOOK_AMBIENT`) lets her comment on her own when the screen
  changed and the conversation is idle (cooldown + hourly cap +
  pixel-change gate). Glances can be remembered in the memory graph
  (`SCREEN_LOOK_MEMORY`) and carry a mild expression preset over MtBridge.
  Metrics: `alicebot_glance_total{outcome}` + `alicebot_glance_latency_seconds`
  on `/metrics`. New modules: `voice/screen_look.py` (intents, headless),
  `ai/vision.py`, `desktop/screen_watch.py` (capture/privacy),
  `desktop/look_service.py` (ambient governor); pipeline hook +
  `GroqClient.vision_chat` sharing the RPM/TPM throttle with chat.
  Verified live on the owner's machine: 1280px capture 0.5 s, explicit
  glance comment 4.9 s, deixis 10.7 s — both in character. When no
  capture source exists (Discord bot on a remote host, companion off),
  explicit asks get an honest "I'm not next to a screen I can look at"
  line instead of a confused normal reply; there is deliberately no
  Discord `/alice-look` (remote bots have no meaningful screen).
- **2026-11-11 resilience**: hosted-STT failure now preloads the local
  engine in the background (the first fallback turn still pays the load,
  every later one is warm); Edge-TTS upgraded to 7.3.1 (the old build's
  403 broke the entire fallback chain).
- **2026-11-11 lip-sync fix**: streamed (first) sentences never published
  to lip-sync — the avatar mouth stayed static on every single-sentence
  reply. Fixed in `streaming_tts._synthesize_and_add` (both paths publish
  now) + `tests/test_lip_sync_publish.py`.
- **2026-11-11 AliceGuard (MateEngine mod)**: the engine's "Hover Reactions"
  mod fires an embarrassed crotch-cover animation whenever the mouse passes
  near the avatar's hips — awkward for this character. REA analysis of
  `Assembly-CSharp.dll` mapped the mechanism (`PetVoiceReactionHandler.regions`
  → `VoiceRegion.targetBone/hoverRadius`; the crotch reaction is the
  Spine-bound region the engine labels `"Intime Region"`; the settings-only
  `GlobalHoverObjectsEnabled` flag is mis-wired to `enableParticles` and only
  gates the cosmetic hover object). Fix: `AliceGuard.dll` — a Doorstop loader
  in `D:\Temp\opencode\MateEngine\MtBridge\` that chain-loads MtBridge
  (bridge preserved) and zeroes that region's hover radius every 5 s
  (self-re-applying; head/hand pats untouched; window sitting is a separate
  system). Mode via `ALICE_GUARD_MODE` (`disarm` default / `observe` / `off`);
  log at `MtBridge\guard.log`; source at `D:\Temp\opencode\MateEngine\AliceGuard.cs`
  (compiled with the system `csc`). Full revert: restore
  `doorstop_config.ini` from `.alice_orig` and delete `AliceGuard.dll`.
  Verified live: head hover fires the pat; crotch hover does nothing.
- **2026-11-11 streamed-sentence prosody fix**: the streamed first sentence
  passed api_v2's `cut5` splitter, which carved short replies into tiny
  fragments ("Yeah..." -> a 4-char segment) with a forced 0.3 s
  inter-fragment silence — audible chop/stutter. The streaming request now
  sends `text_split_method: cut0` (no re-splitting; this build rejects
  `"none"`). One sentence = one AR stream: −0.5 s duration, −15% synthesis
  time, no mid-word chunk-join math on short replies.

- **2026-11-11 AliceGuard (MateEngine mod)**: the engine's "Hover Reactions"
  mod fires an embarrassed crotch-cover animation whenever the mouse passes
  near the avatar's hips — awkward for this character. REA analysis of
  `Assembly-CSharp.dll` mapped the mechanism (`PetVoiceReactionHandler.regions`
  → `VoiceRegion.targetBone/hoverRadius`; the crotch reaction is the
  Spine-bound region the engine labels `"Intime Region"`; the settings-only
  `GlobalHoverObjectsEnabled` flag is mis-wired to `enableParticles` and only
  gates the cosmetic hover object). Fix: `AliceGuard.dll` — a Doorstop loader
  in `D:\Temp\opencode\MateEngine\MtBridge\` that chain-loads MtBridge
  (bridge preserved) and zeroes that region's hover radius every 5 s
  (self-re-applying; head/hand pats untouched; window sitting is a separate
  system). Mode via `ALICE_GUARD_MODE` (`disarm` default / `observe` / `off`);
  log at `MtBridge\guard.log`; source at `D:\Temp\opencode\MateEngine\AliceGuard.cs`
  (compiled with the system `csc`). Full revert: restore
  `doorstop_config.ini` from `.alice_orig` and delete `AliceGuard.dll`.
  Verified live: head hover fires the pat; crotch hover does nothing.
- **2026-11-11 streamed-sentence prosody fix**: the streamed first sentence
  passed api_v2's `cut5` splitter, which carved short replies into tiny
  fragments ("Yeah..." -> a 4-char segment) with a forced 0.3 s
  inter-fragment silence — audible chop/stutter. The streaming request now
  sends `text_split_method: cut0` (no re-splitting; this build rejects
  `"none"`). One sentence = one AR stream: −0.5 s duration, −15% synthesis
  time, no mid-word chunk-join math on short replies.

- **2026-11-11 resource & latency overhaul** (4 GB VRAM budget, glitch fix):
  - **Lazy-BERT patch for GPT-SoVITS** (`desktop/sovits_patch/apply.py`):
    api_v2 eagerly loads the Chinese BERT (roberta-large, ~1.35 GB fp32 on
    CUDA) although `get_bert_inf` emits *zero tensors* for every non-`zh`
    language — and Alice only speaks en/ja. The patch defers that load
    until a zh segment actually arrives (GSV_LAZY_BERT, on by default;
    `--check` / `--revert` supported; originals backed up as
    `*.alice_orig.py`). Measured: SoVITS server GPU footprint
    **2,335 MB → 1,197 MB**; re-run the script after any GPT-SoVITS
    update (prewarm warns when the patch is missing).
  - **Server spawn hardening** (`voice/sovits.py`): `-u` unbuffered so
    `api_boot.log` grows from the first second (the stall watchdog now
    sees real progress; previously a healthy boot looked hung for >2 min),
    and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (unless already
    set) so 4 GB fragmentation can't wedge long sessions.
  - **api_v2 streaming mode is now configurable**
    (`VOICE_SOVITS_STREAM_MODE`, default 1 = old fragment mode): the
    streaming first sentence can use api_v2's true chunked modes —
    2 (chunked + anti-hallucination mute matrix, recommended) or 3
    (fixed-length chunks, fastest, mute matrix off). Only the streamed
    opener changes; every later sentence keeps the high-quality batch
    path. Measured on this box (warm, MateEngine rendering):
    tts_first p50/p95 **1.39/2.56 s (mode 1) → 0.97/1.16 s (mode 2) →
    0.38/0.42 s (mode 3)**; TTFA p50 2.64 → 2.12 s (mode 2). With the
    lazy-BERT patch the warm p50 is **0.45 s (p95 0.58)**.
    A/B WAVs captured via `D:\Temp\opencode\stream_ab.py`.
  - **Latency harness cooldown bug fixed** (`manage.py latency`):
    `record_response()` stamps a cooldown on the decision engine that the
    harness never reset, so every post-warmup run was skipped
    ("decision=skip — no reply measured"). It now clears the engine's
    cooldown stamps per run.
  - What this fixes in practice: the streamed first sentence's chunk
    arrival jitter (the p95 tail) was the audible "glitchy voice" source —
    the player inserts silence when synthesis stalls; a 2.5 s p95 stall on
    a ~0.6 s chunk cadence guarantees gaps. The tail is now ~0.6 s and the
    GPU has ~1.5 GB of headroom back, so DWM/MateEngine/Chrome spikes no
    longer threaten synthesis OOMs and fallbacks.
- **2026-11-10/11 desktop voice overhaul**: the companion's audio chain was
  rebuilt after a cascade of field failures on this 4 GB GPU / quiet-mic
  setup. Highlights (all config-driven, see `.env.example`):
  - **Hosted STT default on desktop** (`VOICE_STT_HOSTED=true`): 4 GB VRAM
    cannot fit whisper-medium *inference* beside GPT-SoVITS fp32 +
    MateEngineX — the local CUDA decode wedged the per-guild STT lock
    silently. Local STT preloads are skipped in hosted mode
    (`prewarm.py`/`voice_app.py`/`pipeline.py`), and every decode now runs
    under a hard budget (`VOICE_STT_DECODE_TIMEOUT=60`) so a wedged engine
    releases the lock instead of stalling the pipeline forever.
  - **Discord-class mic DSP** (`desktop/dsp.py`, optional
    `aec-audio-processing` dep): WebRTC APM echo cancellation with Alice's
    own playback fed as the far end (`SdOutput` device callback → feeder
    thread — never on the audio callback), plus noise suppression.
    Ordering matters: **AGC first, APM second** (NS erases quiet speech
    before normalization). The speakers echo gate is bypassed while the
    APM runs.
  - **Mic AGC** (`VOICE_MIC_AGC_*`): normalizes faint capture devices
    (fast attack / slow release, gain cap 64, never attenuates). This
    Fifine T669 delivers speech at ~0.2–0.5% RMS; the shootout showed the
    Realtek array and the G435 Bluetooth mic are dead endpoints here.
  - **Playback smoothing**: the local player is paced to real time (an
    unpaced reader queues the streaming source's silence frames at CPU
    speed — one starvation burst played back as seconds of inserted
    silence, the "fragmented voice"), `SdOutput` pre-roll
    (`VOICE_TTS_PREROLL_MS=250`) holds the device start until a cushion
    is buffered, and the streaming worker merges already-arrived
    sentences into one synthesis job (no inter-sentence gaps).
  - **Discord-call stand-down**: while a real Discord voice session is
    connected (`voice_state.any_discord_voice_active()`), the desktop mic
    feed sleeps — two Alices used to speak through the same headset.
  - **Behavior tuning**: private-call turn-end silence 0.5→1.2 s (no
    mid-sentence barging), address-ack "Hm?" only when the predicted wait
    exceeds 3 s, G435 Bluetooth output name now matched ("Headset (G435
    ..."), diarization off on this box (pyannote's ~1 GB starved SoVITS —
    tts_first 0.7 s → 4 s, a 253 s TTFA outlier), 30 s "sink vitals" log
    line (packet flow + VAD speech %) so silent-mic/hot-mic failures are
    visible.
  - **is_half experiment**: tested fp16 GPT-SoVITS on this GTX 1650 Ti —
    no synthesis gain (tts_first 0.44 s → 0.8 s warm), reverted; keep
    `is_half: false` in the GPT-SoVITS `tts_infer.yaml`.
- **Cold-start overhaul (desktop)**: launch-time prewarm
  (`DESKTOP_PREWARM`), MateEngine bridge polling replacing the fixed
  32 s wait, and boot-timeline measurements. Desktop no longer pays for
  STT/SoVITS/VAD warm-up on the first utterance.
- **Repo cleanup**: retired Blender/mmd_tools/PMX-overlay tree, repo-local
  ffmpeg triplicates, abandoned translation scratch scripts, Timbre/
  Gemini-Voice-Splitter vendored repos and two tooling venvs (~10.6 GB
  freed). `overlay/` (three.js/Qt attempts) is gone; MateEngine is the sole
  avatar route. Packaging (`pyproject.toml`) and `requirements.txt` now
  match reality (`pillow`/`pywin32` added; PyQt6-WebEngine/pywebview/
  pythonnet dropped).
- **Persona depth**: voice calls keep the compact prompt; text gets the
  full one; gift flows and relationship tones data-driven in
  `character/data/alice/`.
