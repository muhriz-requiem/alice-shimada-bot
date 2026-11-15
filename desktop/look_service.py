"""
desktop/look_service.py
=======================
Glances for the desktop companion: wires the pipeline's screen-look
provider to screen_watch, runs the ambient governor (idle-time remarks
about a CHANGED screen), and applies the MtBridge expression preset
while she comments. The tray owns the permission toggle:
``set_allowed(True/False)`` — no capture happens without it.

Ambient rules (all required): SCREEN_LOOK_ENABLED, tray allowed, ambient
configured on, mixer idle, no speech tasks, cooldown elapsed, hourly
cap not exhausted, screen changed >= SCREEN_LOOK_MIN_CHANGE.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque

import config
from bot.logging_setup import get_logger

log = get_logger("desktop.look_service")


class LookService:
    def __init__(self, pipeline, mate_bridge=None) -> None:
        self._pipeline = pipeline
        self._bridge = mate_bridge
        self._allowed = False  # tray permission (default: never capture)
        self._last_thumb: object | None = None
        self._last_glance_at = 0.0
        self._hour_stamps: deque[float] = deque()
        # Separate task slots: the ambient loop must never block a tray
        # "Look now" (they used to share one slot, which silently killed
        # the tray item whenever ambient was running).
        self._explicit_task: asyncio.Task | None = None
        self._ambient_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        # The pipeline speaks glances through these hooks.
        pipeline.screen_look_provider = self._provide_frame

    # --- tray API -------------------------------------------------------
    @property
    def allowed(self) -> bool:
        return self._allowed

    def set_allowed(self, value: bool) -> None:
        """Tray permission gate. Granting it also runtime-enables the
        feature (the .env master switch stays the boot default); the
        per-boot permission checkbox remains the privacy gate."""
        self._allowed = bool(value)
        config.SCREEN_LOOK_ENABLED = self._allowed
        log.info(f"Screen glances allowed: {self._allowed}")

    def look_now_async(self, monitor: str | None = None) -> None:
        """Tray menu item: one immediate on-demand glance."""
        if self._explicit_task is None or self._explicit_task.done():
            self._explicit_task = asyncio.get_running_loop().create_task(
                self._explicit_glance(monitor=monitor)
            )

    # --- provider (called from the pipeline, voice turn context) --------
    async def _provide_frame(self, monitor: str | None = None):
        if not self._allowed:
            return None
        from desktop import screen_watch

        return await asyncio.to_thread(screen_watch.capture, monitor)

    # --- expression touch ------------------------------------------------
    def _apply_expression(self) -> None:
        preset = (config.SCREEN_LOOK_EXPRESSION or "").strip()
        if not preset:
            return
        weight = max(0.0, min(1.0, config.SCREEN_LOOK_EXPRESSION_WEIGHT))
        try:
            if self._bridge is None:
                from desktop.mate_bridge import MateBridge

                bridge = MateBridge()
                if bridge.connect():
                    self._bridge = bridge
                else:
                    return
            self._bridge.command(f"exp {preset} {weight:.2f}")
        except Exception as e:  # noqa: BLE001 - cosmetic only
            log.debug(f"expression touch failed: {e}")

    # --- explicit glance (tray-triggered / "second screen" asks) ----------
    async def _explicit_glance(self, monitor: str | None = None) -> None:
        from desktop import screen_watch
        from metrics import record_glance

        if not (config.SCREEN_LOOK_ENABLED and self._allowed):
            log.info("Glance requested but disabled/not allowed")
            record_glance("refused")
            return
        t0 = time.monotonic()
        frame = await self._provide_frame(monitor)
        if frame is None:
            log.info("Glance refused (denylist/capture failed)")
            record_glance("refused")
            return
        # Ambient thumbs track the screen the owner actually saw.
        try:
            import io as _io

            from PIL import ImageGrab
            self._last_thumb = screen_watch.thumbnail(
                ImageGrab.open(_io.BytesIO(frame.jpeg))
            )
        except Exception:  # noqa: BLE001
            pass
        comment = await self._make_comment(
            frame, deixis=False, question=""
        )
        if comment:
            record_glance("ok", time.monotonic() - t0)
            self._stamp()
            await self._pipeline.speak_glance(0, comment)
            self._apply_expression()
        else:
            record_glance("failed")

    # --- shared comment path (explicit + ambient) -------------------------
    async def _make_comment(self, frame, *, deixis: bool, question: str) -> str | None:
        from ai import vision as _vision
        from character import prompt_builder

        identity = {}
        try:
            from social.identities import get_identity_async
            identity = await get_identity_async(str(config.OWNER_IDS and next(iter(config.OWNER_IDS))))
        except Exception:  # noqa: BLE001
            identity = {}
        messages = prompt_builder.build_voice_prompt(
            self._pipeline.bot.character,
            identity=identity if identity.get("is_authoritative") else None,
            relationship_key=None,
            history=[],
            user_message="(Owner's screen — see attached image)",
            extra_context="",
        )
        glance = await _vision.glance_reply(
            self._pipeline.bot.ai_client,
            messages,
            frame.jpeg,
            frame.title,
            user_id=str(next(iter(config.OWNER_IDS), "0")),
            channel_id="0",
            deixis=deixis,
            question=question,
        )
        if glance is None:
            return None
        spoken, notes = glance
        if notes:
            # Tray/ambient glances feed the follow-up context too.
            with contextlib.suppress(Exception):  # context is best-effort
                self._pipeline.remember_screen(frame.title, notes)
        return spoken

    def _stamp(self) -> None:
        now = time.time()
        self._last_glance_at = time.monotonic()
        self._hour_stamps.append(now)
        cutoff = now - 3600.0
        while self._hour_stamps and self._hour_stamps[0] < cutoff:
            self._hour_stamps.popleft()

    # --- ambient governor --------------------------------------------------
    def start_ambient(self) -> None:
        if self._ambient_task is not None and not self._ambient_task.done():
            return
        self._stop = asyncio.Event()
        self._ambient_task = asyncio.get_running_loop().create_task(
            self._ambient_loop()
        )

    async def stop_ambient(self) -> None:
        if self._ambient_task is not None and not self._ambient_task.done():
            self._stop.set()
            try:
                await asyncio.wait_for(self._ambient_task, timeout=5.0)
            except Exception:  # noqa: BLE001
                self._ambient_task.cancel()
        self._ambient_task = None

    async def _ambient_loop(self) -> None:
        """Every 20 s: decide whether Alice should glance on her own."""
        from desktop import screen_watch

        while not self._stop.is_set():
            try:
                await asyncio.sleep(20.0)
                if not self._ambient_eligible():
                    continue
                # Capture + change check (cheap; RAM only)
                img = await asyncio.to_thread(screen_watch._capture_image)
                if screen_watch.denylisted(screen_watch.foreground_title()):
                    continue
                thumb = await asyncio.to_thread(screen_watch.thumbnail, img)
                change = screen_watch.change_fraction(self._last_thumb, thumb)
                if change < config.SCREEN_LOOK_MIN_CHANGE:
                    continue
                self._last_thumb = thumb
                frame = await self._provide_frame()
                if frame is None:
                    continue
                comment = await self._make_comment(frame, deixis=False, question="")
                if not comment:
                    continue
                self._stamp()
                log.info(f"Ambient glance: {comment[:80]!r}")
                await self._pipeline.speak_glance(0, comment)
                self._apply_expression()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - governor must not die
                log.warning(f"ambient glance loop error: {e}")

    def _ambient_eligible(self) -> bool:
        if not (config.SCREEN_LOOK_ENABLED and self._allowed):
            return False
        if not config.SCREEN_LOOK_AMBIENT:
            return False
        if self._pipeline.mixer.is_playing(0):
            return False
        if time.monotonic() - self._last_glance_at < max(
            10.0, config.SCREEN_LOOK_AMBIENT_COOLDOWN_S
        ):
            return False
        cutoff = time.time() - 3600.0
        recent = sum(1 for t in self._hour_stamps if t >= cutoff)
        if recent >= max(1, config.SCREEN_LOOK_AMBIENT_MAX_PER_HOUR):
            return False
        # Turn state: only when the conversation is idle/lull-ish and
        # nobody is mid-speech or mid-reply.
        try:
            from voice.turn_detector import TurnState

            state = self._pipeline.turn_detector.get_state(0)
            if state not in (
                TurnState.IDLE,
                TurnState.PAUSE,
                TurnState.LULL,
                TurnState.SINGLE_SPEAKER,
            ):
                return False
        except Exception:  # noqa: BLE001
            pass
        return True
