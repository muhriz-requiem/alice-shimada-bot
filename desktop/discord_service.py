"""
desktop/discord_service.py
==========================
In-process Discord bot service for the Alice desktop companion.

Runs the Discord bot (AliceBot) in a dedicated thread with its own
asyncio event loop. Provides thread-safe start()/stop() methods that
the tray menu can call. The bot runs in the same process as the
desktop companion, sharing the same voice pipeline and memory.

Key design points:
- Single process, separate thread + event loop (same as CompanionSession)
- Shares voice pipeline via voice_state (guild-keyed, Discord uses real guild IDs)
- Start/stop are idempotent and thread-safe
- Graceful shutdown: closes Discord connection, waits for loop to finish
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time

from dotenv import load_dotenv

load_dotenv()

from bot.logging_setup import get_logger, setup_logging

setup_logging()
log = get_logger("desktop.discord_service")

import config


class DiscordService:
    """Manages the Discord bot (AliceBot) in a background thread."""

    def __init__(self) -> None:
        self.state = "off"
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._bot: object | None = None  # AliceBot instance
        self._stop_evt: asyncio.Event | None = None
        self._started_at: float | None = None

    def start(self) -> bool:
        """Start the Discord bot in a background thread.

        Returns False if already running or starting.
        """
        if self.state in ("on", "starting", "stopping") or (
            self._thread is not None and self._thread.is_alive()
        ):
            return False

        # Validate Discord token before starting
        if not config.DISCORD_TOKEN:
            self.error = "DISCORD_TOKEN not set in config"
            self.state = "error"
            log.error("Cannot start Discord bot: DISCORD_TOKEN not configured")
            return False

        self.error = None
        self.state = "starting"
        self._thread = threading.Thread(
            target=self._run_bot, daemon=True, name="discord-bot"
        )
        self._thread.start()
        return True

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the Discord bot gracefully.

        Blocks until the bot thread finishes or timeout expires.
        """
        thread = self._thread
        if thread is None:
            self.state = "off"
            return

        loop, evt = self._loop, self._stop_evt
        if loop is not None and evt is not None:
            # Signal the bot to stop from the bot's event loop
            # (no-op RuntimeError if the loop is already closed)
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(evt.set)

        # Wait for thread to finish naturally - give it enough time for bot.close()
        thread.join(timeout=timeout)
        if thread.is_alive():
            log.warning("Discord bot thread did not stop within timeout (%.1fs)", timeout)
            # Wait a bit more
            thread.join(timeout=15.0)
        # Mark as stopped (but preserve a crash recorded by the run task)
        if self.state != "error":
            self.state = "off"
        self._thread = None
        self._loop = None

    def status(self) -> dict:
        return {
            "state": self.state,
            "error": self.error,
            "uptime": time.time() - self._started_at if self._started_at else 0,
        }

    # --- Bot thread ------------------------------------------------------

    def _run_bot(self) -> None:
        """Thread entry point: create event loop, start bot, wait for stop."""
        self._loop = asyncio.new_event_loop()
        self._stop_evt = asyncio.Event()

        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._bot_coro())
        except Exception as exc:  # noqa: BLE001
            log.exception("Discord bot thread crashed")
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "error"
        finally:
            # Clean up loop
            try:
                # Cancel any remaining tasks
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:  # noqa: BLE001
                pass
            asyncio.set_event_loop(None)
            self._loop.close()
            self._loop = None

    async def _bot_coro(self) -> None:
        """Main bot coroutine: validates config, starts bot, waits for stop."""
        try:
            config.validate()
        except config.ConfigError as e:
            log.error(f"Discord config error: {e}")
            self.error = str(e)
            self.state = "error"
            return

        from memory import store as memory_store
        memory_store.init_db()

        from bot.client import AliceBot

        self._bot = AliceBot()
        self._started_at = time.time()
        self.state = "on"
        log.info("Discord bot starting...")

        # Start health server
        loop = asyncio.get_running_loop()
        from health import start_health_server
        health_task = loop.create_task(
            start_health_server(self._bot, config.HEALTH_HOST, config.HEALTH_PORT)
        )

        # Run the bot. bot.start() blocks until close() -- so the stop
        # signal CANNOT be awaited after it (that code is unreachable).
        # Instead: start() as a task, await the tray's stop event in
        # parallel, then close() makes start() return and join it.
        run_task = loop.create_task(self._bot.start(config.DISCORD_TOKEN))

        # A dead run_task must surface to the tray even when nobody
        # awaits it (e.g. login failure): without this the task's
        # exception sits unretrieved and the tray shows "on" forever.
        def _on_run_done(task: asyncio.Task) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "error"
                log.error("Discord bot exited: %s", exc)

        run_task.add_done_callback(_on_run_done)

        try:
            # Wake on EITHER the tray's stop signal OR the run task dying
            # on its own (bad token, network error) -- otherwise a dead
            # bot leaves the thread blocked until the user clicks stop.
            stop_wait = asyncio.ensure_future(self._stop_evt.wait())
            try:
                await asyncio.wait(
                    [stop_wait, run_task], return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                stop_wait.cancel()

            if not run_task.done():
                log.info("Stop signal received, closing Discord connection...")
                try:
                    await asyncio.wait_for(self._bot.close(), timeout=10.0)
                except TimeoutError:
                    log.warning("bot.close() timed out, forcing shutdown")
                try:
                    # start() returns once close() completed; join to swallow
                    # its result (the done-callback already handled errors).
                    await asyncio.wait_for(run_task, timeout=15.0)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.warning("Discord run task ended: %s", exc)
            else:
                log.info("Discord run task already ended (state=%s)", self.state)

        except Exception as exc:  # noqa: BLE001
            log.exception("Discord bot stop error")
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "error"
        finally:
            if not run_task.done():
                run_task.cancel()
            health_task.cancel()
            log.info("Discord bot stopped")


# Singleton instance
_discord_service: DiscordService | None = None


def get_discord_service() -> DiscordService:
    """Get the singleton DiscordService instance."""
    global _discord_service
    if _discord_service is None:
        _discord_service = DiscordService()
    return _discord_service


if __name__ == "__main__":
    # For direct testing
    svc = get_discord_service()
    if svc.start():
        try:
            # Keep main thread alive
            while svc._thread and svc._thread.is_alive():
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            svc.stop()
