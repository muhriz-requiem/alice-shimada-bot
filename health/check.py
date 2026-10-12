"""
health/check.py
================
Health check endpoint for container orchestration (Kubernetes, Docker Swarm, etc.)
Runs on a separate aiohttp server to avoid blocking the Discord gateway.
"""

from __future__ import annotations

import time

from aiohttp import web

import config
from bot.logging_setup import get_logger
from state import check_redis_health_async

log = get_logger("health")

_START_TIME = time.time()


def _get_uptime() -> float:
    return time.time() - _START_TIME


async def healthz(request: web.Request) -> web.Response:
    """Liveness probe - returns 200 if process is alive."""
    return web.json_response({"status": "alive", "uptime_seconds": round(_get_uptime(), 1)})


async def ready(request: web.Request) -> web.Response:
    """Readiness probe - returns 200 if bot is ready to serve traffic."""
    redis_health = await check_redis_health_async()
    checks = {
        "discord_connected": hasattr(request.app, "bot") and request.app.bot.is_ready(),
        "database": True,  # SQLite is local, always available
        "redis": redis_health,
        "ai_backend": config.AI_BACKEND,
    }

    all_healthy = all(
        v.get("status") != "unhealthy" if isinstance(v, dict) else v
        for v in checks.values()
    )

    status = 200 if all_healthy else 503
    return web.json_response(
        {
            "status": "ready" if all_healthy else "not_ready",
            "checks": checks,
            "uptime_seconds": round(_get_uptime(), 1),
        },
        status=status,
    )


async def metrics(request: web.Request) -> web.Response:
    """Prometheus metrics endpoint."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return web.Response(body=generate_latest(), content_type=CONTENT_TYPE_LATEST)


def create_health_app(bot=None) -> web.Application:
    """Create the aiohttp application for health checks."""
    app = web.Application()
    app["bot"] = bot
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/ready", ready)
    app.router.add_get("/metrics", metrics)
    return app


async def start_health_server(bot=None, host: str = "0.0.0.0", port: int = 8080) -> web.AppRunner:
    """Start the health check HTTP server."""
    app = create_health_app(bot)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    log.info(f"Health check server started on http://{host}:{port}")
    return runner
