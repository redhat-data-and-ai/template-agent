"""Health check endpoint for OpenShift probes.

Provides ``/health``, ``/healthz``, ``/readyz``, and ``/livez``
endpoints consumed by Kubernetes/OpenShift liveness, readiness,
and startup probes.

Response format::

    {
      "status": "healthy" | "degraded" | "unhealthy",
      "version": "0.1.0",
      "uptime_seconds": 1234.5,
      "checks": {
        "database": {"status": "ok", "latency_ms": 5.2},
        "redis": {"status": "ok"},
        "config": {"status": "ok"}
      }
    }

The health module can be used standalone (imported and called)
or wired as ASGI middleware for ``aegra serve``.
"""

import asyncio
import time
from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_start_time = time.monotonic()


async def check_database() -> dict[str, Any]:
    """Ping the Postgres database and measure latency."""
    try:
        from deep_agent.src.settings import settings

        if not settings.database_uri:
            return {"status": "skipped", "reason": "no database_uri configured"}

        import psycopg

        t0 = time.monotonic()
        async with await psycopg.AsyncConnection.connect(settings.database_uri) as conn:
            await conn.execute("SELECT 1")
        latency_ms = (time.monotonic() - t0) * 1000

        return {"status": "ok", "latency_ms": round(latency_ms, 1)}
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:200]}


async def check_redis() -> dict[str, Any]:
    """Ping the Redis server."""
    try:
        from deep_agent.aegra.redis import get_redis_client

        client = get_redis_client()
        if client is None:
            return {"status": "skipped", "reason": "redis not configured"}

        t0 = time.monotonic()
        pong = await asyncio.to_thread(client.ping)
        latency_ms = (time.monotonic() - t0) * 1000

        return {
            "status": "ok" if pong else "error",
            "latency_ms": round(latency_ms, 1),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:200]}


def check_config() -> dict[str, Any]:
    """Validate that core configuration is present."""
    try:
        from deep_agent.src.settings import settings

        issues: list[str] = []
        if not settings.database_uri:
            issues.append("database_uri not set")
        if settings.AGENT_PORT < 1 or settings.AGENT_PORT > 65535:
            issues.append(f"invalid port: {settings.AGENT_PORT}")

        if issues:
            return {"status": "warning", "issues": issues}
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:200]}


def check_cache() -> dict[str, Any]:
    """Return cache statistics if caching is enabled."""
    try:
        from deep_agent.src.cache.metrics import get_stats

        stats = get_stats()
        return {"status": "ok", **stats}
    except Exception:
        return {"status": "skipped"}


def check_otel() -> dict[str, Any]:
    """Return OpenTelemetry initialization status."""
    try:
        from deep_agent.aegra.otel import (
            _initialized,
            _otel_enabled,
            _resolve_config,
            is_tracing_enabled,
        )

        if not _initialized:
            return {"status": "not_initialized"}

        _, endpoint, _, _, _ = _resolve_config()

        # Try to get OTEL SDK version
        sdk_version = None
        try:
            import opentelemetry

            sdk_version = opentelemetry.__version__
        except Exception:
            pass

        return {
            "status": "ok",
            "initialized": _initialized,
            "enabled": _otel_enabled,
            "tracing_active": is_tracing_enabled(),
            "endpoint": endpoint if _otel_enabled else None,
            "sdk_version": sdk_version,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:200]}


async def get_health_status() -> dict[str, Any]:
    """Run all health checks and return a combined status."""
    checks: dict[str, Any] = {}

    checks["config"] = check_config()
    checks["database"] = await check_database()
    checks["redis"] = await check_redis()
    checks["cache"] = check_cache()
    checks["otel"] = check_otel()

    all_statuses = [c.get("status", "unknown") for c in checks.values()]

    if any(s == "error" for s in all_statuses):
        overall = "unhealthy"
    elif any(s == "warning" for s in all_statuses):
        overall = "degraded"
    else:
        overall = "healthy"

    return {
        "status": overall,
        "version": "0.1.0",
        "uptime_seconds": round(time.monotonic() - _start_time, 1),
        "checks": checks,
    }


async def health_response() -> tuple[int, dict[str, Any]]:
    """Return (status_code, body) for the health endpoint."""
    from deep_agent.aegra.shutdown import is_shutting_down

    if is_shutting_down():
        return 503, {
            "status": "shutting_down",
            "version": "0.1.0",
            "uptime_seconds": round(time.monotonic() - _start_time, 1),
        }

    result = await get_health_status()
    code = 200 if result["status"] in ("healthy", "degraded") else 503
    return code, result


HEALTH_PATHS = frozenset({"/health", "/healthz", "/readyz", "/livez"})
