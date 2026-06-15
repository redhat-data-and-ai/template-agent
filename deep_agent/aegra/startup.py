"""Startup orchestrator — coordinated initialization on process boot.

Runs once when the agent process starts. Ensures all subsystems
are initialized in the correct order before the server accepts
traffic.

Startup sequence:
    1. Validate configuration
    2. Ensure database tables exist
    3. Warm caches (if enabled)
    4. Start memory scheduler (if enabled)
    5. Set up Langfuse tracing (if configured)
    6. Log readiness

This module is idempotent — calling ``run_startup()`` multiple
times is safe (each step guards against double-init).
"""

import time

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_startup_complete = False


async def run_startup() -> dict[str, str]:
    """Execute the startup sequence. Returns a status dict.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _startup_complete  # noqa: PLW0603

    if _startup_complete:
        logger.debug("Startup already complete — skipping")
        return {"status": "already_complete"}

    t0 = time.monotonic()
    results: dict[str, str] = {}

    results["config"] = await _validate_config()
    results["database"] = await _ensure_database()
    results["cache"] = await _warm_caches()
    results["scheduler"] = await _start_scheduler()
    results["telemetry"] = _setup_telemetry()

    _upgrade_signal_handlers()

    elapsed = round((time.monotonic() - t0) * 1000, 1)
    _startup_complete = True

    logger.info(
        "Startup complete in %.1fms: %s",
        elapsed,
        results,
    )
    return results


def _upgrade_signal_handlers() -> None:
    """Upgrade to loop-aware signal handlers for async drain."""
    try:
        from deep_agent.aegra.shutdown import register_signal_handlers

        register_signal_handlers()
    except Exception:
        logger.warning("Failed to register signal handlers", exc_info=True)


async def _validate_config() -> str:
    """Validate core settings."""
    try:
        from deep_agent.src.settings import settings, validate_config

        validate_config(settings)
        return "ok"
    except Exception as exc:
        logger.warning("Config validation warning: %s", exc)
        return f"warning: {exc}"


async def _ensure_database() -> str:
    """Create personalization and feedback tables if they don't exist."""
    try:
        from deep_agent.src.feedback.repository import FeedbackRepository
        from deep_agent.src.personalization.repository import (
            PersonalizationRepository,
        )
        from deep_agent.src.settings import settings

        if not settings.database_uri:
            return "skipped: no database_uri"

        repo = PersonalizationRepository(settings.database_uri)
        await repo.ensure_tables()
        feedback_repo = FeedbackRepository(settings.database_uri)
        await feedback_repo.ensure_table()
        return "ok"
    except Exception as exc:
        logger.error("Database setup failed: %s", exc)
        return f"error: {exc}"


async def _warm_caches() -> str:
    """Pre-populate caches if caching is enabled."""
    try:
        from deep_agent.src.cache.config import cache_settings

        if not cache_settings.CACHE_ENABLED:
            return "skipped: caching disabled"

        from deep_agent.src.cache.warming import warm_caches

        warm_caches()
        return "ok"
    except Exception as exc:
        logger.warning("Cache warming failed: %s", exc)
        return f"warning: {exc}"


async def _start_scheduler() -> str:
    """Start background memory scheduler if enabled."""
    try:
        from deep_agent.src.memory.config import memory_settings

        if not memory_settings.MEMORY_CONSOLIDATION_ENABLED:
            return "skipped: memory consolidation disabled"

        from deep_agent.src.memory.scheduler import start_scheduler
        from deep_agent.src.settings import settings

        started = await start_scheduler(settings.database_uri)
        return "ok" if started else "skipped: already running"
    except Exception as exc:
        logger.warning("Scheduler start failed: %s", exc)
        return f"warning: {exc}"


def _setup_telemetry() -> str:
    """Register Langfuse tracing if credentials are configured."""
    try:
        from deep_agent.aegra.telemetry import setup_langfuse_tracing

        setup_langfuse_tracing()
        return "ok"
    except Exception as exc:
        logger.warning("Telemetry setup failed: %s", exc)
        return f"warning: {exc}"


def is_ready() -> bool:
    """Return True if startup has completed."""
    return _startup_complete
