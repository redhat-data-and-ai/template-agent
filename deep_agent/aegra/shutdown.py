"""Shutdown orchestrator — coordinated teardown on SIGTERM.

Mirrors ``startup.py``: idempotent orchestrator, individual step
functions, structured logging, defensive error handling.

Three independent paths trigger shutdown (belt-and-suspenders for
uncertain Aegra lifespan behavior):

1. FastAPI lifespan exit (feedback.py)
2. SIGTERM/SIGINT signal handler (registered at startup)
3. atexit callback (fallback for normal process exit)

All three call ``run_shutdown()`` which is idempotent — the second
call returns immediately.

Shutdown sequence (within ``terminationGracePeriodSeconds: 60``):

    1. Set ``_shutting_down`` flag → health probes return 503
    2. Drain period — in-flight requests finish
    3. Flush and stop Langfuse
    4. Stop memory scheduler
    5. Clear graph cache
    6. Close Redis
"""

import asyncio
import os
import time
from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_shutting_down = False
_shutdown_complete = False

SHUTDOWN_DRAIN_SECONDS = int(os.environ.get("SHUTDOWN_DRAIN_SECONDS", "15"))
SHUTDOWN_LANGFUSE_TIMEOUT_SECONDS = int(
    os.environ.get("SHUTDOWN_LANGFUSE_TIMEOUT_SECONDS", "5")
)
SHUTDOWN_SCHEDULER_TIMEOUT_SECONDS = int(
    os.environ.get("SHUTDOWN_SCHEDULER_TIMEOUT_SECONDS", "10")
)


def is_shutting_down() -> bool:
    """Return True once shutdown has been initiated."""
    return _shutting_down


async def run_shutdown() -> dict[str, str]:
    """Execute the shutdown sequence. Returns a status dict.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _shutting_down, _shutdown_complete  # noqa: PLW0603

    if _shutting_down:
        logger.debug("Shutdown already initiated — skipping")
        return {"status": "already_complete"}

    _shutting_down = True
    t0 = time.monotonic()
    results: dict[str, str] = {}

    logger.info("Shutdown initiated")

    for key, step in [
        ("drain", _drain),
        ("langfuse", _shutdown_langfuse),
        ("scheduler", _stop_scheduler),
        ("graph_cache", _clear_graph_cache),
        ("redis", _close_redis),
    ]:
        try:
            result = step()
            if asyncio.iscoroutine(result):
                result = await result
            results[key] = result
        except Exception as exc:
            logger.warning("Shutdown step '%s' failed: %s", key, exc)
            results[key] = f"error: {exc}"

    _shutdown_complete = True
    elapsed = round((time.monotonic() - t0) * 1000, 1)

    logger.info("Shutdown complete in %.1fms: %s", elapsed, results)
    return results


def run_shutdown_sync() -> None:
    """Synchronous fallback for atexit. Bootstraps a loop if needed."""
    if _shutdown_complete or _shutting_down:
        return
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(run_shutdown())
            return
        loop.run_until_complete(run_shutdown())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_shutdown())
        finally:
            loop.close()


def register_signal_handlers() -> None:
    """Install SIGTERM/SIGINT handlers on the running event loop.

    Must be called from the main thread while a loop is running.
    Uses ``loop.add_signal_handler`` (Unix-only — fine for OpenShift).
    """
    import signal

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("No running event loop — signal handlers not registered")
        return

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal, sig, loop)

    logger.info("Shutdown signal handlers registered (SIGTERM, SIGINT)")


def _handle_signal(signum: int, loop: asyncio.AbstractEventLoop) -> None:
    global _shutting_down  # noqa: PLW0603
    _shutting_down = True
    logger.info("Signal %d received — scheduling shutdown", signum)
    loop.create_task(run_shutdown())


# -- Individual shutdown steps -----------------------------------------------


async def _drain() -> str:
    if SHUTDOWN_DRAIN_SECONDS <= 0:
        return "skipped: drain disabled"
    logger.info("Draining for %ds", SHUTDOWN_DRAIN_SECONDS)
    await asyncio.sleep(SHUTDOWN_DRAIN_SECONDS)
    return "ok"


async def _shutdown_langfuse() -> str:
    try:
        from deep_agent.aegra.telemetry import get_langfuse_client

        client = get_langfuse_client()
        if client is None:
            return "skipped: not configured"

        await asyncio.wait_for(
            asyncio.to_thread(_langfuse_shutdown_blocking, client),
            timeout=SHUTDOWN_LANGFUSE_TIMEOUT_SECONDS,
        )
        return "ok"
    except asyncio.TimeoutError:
        logger.warning(
            "Langfuse shutdown timed out after %ds", SHUTDOWN_LANGFUSE_TIMEOUT_SECONDS
        )
        return "timeout"
    except Exception as exc:
        logger.warning("Langfuse shutdown failed: %s", exc)
        return f"error: {exc}"


def _langfuse_shutdown_blocking(client: Any) -> None:
    """Run the sync Langfuse shutdown in a thread."""
    if hasattr(client, "shutdown"):
        client.shutdown()
    elif hasattr(client, "flush"):
        client.flush()


async def _stop_scheduler() -> str:
    try:
        from deep_agent.src.memory.scheduler import stop_scheduler

        await asyncio.wait_for(
            stop_scheduler(),
            timeout=SHUTDOWN_SCHEDULER_TIMEOUT_SECONDS,
        )
        return "ok"
    except asyncio.TimeoutError:
        logger.warning(
            "Scheduler stop timed out after %ds", SHUTDOWN_SCHEDULER_TIMEOUT_SECONDS
        )
        return "timeout"
    except Exception as exc:
        logger.warning("Scheduler stop failed: %s", exc)
        return f"error: {exc}"


def _clear_graph_cache() -> str:
    try:
        from deep_agent.aegra.graph import _graph_cache, _graph_cache_ts

        count = len(_graph_cache)
        _graph_cache.clear()
        _graph_cache_ts.clear()
        if count > 0:
            logger.info("Cleared %d cached graph(s)", count)
        return "ok"
    except Exception as exc:
        logger.warning("Graph cache clear failed: %s", exc)
        return f"error: {exc}"


def _close_redis() -> str:
    try:
        from deep_agent.aegra.redis import close_redis_client

        close_redis_client()
        return "ok"
    except Exception as exc:
        logger.warning("Redis close failed: %s", exc)
        return f"error: {exc}"
