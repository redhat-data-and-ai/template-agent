"""Shutdown orchestrator — coordinated teardown on SIGTERM.

Mirrors ``startup.py``: idempotent orchestrator, individual step
functions, structured logging, defensive error handling.

Two independent paths trigger shutdown:

1. ``atexit`` callback (registered at import time from ``feedback.py``)
   — fires reliably when uvicorn handles SIGTERM and exits normally.
   Runs a synchronous cleanup (Langfuse flush, Redis close, graph
   cache clear). No event loop needed.

2. ``loop.add_signal_handler`` (registered on first graph request via
   ``startup.py``) — overrides uvicorn's handler, runs the full async
   shutdown with drain period. Only active after the first graph
   request, but that's when there's actually work to drain.

Aegra strips our custom app's lifespan and middleware, so neither
ASGI lifespan nor middleware-based registration works. The atexit
path is guaranteed because Aegra always imports ``feedback.py``.

Both paths call idempotent cleanup — the second call is a no-op.

Shutdown sequence (within ``terminationGracePeriodSeconds: 60``):

    1. Set ``_shutting_down`` flag → health probes return 503
    2. Drain period — in-flight requests finish (async path only)
    3. Flush and stop Langfuse
    4. Stop memory scheduler (async path only)
    5. Clear graph cache
    6. Close Redis
"""

import asyncio
import os
import signal
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


# -- Primary path: atexit (sync) ---------------------------------------------


def register_atexit() -> None:
    """Register the sync shutdown as an atexit callback.

    Called at import time from ``feedback.py``. Unlike signal handlers,
    atexit callbacks are not overwritten by uvicorn.
    """
    import atexit

    atexit.register(run_shutdown_sync)
    logger.info("Shutdown atexit handler registered")


def run_shutdown_sync() -> None:
    """Synchronous shutdown — runs at process exit via atexit.

    Handles cleanup that doesn't need an event loop: Langfuse flush,
    Redis close, graph cache clear. Skips drain and async scheduler
    stop (those only run in the async path).
    """
    global _shutting_down, _shutdown_complete  # noqa: PLW0603

    if _shutdown_complete:
        return
    if _shutting_down:
        logger.debug("Async shutdown already ran — sync cleanup skipped")
        _shutdown_complete = True
        return

    _shutting_down = True
    t0 = time.monotonic()
    results: dict[str, str] = {}

    import sys

    print("[shutdown] Graceful shutdown started", file=sys.stderr, flush=True)
    logger.info("Sync shutdown initiated (atexit)")

    for key, step in [
        ("langfuse", _shutdown_langfuse_sync),
        ("graph_cache", _clear_graph_cache),
        ("redis", _close_redis),
    ]:
        try:
            results[key] = step()
        except Exception as exc:
            logger.warning("Shutdown step '%s' failed: %s", key, exc)
            results[key] = f"error: {exc}"

    _shutdown_complete = True
    elapsed = round((time.monotonic() - t0) * 1000, 1)
    print(
        f"[shutdown] Graceful shutdown complete in {elapsed}ms: {results}",
        file=sys.stderr,
        flush=True,
    )


# -- Secondary path: signal handler (async) ----------------------------------


def register_signal_handlers() -> None:
    """Install loop-aware SIGTERM/SIGINT handlers.

    Uses ``loop.add_signal_handler`` which overrides uvicorn's handler.
    Must be called from inside a running event loop. Called from
    ``startup.py`` after the first graph request.
    """
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
    logger.info("Signal %d received — scheduling async shutdown", signum)
    loop.create_task(run_shutdown())


_async_shutdown_started = False


async def run_shutdown() -> dict[str, str]:
    """Full async shutdown with drain period.

    Only runs when signal handlers were registered (after first graph
    request). Safe to call multiple times — subsequent calls are no-ops.
    """
    global _shutting_down, _shutdown_complete, _async_shutdown_started  # noqa: PLW0603

    if _shutdown_complete or _async_shutdown_started:
        return {"status": "already_complete"}

    _async_shutdown_started = True
    _shutting_down = True

    t0 = time.monotonic()
    results: dict[str, str] = {}

    logger.info("Async shutdown initiated")

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

    logger.info("Async shutdown complete in %.1fms: %s", elapsed, results)
    return results


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


def _shutdown_langfuse_sync() -> str:
    """Sync Langfuse flush for atexit path.

    Only flushes if a client was already initialized — avoids creating
    a new client during interpreter shutdown (which triggers
    ``RuntimeError: cannot schedule new futures``).
    """
    try:
        from deep_agent.aegra.telemetry import _langfuse_configured

        if not _langfuse_configured():
            return "skipped: not configured"

        from langfuse import get_client

        client = get_client()
        _langfuse_shutdown_blocking(client)
        return "ok"
    except Exception as exc:
        return f"skipped: {exc}"


def _langfuse_shutdown_blocking(client: Any) -> None:
    """Run the sync Langfuse shutdown."""
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
