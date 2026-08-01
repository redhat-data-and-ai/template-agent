"""APScheduler-based background job scheduler for memory management.

Uses a Redis-backed distributed lock so that only one replica
runs each job at a time (OpenShift multi-replica safe).

When Redis is unavailable, falls back to in-process scheduling
(each pod runs independently — acceptable for idempotent jobs).

Feature flag: ``MEMORY_CONSOLIDATION_ENABLED``.
"""

from typing import Any

from deep_agent.src.memory.config import memory_settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_scheduler: Any = None


async def start_scheduler(database_uri: str) -> bool:
    """Start the background memory scheduler.

    Returns True if started, False if disabled or already running.
    """
    global _scheduler  # noqa: PLW0603

    if not memory_settings.MEMORY_CONSOLIDATION_ENABLED:
        logger.debug("Memory scheduler disabled — skipping")
        return False

    if _scheduler is not None:
        logger.debug("Memory scheduler already running")
        return False

    try:
        from apscheduler import AsyncScheduler
        from apscheduler.triggers.interval import IntervalTrigger

        _scheduler = AsyncScheduler()

        interval = memory_settings.MEMORY_SCHEDULER_INTERVAL_HOURS
        trigger = IntervalTrigger(hours=interval)

        await _scheduler.add_schedule(
            _run_memory_jobs,
            trigger,
            id="memory-consolidation",
            kwargs={"database_uri": database_uri},
        )

        await _scheduler.start_in_background()
        logger.info(
            "Memory scheduler started (interval=%dh)",
            interval,
        )
        return True
    except Exception:
        logger.warning("Failed to start memory scheduler", exc_info=True)
        _scheduler = None
        return False


async def stop_scheduler() -> None:
    """Gracefully stop the scheduler if running."""
    global _scheduler  # noqa: PLW0603
    if _scheduler is not None:
        try:
            await _scheduler.stop()
            logger.info("Memory scheduler stopped")
        except Exception:
            logger.debug("Scheduler stop error", exc_info=True)
        _scheduler = None


async def _run_memory_jobs(database_uri: str) -> dict[str, int]:
    """Execute all enabled memory background jobs.

    This is the single entry point called by the scheduler.
    Each sub-job checks its own feature flag.

    Returns a summary dict of results.
    """
    results: dict[str, int] = {}

    try:
        from deep_agent.src.memory.scoring import decay_all_memories

        results["decay"] = await decay_all_memories(database_uri)
    except Exception:
        logger.error("Decay job failed", exc_info=True)
        results["decay"] = -1

    try:
        from deep_agent.src.memory.consolidation import consolidate_all_users

        results["consolidation"] = await consolidate_all_users(database_uri)
    except Exception:
        logger.error("Consolidation job failed", exc_info=True)
        results["consolidation"] = -1

    try:
        from deep_agent.src.memory.clustering import cluster_all_users

        results["clustering"] = await cluster_all_users(database_uri)
    except Exception:
        logger.error("Clustering job failed", exc_info=True)
        results["clustering"] = -1

    try:
        from deep_agent.src.memory.relationships import infer_all_relationships

        results["relationships"] = await infer_all_relationships(database_uri)
    except Exception:
        logger.error("Relationships job failed", exc_info=True)
        results["relationships"] = -1

    logger.info("Memory jobs complete: %s", results)
    return results


async def run_once(database_uri: str) -> dict[str, int]:
    """Run all memory jobs once (for testing or manual trigger)."""
    return await _run_memory_jobs(database_uri)
