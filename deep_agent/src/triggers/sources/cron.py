"""Cron trigger source — lightweight cron scheduler using croniter.

Parses cron expressions from ``CronTriggerConfig``, computes the next
fire time for each job, and sleeps until it arrives.  Each firing puts
a ``TriggerEvent`` onto an internal ``asyncio.Queue`` consumed through
the async-iterator protocol.

Uses ``apscheduler.triggers.cron.CronTrigger`` only for parsing the
crontab expression — scheduling is done with plain ``asyncio.sleep``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from deep_agent.src.triggers.config import CronJobConfig, CronTriggerConfig
from deep_agent.src.triggers.sources.protocol import TriggerEvent
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _parse_cron_fields(schedule: str) -> dict[str, str] | None:
    """Parse a 5-field crontab string into CronTrigger kwargs. Returns None on error."""
    parts = schedule.strip().split()
    if len(parts) != 5:
        return None
    return {
        "minute": parts[0],
        "hour": parts[1],
        "day": parts[2],
        "month": parts[3],
        "day_of_week": parts[4],
    }


class CronTriggerSource:
    """Fires ``TriggerEvent`` instances on cron schedules.

    Implements the ``TriggerSource`` protocol.  ``start()`` launches a
    background task per job that sleeps until the next fire time, emits
    the event, and loops.  ``stop()`` cancels all tasks.
    """

    def __init__(self, config: CronTriggerConfig) -> None:
        """Initialize the cron trigger source with the given configuration."""
        self._config = config
        self._queue: asyncio.Queue[TriggerEvent] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        """Start background tasks for each configured cron job."""
        for job_cfg in self._config.jobs:
            fields = _parse_cron_fields(job_cfg.schedule)
            if fields is None:
                logger.warning(
                    "invalid cron schedule, skipping job",
                    job_name=job_cfg.name,
                    schedule=job_cfg.schedule,
                )
                continue

            task = asyncio.create_task(self._run_job(job_cfg, fields))
            self._tasks.append(task)
            logger.info(
                "cron job scheduled",
                job_name=job_cfg.name,
                schedule=job_cfg.schedule,
            )

        logger.info("cron trigger source started", job_count=len(self._tasks))

    async def _run_job(self, job_cfg: CronJobConfig, fields: dict[str, Any]) -> None:
        """Background loop for a single cron job."""
        from apscheduler.triggers.cron import CronTrigger

        trigger = CronTrigger(**fields)

        while True:
            now = datetime.now(timezone.utc)
            next_fire = trigger.next()
            if next_fire is None:
                return
            delay = (next_fire - now).total_seconds()
            if delay > 0:
                await asyncio.sleep(delay)

            event = TriggerEvent(
                name=job_cfg.name,
                payload=dict(job_cfg.payload),
                source="cron",
                metadata={"schedule": job_cfg.schedule},
            )
            await self._queue.put(event)
            logger.debug("cron event fired", job_name=job_cfg.name)

    async def stop(self) -> None:
        """Cancel all running cron job tasks."""
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        logger.info("cron trigger source stopped")

    def __aiter__(self) -> AsyncIterator[TriggerEvent]:
        """Return the async iterator."""
        return self

    async def __anext__(self) -> TriggerEvent:
        """Return the next cron trigger event."""
        return await self._queue.get()
