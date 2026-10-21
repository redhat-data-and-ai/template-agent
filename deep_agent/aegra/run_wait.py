"""Make `/runs/wait` notice a finished run promptly.

Aegra's ``WorkerExecutor.wait_for_completion`` checks a Redis done-key
and then sleeps a flat two seconds:

    while loop.time() < deadline:
        if await client.exists(done_key):
            return
        ...
        await asyncio.sleep(2.0)

It is a poll, not a subscription, so the wait is not bounded by how
long the run takes — it is bounded by the next poll boundary. A run
that finishes in 300ms is reported to the caller at 2.0s.

That is measurable from outside and was, before anyone went looking
for it: turn latencies clustered at 2.09s, 4.10s and 6.11s, spaced
2.01s apart. None of those numbers had anything to do with the model.
Shrinking the prompt from 119,769 tokens to 7,965 did not move them,
which is what made the polling visible — a tenfold cut in work with
no change in wall clock is not a model being slow.

The fix is to poll quickly at first and back off, rather than poll
slowly throughout. Most runs finish in the first second, so the early
checks catch them; a long run settles onto the original cadence and
costs no more than it did before. Average dead time drops from about
a second to a few tens of milliseconds.

A subscription would be better still and is the right upstream
change. It needs the worker to publish, which is the other side of a
vendored package, so this patches the half that can be reached from
here — the same arrangement as the pool liveness check in
``startup.py``, and for the same reason.
"""

from __future__ import annotations

import asyncio
from typing import Any

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_patched = False


def _backoff_schedule() -> list[float]:
    """Sleep intervals, doubling from the floor up to the ceiling.

    Returned as a list so the growth is visible at a glance rather
    than hidden in arithmetic: 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.0 …
    """
    delays: list[float] = []
    delay = settings.RUN_WAIT_MIN_POLL_SECONDS
    while delay < settings.RUN_WAIT_MAX_POLL_SECONDS:
        delays.append(delay)
        delay *= 2
    delays.append(settings.RUN_WAIT_MAX_POLL_SECONDS)
    return delays


def install_fast_run_wait() -> str:
    """Replace the flat 2s poll with an adaptive one. Idempotent."""
    global _patched  # noqa: PLW0603
    if _patched:
        return "ok: already patched"
    if not settings.RUN_WAIT_FAST_POLL:
        return "skipped: disabled"

    try:
        from aegra_api.core.redis_manager import (
            redis_manager,
        )
        from aegra_api.services.worker_executor import (
            WorkerExecutor,
            _is_run_terminal,
        )
        from aegra_api.settings import (
            settings as aegra_settings,
        )
        from redis.exceptions import RedisError
    except Exception as exc:
        # An upstream rename lands here rather than at the first
        # request. Losing the optimisation is acceptable; failing to
        # start over it is not.
        logger.warning("Fast run-wait not installed: %s", exc)
        return f"skipped: {exc}"

    async def wait_for_completion(
        self: Any, run_id: str, *, timeout: float = 300.0
    ) -> None:
        """Wait for a run to finish, checking often at first."""
        done_key = f"{aegra_settings.redis.REDIS_CHANNEL_PREFIX}done:{run_id}"
        client = redis_manager.get_client()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        delays = _backoff_schedule()
        index = 0
        next_db_check = loop.time() + settings.RUN_WAIT_DB_CHECK_SECONDS

        while loop.time() < deadline:
            try:
                if await client.exists(done_key):
                    return
            except RedisError:
                # Redis being unreachable must not turn a finished run
                # into a timeout — the database check below is the
                # authority, and it is why this is a fallback rather
                # than a failure path.
                pass

            # The database fallback is rate-limited by *time*, not by
            # a count of polls. Upstream checks every second poll,
            # which at its 2s interval meant every 4s; keeping that
            # ratio against a 50ms interval would mean every 100ms,
            # and a query per waiting request twenty times a second
            # is how a latency fix becomes a database problem. It
            # measurably was: the first version of this patch made
            # turns slower, not faster.
            now = loop.time()
            if now >= next_db_check:
                next_db_check = now + settings.RUN_WAIT_DB_CHECK_SECONDS
                if await _is_run_terminal(run_id):
                    return

            await asyncio.sleep(delays[index])
            if index < len(delays) - 1:
                index += 1

        raise TimeoutError(f"Run {run_id} did not complete within {timeout}s")

    WorkerExecutor.wait_for_completion = wait_for_completion
    _patched = True
    logger.info(
        "Fast run-wait installed: first poll after %.0fms, backing off to %.1fs",
        settings.RUN_WAIT_MIN_POLL_SECONDS * 1000,
        settings.RUN_WAIT_MAX_POLL_SECONDS,
    )
    return "ok"
