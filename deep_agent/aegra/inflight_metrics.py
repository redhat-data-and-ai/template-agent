"""Expose how much work is in flight, so autoscaling can measure it.

Scaling down on a timer (`scaledownPeriod`) is a guess: it has to exceed
the longest run anyone might start, so every burst leaves pods idling
for that long afterwards. Too short and a scale-in severs a running
agent mid-answer.

`aegra_runs_in_flight` replaces the guess with the actual count, so a
scaler can hold a replica exactly as long as there is work and no
longer.

Counted from Postgres rather than per-process state because queued runs
execute on whichever worker claims them: a pod that is idle itself may
still be the last one standing while another pod's run is active. The
gauge is therefore cluster-wide and identical on every replica, which
is what `max()` in a Prometheus query wants.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Seconds between refreshes. Faster than KEDA's polling interval, so the
# scaler never reads a value staler than its own decision window.
REFRESH_INTERVAL_SECONDS = settings.INFLIGHT_METRIC_INTERVAL_SECONDS
# Statuses that mean "work is happening or about to". `pending` counts:
# a queued run with no pod to claim it is exactly the case where
# scaling to zero would strand work.
_ACTIVE_STATUSES = ("pending", "running")

_gauge: Any = None
_pod_gauge: Any = None
_task: asyncio.Task[None] | None = None


def _get_gauge() -> Any:
    """Lazily create the gauge so importing this module is side-effect free."""
    global _gauge  # noqa: PLW0603
    if _gauge is None:
        import prometheus_client

        _gauge = prometheus_client.Gauge(
            "aegra_runs_in_flight",
            "Runs currently pending or running, across all instances",
        )
    return _gauge


def _get_pod_gauge() -> Any:
    """This pod's own share, for the HPA.

    The cluster-wide gauge above is the right input for a scale-to-
    zero decision, and the wrong one for a `Pods` metric with an
    `AverageValue` target: every replica reports the same total, so
    the per-pod average never falls as replicas are added. The HPA
    computes `ceil(n * total / target)`, which grows with `n`, and
    the fleet climbed to maxReplicas whenever the total crossed the
    target and stayed there.
    """
    global _pod_gauge  # noqa: PLW0603
    if _pod_gauge is None:
        import prometheus_client

        _pod_gauge = prometheus_client.Gauge(
            "aegra_runs_in_flight_pod",
            "Runs this instance is executing right now",
        )
    return _pod_gauge


def count_pod_active_runs() -> int:
    """Runs this process is executing, from Aegra's in-memory registry."""
    try:
        from deep_agent.aegra.lifecycle import _get_active_runs

        return len(_get_active_runs() or {})
    except Exception:
        logger.debug("Could not read this pod's active runs", exc_info=True)
        return 0


async def count_active_runs() -> int:
    """Runs pending or running right now.

    Runs left behind by a dead pod keep their `pending`/`running`
    status for ever — `queries.py` calls those `stale` and reports
    them separately. Counted here they held the gauge above zero, and
    a cluster that can never read zero never scales to zero. Same age
    bound, so both agree on what is still in flight.
    """
    from datetime import datetime, timedelta, timezone

    from aegra_api.core.orm import Run as RunORM
    from aegra_api.core.orm import _get_session_maker
    from sqlalchemy import func, select

    cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=settings.ANALYTICS_STALE_RUN_SECONDS
    )
    maker = _get_session_maker()
    async with maker() as session:
        result = await session.execute(
            select(func.count())
            .select_from(RunORM)
            .where(RunORM.status.in_(_ACTIVE_STATUSES))
            .where(RunORM.updated_at >= cutoff)
        )
        return int(result.scalar() or 0)


async def _refresh_loop() -> None:
    """Keep the gauge current until cancelled."""
    gauge = _get_gauge()
    pod_gauge = _get_pod_gauge()
    while True:
        try:
            gauge.set(await count_active_runs())
            pod_gauge.set(count_pod_active_runs())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A scrape failure must not kill the loop, and must not
            # report zero — a false zero lets the scaler kill live work.
            logger.warning("Could not refresh in-flight gauge: %s", exc)
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)


def start_inflight_metrics() -> None:
    """Begin publishing the gauge. Safe to call more than once."""
    global _task  # noqa: PLW0603
    if _task is not None and not _task.done():
        return

    _get_gauge()
    _task = asyncio.create_task(_refresh_loop())
    logger.info("Publishing aegra_runs_in_flight every %ds", REFRESH_INTERVAL_SECONDS)


async def stop_inflight_metrics() -> None:
    """Stop publishing, for a clean shutdown."""
    global _task  # noqa: PLW0603
    if _task is None:
        return
    _task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await _task
    _task = None
