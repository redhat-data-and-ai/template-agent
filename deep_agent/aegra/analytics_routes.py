"""Usage analytics for operators.

Gated on the same privileged LDAP role as the eval and admin endpoints.
Everything is derived from tables the platform already writes — see
`deep_agent/src/analytics/queries.py` for why there is no event
pipeline behind this, and why no endpoint here returns message text.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from deep_agent.aegra.eval_routes import _require_developer
from deep_agent.src.analytics import queries
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _window(days: int) -> queries.Window:
    return queries.Window(days=days)


def _unavailable(exc: Exception) -> HTTPException:
    """A dashboard that cannot reach the database should say so.

    Returning zeroes would be worse than an error: a chart of flat
    zeroes is indistinguishable from "nobody used it", which is a
    conclusion somebody will act on.
    """
    logger.warning("Analytics query failed: %s", exc)
    return HTTPException(status_code=503, detail="Analytics unavailable")


@router.get("/summary")
async def analytics_summary(
    days: int = Query(7, ge=1, le=queries.MAX_WINDOW_DAYS),
    _token: str = Depends(_require_developer),
) -> dict[str, Any]:
    """Volume, reliability, latency, tokens and satisfaction."""
    try:
        result = await queries.summary(_window(days))
    except Exception as exc:
        raise _unavailable(exc) from None

    return {
        "window_days": result.window_days,
        "runs": result.runs,
        "threads": result.threads,
        "users": result.users,
        "successes": result.successes,
        "errors": result.errors,
        "conversations": result.conversations,
        "in_flight": result.in_flight,
        "stale": result.stale,
        # Dropped by a restart, awaiting the reaper — not a finished
        # outcome, so it is neither a success nor an error.
        "interrupted": result.interrupted,
        # None, not 0 — see Summary.success_rate.
        "success_rate": result.success_rate,
        "p50_seconds": result.p50_seconds,
        "p95_seconds": result.p95_seconds,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "feedback_up": result.feedback_up,
        "feedback_down": result.feedback_down,
        "satisfaction": result.satisfaction,
    }


@router.get("/timeseries")
async def analytics_timeseries(
    days: int = Query(30, ge=1, le=queries.MAX_WINDOW_DAYS),
    bucket: str = Query("day", pattern="^(hour|day)$"),
    _token: str = Depends(_require_developer),
) -> dict[str, Any]:
    """Runs, errors and p95 over time, with empty buckets filled."""
    try:
        points = await queries.timeseries(_window(days), bucket=bucket)
    except Exception as exc:
        raise _unavailable(exc) from None
    return {"bucket": bucket, "points": points}


@router.get("/models")
async def analytics_models(
    days: int = Query(7, ge=1, le=queries.MAX_WINDOW_DAYS),
    _token: str = Depends(_require_developer),
) -> dict[str, Any]:
    """Per-model volume, tokens, latency and errors."""
    try:
        rows = await queries.by_model(_window(days))
    except Exception as exc:
        raise _unavailable(exc) from None
    return {"window_days": _window(days).days, "models": rows}


@router.get("/errors")
async def analytics_errors(
    days: int = Query(7, ge=1, le=queries.MAX_WINDOW_DAYS),
    _token: str = Depends(_require_developer),
) -> dict[str, Any]:
    """Failures by class.

    Classes, not messages: an exception string routinely contains the
    input that produced it, and a dashboard is the wrong place for
    somebody's document path.
    """
    try:
        rows = await queries.errors(_window(days))
    except Exception as exc:
        raise _unavailable(exc) from None
    return {"window_days": _window(days).days, "classes": rows}


@router.get("/users")
async def analytics_users(
    days: int = Query(7, ge=1, le=queries.MAX_WINDOW_DAYS),
    limit: int = Query(50, ge=1, le=500),
    _token: str = Depends(_require_developer),
) -> dict[str, Any]:
    """Per-user usage, when a deployment has enabled it.

    Returns 404 rather than an empty list when disabled — an empty
    list reads as "nobody used it", which is the wrong answer to a
    question that was never asked.
    """
    enabled = _expose_users()
    try:
        breakdown = await queries.by_user(_window(days), enabled=enabled, limit=limit)
    except Exception as exc:
        raise _unavailable(exc) from None

    if not breakdown.enabled:
        raise HTTPException(status_code=404, detail=breakdown.reason)

    return {
        "window_days": _window(days).days,
        "users": [
            {
                "user_id": row.user_id,
                "runs": row.runs,
                "threads": row.threads,
                "tokens": row.tokens,
                "errors": row.errors,
            }
            for row in breakdown.rows
        ],
    }


def _expose_users() -> bool:
    """`analytics.expose_users` in agent.yaml. Default false."""
    try:
        from deep_agent.src.agent.config import agent_config

        raw = (
            (agent_config.get_raw_section("analytics") or {})
            if hasattr(agent_config, "get_raw_section")
            else {}
        )
        return bool(raw.get("expose_users", False))
    except Exception:
        # The safe default for a privacy switch is off, including when
        # the config cannot be read.
        return False
