"""Per-run token and model usage, recorded where the tokens are spent.

Analytics used to derive tokens and model names from
``runs.output -> messages[]``. That column is empty for every run the
UI actually makes: Aegra persists a final output only for the default
``values`` stream mode, and the UI streams ``updates``/``messages``/
``events``. In the live database that left 354 of 451 runs with no
messages at all, so the Usage tab attributed 79% of runs to model
"unknown" with zero tokens — wrong by roughly a factor of five on the
two figures anyone reads that page for.

So usage is recorded at the one place that sees every call regardless
of stream mode: the LangChain callback that fires when a model
returns. It is still not a second *source* of truth in the sense
``queries`` warns about — nothing else counts tokens per run, so there
is nothing for it to disagree with.

Two deliberate choices:

- It merges into ``runs.execution_params``, the jsonb column
  ``lifecycle.py`` already merges shutdown provenance into. No
  migration, no new table, and a template repo does not get to hand
  every fork a schema change.
- Totals are added **in SQL**, never read-modify-written in Python. A
  run's subagents call models concurrently; adding in the handler
  would silently drop increments.

Per-model totals are kept as a map rather than a single winning name.
The previous ``MAX(model_name)`` picked whichever name sorted last,
which is not a fact about anything.
"""

from __future__ import annotations

import asyncio
from typing import Any

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# One statement, so concurrent subagent calls serialise on the row lock
# instead of racing. Every `execution_params` on the right-hand side is
# the pre-update value.
_RECORD_SQL = """
UPDATE runs
SET execution_params =
    COALESCE(execution_params, '{}'::jsonb)
    || jsonb_build_object('usage',
        COALESCE(execution_params -> 'usage', '{}'::jsonb)
        || jsonb_build_object(
            'input_tokens',
                COALESCE((execution_params #>> '{usage,input_tokens}')::bigint, 0)
                + %(input_tokens)s,
            'output_tokens',
                COALESCE((execution_params #>> '{usage,output_tokens}')::bigint, 0)
                + %(output_tokens)s,
            'calls',
                COALESCE((execution_params #>> '{usage,calls}')::bigint, 0) + 1,
            'models',
                COALESCE(execution_params #> '{usage,models}', '{}'::jsonb)
                || jsonb_build_object(%(model)s, jsonb_build_object(
                    'input_tokens',
                        COALESCE((execution_params #> '{usage,models}'
                                  -> %(model)s ->> 'input_tokens')::bigint, 0)
                        + %(input_tokens)s,
                    'output_tokens',
                        COALESCE((execution_params #> '{usage,models}'
                                  -> %(model)s ->> 'output_tokens')::bigint, 0)
                        + %(output_tokens)s,
                    'calls',
                        COALESCE((execution_params #> '{usage,models}'
                                  -> %(model)s ->> 'calls')::bigint, 0) + 1
                ))
        ))
WHERE run_id = %(run_id)s
"""

_pool: Any = None
_pool_lock = asyncio.Lock()

# Recording is best-effort by construction — a dashboard figure must
# never be the reason a user's answer fails — so repeated failures are
# only visible if we count them.
_failures = 0
_FAILURE_ALERT_THRESHOLD = 5


async def _get_pool() -> Any:
    """A small shared pool: this runs once per LLM call, not per run."""
    global _pool  # noqa: PLW0603
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:
            from psycopg_pool import AsyncConnectionPool

            pool = AsyncConnectionPool(
                settings.database_uri, min_size=1, max_size=4, open=False
            )
            await pool.open()
            _pool = pool
    return _pool


async def close_pool() -> None:
    """Release the pool on shutdown."""
    global _pool  # noqa: PLW0603
    if _pool is not None:
        await _pool.close()
        _pool = None


async def record_llm_usage(
    run_id: str,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Add one model call's tokens to its run.

    Calls that report no tokens are still recorded: a model that
    answered is a fact about the run even when the provider declined
    to say what it cost, and dropping it makes `calls` disagree with
    reality.
    """
    global _failures  # noqa: PLW0603
    if not run_id:
        return
    params = {
        "run_id": run_id,
        # 'unknown' rather than NULL: a jsonb object key cannot be
        # null, and an unnamed model is a thing operators should see
        # rather than a row that quietly vanishes.
        "model": model or "unknown",
        "input_tokens": max(0, int(input_tokens or 0)),
        "output_tokens": max(0, int(output_tokens or 0)),
    }
    try:
        pool = await _get_pool()
        async with pool.connection() as conn:
            await conn.execute(_RECORD_SQL, params)
        _failures = 0
    except Exception:
        _failures += 1
        if _failures >= _FAILURE_ALERT_THRESHOLD:
            logger.error("usage_recording_degraded", consecutive_failures=_failures)
        else:
            logger.debug("Failed to record usage for run %s", run_id, exc_info=True)
