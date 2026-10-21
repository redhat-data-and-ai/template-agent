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
# Every parameter is cast explicitly. `model` is used as a
# jsonb_build_object *key*, where Postgres cannot infer a type from
# context and answers "could not determine data type of parameter
# $3" — which `record_llm_usage` caught, logged at debug and
# counted as a failure, so every write since this feature shipped
# failed silently and the Usage tab reported zero tokens as fact.
_RECORD_SQL = """
UPDATE runs
SET execution_params =
    COALESCE(execution_params, '{}'::jsonb)
    || jsonb_build_object('usage',
        COALESCE(execution_params -> 'usage', '{}'::jsonb)
        || jsonb_build_object(
            'input_tokens',
                COALESCE((execution_params #>> '{usage,input_tokens}')::bigint, 0)
                + %(input_tokens)s::bigint,
            'output_tokens',
                COALESCE((execution_params #>> '{usage,output_tokens}')::bigint, 0)
                + %(output_tokens)s::bigint,
            'cached_tokens',
                COALESCE((execution_params #>> '{usage,cached_tokens}')::bigint, 0)
                + %(cached_tokens)s::bigint,
            'calls',
                COALESCE((execution_params #>> '{usage,calls}')::bigint, 0) + 1,
            'models',
                COALESCE(execution_params #> '{usage,models}', '{}'::jsonb)
                || jsonb_build_object(%(model)s::text, jsonb_build_object(
                    'input_tokens',
                        COALESCE((execution_params #> '{usage,models}'
                                  -> %(model)s::text ->> 'input_tokens')::bigint, 0)
                        + %(input_tokens)s::bigint,
                    'output_tokens',
                        COALESCE((execution_params #> '{usage,models}'
                                  -> %(model)s::text ->> 'output_tokens')::bigint, 0)
                        + %(output_tokens)s::bigint,
                    'cached_tokens',
                        COALESCE((execution_params #> '{usage,models}'
                                  -> %(model)s::text ->> 'cached_tokens')::bigint, 0)
                        + %(cached_tokens)s::bigint,
                    'calls',
                        COALESCE((execution_params #> '{usage,models}'
                                  -> %(model)s::text ->> 'calls')::bigint, 0) + 1
                ))
        ))
WHERE run_id = %(run_id)s::text
"""

# Recording is best-effort by construction — a dashboard figure must
# never be the reason a user's answer fails — so repeated failures are
# only visible if we count them.
_failures = 0
_FAILURE_ALERT_THRESHOLD = settings.USAGE_FAILURE_ALERT_THRESHOLD
# Accounting must never be the slowest thing in a turn.
_ACQUIRE_TIMEOUT_SECONDS = 2.0
_WRITE_TIMEOUT_SECONDS = 5.0


async def _get_pool() -> Any:
    """The process-wide pool.

    This used to be four connections of its own, held per pod so that
    a best-effort telemetry write would never wait. That is a poor
    trade once the fleet's connection budget is the binding
    constraint: recording usage is the least important thing the pod
    does, and it should queue behind real work rather than reserve
    capacity ahead of it.
    """
    from deep_agent.src.db.pool import get_pool as get_shared_pool

    return await get_shared_pool()


async def close_pool() -> None:
    """Retained for callers that close analytics explicitly.

    The pool is now shared and owned by ``deep_agent.src.db.pool``,
    which the shutdown sequence closes once. Closing it from here
    would pull it out from under everything else still draining.
    """
    return None


async def record_llm_usage(
    run_id: str,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> None:
    """Add one model call's tokens to its run.

    Calls that report no tokens are still recorded: a model that
    answered is a fact about the run even when the provider declined
    to say what it cost, and dropping it makes `calls` disagree with
    reality.

    ``cached_tokens`` is a subset of ``input_tokens``, not an
    addition to it — cached prefix tokens are billed at a discount,
    not for free. It is recorded separately because it is the only
    signal that prompt caching is working at all: without it a hit
    and a miss look identical, and the system prompt carried a
    per-second timestamp for the life of the feature precisely
    because nothing would have shown the cost.
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
        "cached_tokens": max(0, int(cached_tokens or 0)),
    }
    try:
        pool = await _get_pool()
        # Bounded wait for a connection, and for the write.
        #
        # This is awaited inline on every model call, and the pool
        # holds at most four connections with psycopg_pool's default
        # 30-second acquire timeout. A database that is unreachable or
        # merely busy therefore added up to 30 seconds to each LLM
        # call — the opposite of best-effort. Usage accounting is
        # worth losing; a turn is not.
        async with asyncio.timeout(_WRITE_TIMEOUT_SECONDS):
            async with pool.connection(timeout=_ACQUIRE_TIMEOUT_SECONDS) as conn:
                await conn.execute(_RECORD_SQL, params)
        _failures = 0
    except Exception as exc:
        _failures += 1
        # The first failure is a warning, not a debug line. This
        # write has no legitimate failure mode — it is one UPDATE on
        # a pooled connection against a row that exists — so one
        # failure already means something is wrong.
        #
        # It mattered: a type-inference error in the SQL made every
        # write fail from the day the feature shipped. Each was
        # caught here, logged at debug, and the counter reset on
        # every pod restart, so the threshold never fired. The Usage
        # tab reported zero tokens as fact for the life of the
        # feature, and nothing anywhere said otherwise.
        if _failures >= _FAILURE_ALERT_THRESHOLD:
            logger.error("usage_recording_degraded", consecutive_failures=_failures)
        else:
            logger.warning(
                "Failed to record usage for run %s: %s", run_id, exc, exc_info=True
            )
