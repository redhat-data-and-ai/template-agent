"""Usage analytics, derived from tables the platform already writes.

**No write path.** Everything here is a query over `runs`, `thread` and
`message_feedback` — rows Aegra and the feedback endpoint produce as a
side effect of working. open-webui maintains a separate `chat_messages`
table for its analytics; a second write path is a second thing to keep
correct, and when the two disagree the dashboard is the one nobody
checks against reality. A query cannot drift from its source.

**No message content, ever.** open-webui's analytics API includes an
endpoint that returns users' actual messages to an admin. On an
internal platform that is a surveillance surface wearing a dashboard's
clothes, and it is not a capability this template should hand a fork by
default. Counts, durations, rates, token totals — never text. Error
messages are classified into types rather than echoed, because an
exception string routinely contains the input that caused it.

What is measured is chosen to answer the questions an operator actually
has on week one: is it being used, is it working, how slow is it, what
is it costing, and is anyone happy with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.rows import dict_row

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# A window wider than this turns a dashboard refresh into a sequential
# scan of the whole runs table.
MAX_WINDOW_DAYS = 365

# Statuses that mean the run is over and its duration is a duration.
#
# `interrupted` is deliberately absent. In this deployment it does not
# mean "a person paused it" — `lifecycle.py` and `startup.py` write it
# when a pod is drained or loses its lease, and the LeaseReaper then
# resumes the run. So an interrupted run is unfinished work, not a
# finished outcome. Counting it as terminal did two wrong things at
# once: its elapsed time is the 30s lease window rather than any work
# (93 such runs pulled the reported p95 from 52.9s down to 37.5s), and
# `success_rate` divides by successes + errors, so a fifth of the
# window silently left the "Succeeded" figure altogether.
_TERMINAL = ["success", "error", "timeout"]

# Statuses where the run is still expected to produce an answer —
# either running now, queued, or awaiting recovery.
_RECOVERABLE = ["interrupted"]

# How long a run may sit in a non-terminal status before it is read
# as stranded rather than working.
#
# A pod that dies mid-run leaves its row `running` forever: nothing
# rewrites a status on behalf of a process that is gone, and the
# reaper only resumes runs it holds a lease for. Two runs from a
# restart two days earlier were still being counted as "in flight",
# and because the count only ever grows, the figure drifts from
# "what is happening now" to "how many times has a pod been
# restarted mid-run".
#
# An hour matches `max_execute_timeout` in agent.yaml — the longest
# a single operation is permitted to take, so nothing legitimate is
# still running past it.
_STALE_AFTER_SECONDS = 3600


# Runs this agent performed for a person, as opposed to for a test.
#
# The BFF stamps `config.metadata.trace_id` on every run it proxies.
# Benchmark harnesses, internal planner invocations and subagent
# calls reach Aegra directly and carry none, and on this deployment
# they were 542 of 553 runs — so every unfiltered figure on the page
# was describing the test suite rather than the product.
#
# Applied to *every* query here, deliberately. A page where some
# numbers are filtered and others are not is worse than one where
# none are, because nothing on it can be compared with anything else.
_CLIENT_ONLY = "(config->'metadata'->>'trace_id') IS NOT NULL"
_CLIENT_ONLY_R = "(r.config->'metadata'->>'trace_id') IS NOT NULL"

# Where tokens come from, and why not from `runs.output`.
#
# Aegra persists a final output only for the default `values` stream
# mode. The UI streams `updates`/`messages`/`events`, so its runs land
# with `output = '{}'` — 354 of 451 rows in the live database. Deriving
# tokens from `output -> messages[]` therefore measured everything
# except real traffic. `deep_agent/src/analytics/usage.py` now records
# tokens per run from the LLM callback, which fires regardless of
# stream mode, and writes them to `runs.execution_params -> 'usage'`.
#
# The message-derived figures remain as a fallback so runs recorded
# before that change still read correctly. COALESCE, not a sum: a run
# must never be counted from both sources.
_TOKENS_CTE = """
derived AS (
    SELECT
        r.run_id,
        COALESCE(SUM((m -> 'usage_metadata' ->> 'input_tokens')::bigint), 0)  AS input_tokens,
        COALESCE(SUM((m -> 'usage_metadata' ->> 'output_tokens')::bigint), 0) AS output_tokens
    FROM runs r
    CROSS JOIN LATERAL jsonb_array_elements(
        CASE WHEN jsonb_typeof(r.output -> 'messages') = 'array'
             THEN r.output -> 'messages' ELSE '[]'::jsonb END
    ) AS m
    WHERE r.created_at >= now() - make_interval(days => %(days)s)
          AND (r.config->'metadata'->>'trace_id') IS NOT NULL
    GROUP BY r.run_id
),
tokens AS (
    SELECT
        r.run_id,
        COALESCE(
            (r.execution_params #>> '{usage,input_tokens}')::bigint,
            d.input_tokens, 0
        ) AS input_tokens,
        COALESCE(
            (r.execution_params #>> '{usage,output_tokens}')::bigint,
            d.output_tokens, 0
        ) AS output_tokens
    FROM runs r
    LEFT JOIN derived d ON d.run_id = r.run_id
    WHERE r.created_at >= now() - make_interval(days => %(days)s)
          AND (r.config->'metadata'->>'trace_id') IS NOT NULL
)
"""


@dataclass
class Window:
    """The period a figure covers, carried alongside it.

    A number without its window is unreadable — "412 runs" means
    nothing until you know whether that is a day or a quarter.
    """

    days: int

    def __post_init__(self) -> None:
        """Clamp the window so a hand-written query cannot scan the whole table."""
        self.days = max(1, min(int(self.days), MAX_WINDOW_DAYS))


@dataclass
class Summary:
    """Headline counters for one analytics window."""

    window_days: int
    runs: int = 0
    # Equal to `threads` by construction: every query in this module
    # is client-only, so a thread in scope is a conversation. Kept as
    # a distinct name because that is what the page asks for.
    conversations: int = 0
    threads: int = 0
    users: int = 0
    successes: int = 0
    errors: int = 0
    in_flight: int = 0
    # Runs a pod dropped and the reaper has not yet resumed. Its own
    # figure rather than part of `in_flight`: "running now" and
    # "stranded by a restart" prompt different actions, and merging
    # them hides a stuck queue behind a healthy-looking number.
    interrupted: int = 0
    # Non-terminal and far past any legitimate duration — a pod died
    # holding them. Separate from `in_flight` because "working now"
    # and "abandoned" call for different responses, and separate from
    # `interrupted` because nothing is going to resume these.
    stale: int = 0
    p50_seconds: float = 0.0
    p95_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    feedback_up: int = 0
    feedback_down: int = 0

    @property
    def success_rate(self) -> float | None:
        """Share of *finished* runs that worked, or None if none have.

        None rather than 0.0 or 100.0: a fresh deployment has no success
        rate, and inventing one puts a green number on a system nobody
        has used.
        """
        finished = self.successes + self.errors
        return round(self.successes / finished * 100, 1) if finished else None

    @property
    def satisfaction(self) -> float | None:
        """Thumbs-up share of rated runs, or None when nothing was rated."""
        rated = self.feedback_up + self.feedback_down
        return round(self.feedback_up / rated * 100, 1) if rated else None


async def _connect() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(
        settings.database_uri, row_factory=dict_row
    )


async def summary(window: Window) -> Summary:
    """Headline figures for the window."""
    sql = f"""
    WITH {_TOKENS_CTE},
    scoped AS (
        SELECT r.run_id, r.thread_id, r.user_id, r.status, r.updated_at,
               EXTRACT(EPOCH FROM (r.updated_at - r.created_at)) AS seconds
        FROM runs r
        WHERE r.created_at >= now() - make_interval(days => %(days)s)
          AND {_CLIENT_ONLY_R}
    )
    SELECT
        COUNT(*)                                             AS runs,
        -- Every query here is already client-only, so a thread in
        -- scope *is* a conversation. Both names are returned because
        -- the API had shipped with them; they are the same number by
        -- construction rather than by coincidence.
        COUNT(DISTINCT thread_id)                            AS conversations,
        COUNT(DISTINCT thread_id)                            AS threads,
        COUNT(DISTINCT user_id)                              AS users,
        COUNT(*) FILTER (WHERE status = 'success')           AS successes,
        COUNT(*) FILTER (WHERE status = 'error')             AS errors,
        COUNT(*) FILTER (WHERE NOT (status = ANY(%(terminal)s))
                           AND NOT (status = ANY(%(recoverable)s))
                           AND updated_at >= now() - make_interval(
                                 secs => %(stale_after)s))          AS in_flight,
        COUNT(*) FILTER (WHERE status = ANY(%(recoverable)s))        AS interrupted,
        -- Non-terminal and untouched for longer than any run may
        -- legitimately take. Reported rather than hidden: a growing
        -- number here means pods are dying mid-run.
        COUNT(*) FILTER (WHERE NOT (status = ANY(%(terminal)s))
                           AND NOT (status = ANY(%(recoverable)s))
                           AND updated_at < now() - make_interval(
                                 secs => %(stale_after)s))           AS stale,
        -- Percentiles over finished runs only. An in-flight run's
        -- elapsed time is not its duration, and including it drags
        -- p95 toward "however long ago the request arrived".
        COALESCE(PERCENTILE_CONT(0.5) WITHIN GROUP (
            ORDER BY seconds) FILTER (WHERE status = ANY(%(terminal)s)), 0) AS p50,
        COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY seconds) FILTER (WHERE status = ANY(%(terminal)s)), 0) AS p95,
        COALESCE((SELECT SUM(input_tokens) FROM tokens), 0)  AS input_tokens,
        COALESCE((SELECT SUM(output_tokens) FROM tokens), 0) AS output_tokens
    FROM scoped
    """
    params = {
        "days": window.days,
        "terminal": _TERMINAL,
        "recoverable": _RECOVERABLE,
        "stale_after": _STALE_AFTER_SECONDS,
    }

    async with await _connect() as conn:
        row = await (await conn.execute(sql, params)).fetchone() or {}
        feedback = (
            await (
                await conn.execute(
                    """
                SELECT
                    COUNT(*) FILTER (WHERE feedback = 'up')   AS up,
                    COUNT(*) FILTER (WHERE feedback = 'down') AS down
                FROM message_feedback
                WHERE created_at >= now() - make_interval(days => %(days)s)
                """,
                    {"days": window.days},
                )
            ).fetchone()
            or {}
        )

    return Summary(
        window_days=window.days,
        runs=int(row.get("runs") or 0),
        conversations=int(row.get("conversations") or 0),
        threads=int(row.get("threads") or 0),
        users=int(row.get("users") or 0),
        successes=int(row.get("successes") or 0),
        errors=int(row.get("errors") or 0),
        in_flight=int(row.get("in_flight") or 0),
        stale=int(row.get("stale") or 0),
        interrupted=int(row.get("interrupted") or 0),
        p50_seconds=round(float(row.get("p50") or 0), 2),
        p95_seconds=round(float(row.get("p95") or 0), 2),
        input_tokens=int(row.get("input_tokens") or 0),
        output_tokens=int(row.get("output_tokens") or 0),
        feedback_up=int(feedback.get("up") or 0),
        feedback_down=int(feedback.get("down") or 0),
    )


async def timeseries(window: Window, bucket: str = "day") -> list[dict[str, Any]]:
    """Runs, errors and p95 per bucket.

    Buckets with no runs are emitted as zeroes by the generated series
    rather than omitted — a gap in a chart reads as missing data, not
    as a quiet weekend.

    The series starts one unit *after* the truncated window edge. The
    first bucket would otherwise be a partial one — a 7-day window
    truncated to midnight only covers part of that first day — and it
    rendered as a short bar that looked like a drop in traffic rather
    than like an arbitrary cut-off.
    """
    unit = "hour" if bucket == "hour" else "day"
    sql = f"""
    WITH bounds AS (
        SELECT date_trunc(%(unit)s, now() - make_interval(days => %(days)s))
               + ('1 ' || %(unit)s)::interval AS first_bucket
    ),
    buckets AS (
        SELECT generate_series(
            (SELECT first_bucket FROM bounds),
            date_trunc(%(unit)s, now()),
            ('1 ' || %(unit)s)::interval
        ) AS bucket
    ),
    scoped AS (
        SELECT date_trunc(%(unit)s, created_at) AS bucket,
               status,
               EXTRACT(EPOCH FROM (updated_at - created_at)) AS seconds
        FROM runs
        WHERE created_at >= (SELECT first_bucket FROM bounds)
          AND {_CLIENT_ONLY}
    )
    SELECT
        b.bucket,
        COUNT(s.*)                                   AS runs,
        COUNT(s.*) FILTER (WHERE s.status = 'error') AS errors,
        COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY s.seconds) FILTER (WHERE s.status = ANY(%(terminal)s)), 0) AS p95
    FROM buckets b
    LEFT JOIN scoped s ON s.bucket = b.bucket
    GROUP BY b.bucket
    ORDER BY b.bucket
    """
    async with await _connect() as conn:
        rows = await (
            await conn.execute(
                sql,
                {"unit": unit, "days": window.days, "terminal": _TERMINAL},
            )
        ).fetchall()

    return [
        {
            "bucket": row["bucket"].isoformat(),
            "runs": int(row["runs"] or 0),
            "errors": int(row["errors"] or 0),
            "p95_seconds": round(float(row["p95"] or 0), 2),
        }
        for row in rows
    ]


async def by_model(window: Window) -> list[dict[str, Any]]:
    """Volume, tokens, latency and error rate per model.

    Per *model*, not per assistant: model choice is the lever an
    operator actually has, and §24 was a long argument about picking
    one on measurements rather than on reputation.

    A run is counted under every model it used, because a run with an
    orchestrator on one model and subagents on another is evidence
    about both. The earlier version collapsed each run to
    `MAX(model_name)` — whichever name sorted last — which attributed
    work to a model that may not have done it. `runs` is therefore a
    distinct count and will sum to more than the total run count on a
    deployment that mixes models; `calls` is the per-model request
    count, which is the figure that tracks cost.
    """
    sql = """
    WITH per_run AS (
        SELECT
            r.run_id,
            r.status,
            EXTRACT(EPOCH FROM (r.updated_at - r.created_at)) AS seconds,
            COALESCE(r.execution_params #> '{usage,models}', '{}'::jsonb) AS models
        FROM runs r
        WHERE r.created_at >= now() - make_interval(days => %(days)s)
          AND (r.config->'metadata'->>'trace_id') IS NOT NULL
    ),
    -- Same fallback as the token totals: runs that predate per-model
    -- recording are read from their output messages where they have
    -- any. Only `ai` messages count as calls — a human turn has no
    -- model and would inflate the figure that stands in for cost.
    derived AS (
        SELECT
            p.run_id,
            COALESCE(msg -> 'response_metadata' ->> 'model_name', 'unknown') AS model,
            COALESCE(SUM((msg -> 'usage_metadata' ->> 'input_tokens')::bigint), 0)  AS input_tokens,
            COALESCE(SUM((msg -> 'usage_metadata' ->> 'output_tokens')::bigint), 0) AS output_tokens,
            COUNT(*)                                                                AS calls
        FROM per_run p
        JOIN runs r ON r.run_id = p.run_id
        CROSS JOIN LATERAL jsonb_array_elements(
            CASE WHEN jsonb_typeof(r.output -> 'messages') = 'array'
                 THEN r.output -> 'messages' ELSE '[]'::jsonb END
        ) AS msg
        WHERE p.models = '{}'::jsonb
          AND msg ->> 'type' = 'ai'
        GROUP BY 1, 2
    ),
    -- Runs with neither source still have to appear somewhere. They
    -- land under 'unrecorded' rather than being silently dropped or
    -- folded into a real model's totals — a row of zeroes against a
    -- named model is a claim about that model that nothing supports.
    expanded AS (
        SELECT p.run_id, p.status, p.seconds,
               m.key AS model,
               COALESCE((m.value ->> 'input_tokens')::bigint, 0)  AS input_tokens,
               COALESCE((m.value ->> 'output_tokens')::bigint, 0) AS output_tokens,
               COALESCE((m.value ->> 'calls')::bigint, 0)         AS calls
        FROM per_run p
        CROSS JOIN LATERAL jsonb_each(p.models) AS m
        UNION ALL
        SELECT p.run_id, p.status, p.seconds,
               d.model, d.input_tokens, d.output_tokens, d.calls
        FROM per_run p
        JOIN derived d ON d.run_id = p.run_id
        UNION ALL
        SELECT p.run_id, p.status, p.seconds, 'unrecorded', 0, 0, 0
        FROM per_run p
        WHERE p.models = '{}'::jsonb
          AND NOT EXISTS (SELECT 1 FROM derived d WHERE d.run_id = p.run_id)
    )
    SELECT
        model,
        COUNT(DISTINCT run_id)                                  AS runs,
        SUM(calls)                                              AS calls,
        COUNT(DISTINCT run_id) FILTER (WHERE status = 'error')  AS errors,
        COALESCE(SUM(input_tokens), 0)                          AS input_tokens,
        COALESCE(SUM(output_tokens), 0)                         AS output_tokens,
        COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY seconds
        ) FILTER (WHERE status = ANY(%(terminal)s)), 0)         AS p95
    FROM expanded
    GROUP BY 1
    ORDER BY runs DESC
    """
    async with await _connect() as conn:
        rows = await (
            await conn.execute(sql, {"days": window.days, "terminal": _TERMINAL})
        ).fetchall()

    return [
        {
            "model": row["model"],
            "runs": int(row["runs"]),
            "calls": int(row["calls"] or 0),
            "errors": int(row["errors"]),
            "input_tokens": int(row["input_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            "p95_seconds": round(float(row["p95"] or 0), 2),
        }
        for row in rows
    ]


# Error strings carry user input — a path, a filename, a prompt
# fragment. Classifying them keeps the operational signal (what kind of
# thing is failing, how often) without putting anyone's data on a
# dashboard. Order matters: first match wins, so specific classes come
# before broad ones.
#
# The list is grounded in failures this deployment has actually
# produced, not in what an exception taxonomy ought to contain. Every
# class below has a run behind it. The first version of this list
# sorted all 17 errors in the live database into `other`, which is the
# useful thing a classifier can tell you about itself.
_ERROR_CLASSES: list[tuple[str, tuple[str, ...]]] = [
    ("model_timeout", ("timeout", "timed out", "deadline exceeded")),
    ("model_rate_limit", ("rate limit", "429", "resource exhausted", "quota")),
    ("model_refused", ("safety", "blocked", "recitation", "finish_reason")),
    # Credentials and bad model names look alike from here: both mean
    # the model could not be constructed, and both are fixed in config
    # rather than in code. "No Google credentials found" and
    # "Failed to create Gemini model 'x'" are both real.
    (
        "model_config",
        (
            "credentials",
            "api key",
            "api_key",
            "failed to create",
            "model not found",
            "unknown model",
            "invalid_grant",
            "unauthorized",
            "permission denied",
            "403",
        ),
    ),
    ("context_too_long", ("context length", "token limit", "too many tokens")),
    # "Max retries exceeded after repeated worker failures" — the
    # run gave up rather than failing on one cause, so the real
    # failure is upstream and this is its symptom.
    ("retries_exhausted", ("max retries", "retries exceeded")),
    ("storage", ("psycopg", "postgres", "connection refused", "redis")),
    ("cancelled", ("cancelled", "canceled", "interrupted")),
    # A Python exception reaching the user is our bug, not the
    # model's, and it should never sit in the same bucket as a rate
    # limit. "string indices must be integers, not 'str'" was §25's
    # FileData bug and took an end-to-end run to find.
    (
        "agent_defect",
        (
            "indices must be",
            "attributeerror",
            "typeerror",
            "keyerror",
            "nonetype",
            "not subscriptable",
            "unexpected keyword",
        ),
    ),
    ("tool_failure", ("tool", "mcp", "subprocess")),
]


def classify_error(message: str | None) -> str:
    """Bucket an exception string. Never returns the string itself."""
    if not message:
        return "unknown"
    lowered = message.lower()
    for label, needles in _ERROR_CLASSES:
        if any(needle in lowered for needle in needles):
            return label
    return "other"


async def errors(window: Window) -> list[dict[str, Any]]:
    """Failure counts by class, most frequent first."""
    async with await _connect() as conn:
        rows = await (
            await conn.execute(
                f"""
                SELECT error_message, updated_at
                FROM runs
                WHERE status = 'error'
                  AND created_at >= now() - make_interval(days => %(days)s)
                  AND {_CLIENT_ONLY}
                ORDER BY updated_at DESC
                LIMIT 5000
                """,
                {"days": window.days},
            )
        ).fetchall()

    counts: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = classify_error(row.get("error_message"))
        entry = counts.setdefault(
            label, {"class": label, "count": 0, "last_seen": None}
        )
        entry["count"] += 1
        seen = row.get("updated_at")
        if seen and (entry["last_seen"] is None or seen > entry["last_seen"]):
            entry["last_seen"] = seen

    return sorted(
        (
            {
                **entry,
                "last_seen": entry["last_seen"].isoformat()
                if entry["last_seen"]
                else None,
            }
            for entry in counts.values()
        ),
        key=lambda e: -e["count"],
    )


@dataclass
class UserRow:
    """One user's activity over the window."""

    user_id: str
    runs: int = 0
    threads: int = 0
    tokens: int = 0
    errors: int = 0


@dataclass
class UserBreakdown:
    """Per-user usage. Off unless a deployment turns it on."""

    enabled: bool
    rows: list[UserRow] = field(default_factory=list)
    reason: str = ""


async def by_user(window: Window, *, enabled: bool, limit: int = 50) -> UserBreakdown:
    """Usage attributed to named people.

    Gated, and off by default. Cost attribution is a real need, so this
    exists — but "who talked to the assistant most this month" is also
    a management report nobody consented to, and on an internal tool
    the difference between the two is a config flag somebody should
    have to set deliberately.
    """
    if not enabled:
        return UserBreakdown(
            enabled=False,
            reason="per-user analytics are disabled (analytics.expose_users)",
        )

    sql = f"""
    WITH {_TOKENS_CTE}
    SELECT
        r.user_id,
        COUNT(*)                                   AS runs,
        COUNT(DISTINCT r.thread_id)                AS threads,
        COUNT(*) FILTER (WHERE r.status = 'error') AS errors,
        COALESCE(SUM(t.input_tokens + t.output_tokens), 0) AS tokens
    FROM runs r
    LEFT JOIN tokens t ON t.run_id = r.run_id
    WHERE r.created_at >= now() - make_interval(days => %(days)s)
          AND (r.config->'metadata'->>'trace_id') IS NOT NULL
    GROUP BY r.user_id
    ORDER BY runs DESC
    LIMIT %(limit)s
    """
    async with await _connect() as conn:
        rows = await (
            await conn.execute(sql, {"days": window.days, "limit": limit})
        ).fetchall()

    return UserBreakdown(
        enabled=True,
        rows=[
            UserRow(
                user_id=row["user_id"],
                runs=int(row["runs"]),
                threads=int(row["threads"]),
                tokens=int(row["tokens"]),
                errors=int(row["errors"]),
            )
            for row in rows
        ],
    )
