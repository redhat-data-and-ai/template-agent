"""PostgreSQL helpers for the eval pod.

Handles all database access: table setup, writing completed eval results to
the `evals` table, and reading back per-run results from `evaluation_results`.

Uses synchronous psycopg2 — called from a thread pool inside the asyncio
event loop managed by eval_api.py.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "template_agent")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")
if not POSTGRES_PASSWORD:
    log.warning("POSTGRES_PASSWORD not set — connection may fail")

_HASH_EXTENSIONS = {".md", ".yaml", ".json"}
_HASH_EXCLUDE_DIRS = {"evals", "deployment"}


def _compute_config_hash(config_dir: str) -> str:
    """SHA256 of behavior-relevant config files (prompts, skills, runtime, tools)."""
    import hashlib
    from pathlib import Path

    h = hashlib.sha256()
    base = Path(config_dir)
    if base.exists():
        for fpath in sorted(base.rglob("*")):
            if not fpath.is_file():
                continue
            if fpath.suffix not in _HASH_EXTENSIONS:
                continue
            if any(
                part in _HASH_EXCLUDE_DIRS for part in fpath.relative_to(base).parts
            ):
                continue
            h.update(str(fpath.relative_to(base)).encode())
            h.update(fpath.read_bytes())
    return h.hexdigest()[:16]  # 16-char prefix is enough


_config_dir = os.environ.get("AGENT_CONFIG_DIR", "config/agent")
_env_hash = os.environ.get("AGENT_CONFIG_HASH", "")


def _get_config_hash() -> str:
    """Return config hash — env var first, compute from files as fallback (CLI path only)."""
    return _env_hash or _compute_config_hash(_config_dir)


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS evals (
    id              SERIAL PRIMARY KEY,
    config_hash     TEXT NOT NULL,
    eval_status     TEXT NOT NULL DEFAULT 'in_progress',
    eval_score      FLOAT,
    ls_run_ids      TEXT[],
    pass            INTEGER DEFAULT 0,
    fail            INTEGER DEFAULT 0,
    error           INTEGER DEFAULT 0,
    judge_model     TEXT,
    results_detail  JSONB,
    force_reeval    BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS evals_config_hash ON evals (config_hash);
CREATE INDEX IF NOT EXISTS evals_status ON evals (eval_status);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id                      SERIAL PRIMARY KEY,
    run_id                  VARCHAR(36) NOT NULL,
    timestamp               TIMESTAMP NOT NULL,
    conversation_group_id   VARCHAR(255) NOT NULL,
    tag                     VARCHAR(100),
    turn_id                 VARCHAR(100),
    metric_identifier       VARCHAR(255) NOT NULL,
    metric_metadata         TEXT,
    result                  VARCHAR(20) NOT NULL,
    score                   FLOAT,
    threshold               FLOAT,
    reason                  TEXT,
    query                   TEXT,
    response                TEXT,
    execution_time          FLOAT,
    evaluation_latency      FLOAT,
    api_input_tokens        INTEGER,
    api_output_tokens       INTEGER,
    judge_llm_input_tokens  INTEGER,
    judge_llm_output_tokens INTEGER,
    embedding_tokens        INTEGER,
    judge_scores            TEXT,
    time_to_first_token     FLOAT,
    streaming_duration      FLOAT,
    agent_latency           FLOAT,
    tokens_per_second       FLOAT,
    tool_calls              TEXT,
    contexts                TEXT,
    expected_response       TEXT,
    expected_intent         TEXT,
    expected_keywords       TEXT,
    expected_tool_calls     TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_results_run_id ON evaluation_results (run_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_group_id ON evaluation_results (conversation_group_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_metric ON evaluation_results (metric_identifier);
CREATE INDEX IF NOT EXISTS idx_eval_results_timestamp ON evaluation_results (timestamp);
"""


def _get_conn() -> Any:
    import psycopg2

    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        connect_timeout=5,
    )


_table_ensured = False


_MIGRATIONS = [
    "ALTER TABLE evals ADD COLUMN IF NOT EXISTS ls_run_ids TEXT[]",
    "ALTER TABLE evals DROP COLUMN IF EXISTS ls_run_id",
    "DROP INDEX IF EXISTS evals_ls_run_id",
]


def ensure_table() -> None:
    """Create evals table if it doesn't exist, then apply additive migrations."""
    global _table_ensured
    if _table_ensured:
        return
    try:
        conn = _get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(_CREATE_TABLE)
                    for migration in _MIGRATIONS:
                        cur.execute(migration)
        finally:
            conn.close()
        _table_ensured = True
        log.info("eval_postgres_tables_ready: evals + evaluation_results verified")
    except Exception as exc:
        log.warning(
            "eval_postgres_ensure_table_failed (%s): %s", type(exc).__name__, exc
        )


def write_eval_result(
    passed: int,
    failed: int,
    errors: int,
    eval_score: float,
    ls_run_ids: list[str] | None = None,
    judge_model: str = "",
    results_detail: dict[str, Any] | None = None,
    config_hash: str | None = None,
    eval_row_id: int | None = None,
) -> None:
    """Persist completed eval results to PostgreSQL.

    When eval_row_id is provided, targets that exact row to avoid
    concurrent runs updating the wrong record.
    """
    effective_hash = config_hash or _get_config_hash()
    try:
        conn = _get_conn()
        now = datetime.now(UTC)
        try:
            with conn:
                with conn.cursor() as cur:
                    if eval_row_id is not None:
                        cur.execute(
                            """
                            UPDATE evals
                            SET eval_status   = 'completed',
                                ls_run_ids    = %s,
                                eval_score    = %s,
                                pass          = %s,
                                fail          = %s,
                                error         = %s,
                                judge_model   = %s,
                                results_detail = %s,
                                updated_at    = %s,
                                completed_at  = %s
                            WHERE id = %s
                            RETURNING id
                            """,
                            (
                                ls_run_ids,
                                eval_score,
                                passed,
                                failed,
                                errors,
                                judge_model,
                                json.dumps(results_detail) if results_detail else None,
                                now,
                                now,
                                eval_row_id,
                            ),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE evals
                            SET eval_status   = 'completed',
                                ls_run_ids    = %s,
                                eval_score    = %s,
                                pass          = %s,
                                fail          = %s,
                                error         = %s,
                                judge_model   = %s,
                                results_detail = %s,
                                updated_at    = %s,
                                completed_at  = %s
                            WHERE id = (
                                SELECT id FROM evals
                                WHERE config_hash = %s
                                  AND eval_status IN ('in_progress', 'error')
                                ORDER BY created_at DESC
                                LIMIT 1
                            )
                            RETURNING id
                            """,
                            (
                                ls_run_ids,
                                eval_score,
                                passed,
                                failed,
                                errors,
                                judge_model,
                                json.dumps(results_detail) if results_detail else None,
                                now,
                                now,
                                effective_hash,
                            ),
                        )
                    row = cur.fetchone()
        finally:
            conn.close()

        if row:
            log.info(
                "eval_result_written_to_postgres id=%s score=%.3f pass=%d fail=%d error=%d",
                row[0],
                eval_score,
                passed,
                failed,
                errors,
            )
        else:
            log.warning(
                "eval_postgres_no_matching_record config_hash=%s "
                "(check AGENT_CONFIG_DIR env var — CWD may resolve wrong directory)",
                effective_hash,
            )
    except Exception as exc:
        log.error("eval_postgres_write_failed (%s): %s", type(exc).__name__, exc)


def load_results_since(run_started_at: datetime) -> dict[str, Any]:
    """Load eval results from evaluation_results written since run_started_at.

    Finds all run_ids produced after run_started_at (parallel tag runs each
    write their own run_id), fetches every row for those runs, and returns a
    dict with 'turns', 'summary', and 'ls_run_ids'. Returns {} on any failure.

    Each agent deployment has its own Postgres DB so results from other agents
    cannot appear here. The agentpod's _atomic_set_in_progress prevents most
    concurrent runs, but with KEDA max>1 eval-runner pods two near-simultaneous
    triggers could slip through the SELECT+INSERT gap and mix their results.
    Accepted risk — concurrent triggers are rare and KEDA is typically max=1 in
    production. If this becomes a problem, add org/name columns to
    evaluation_results and backfill after each gather (see git history).
    """
    try:
        import psycopg2.extras

        conn = _get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM evaluation_results WHERE timestamp >= %s",
                    (run_started_at,),
                )
                raw_rows = cur.fetchall()
        finally:
            conn.close()

        if not raw_rows:
            log.warning("No evaluation_results rows found since %s", run_started_at)
            return {}

        run_ids = list(dict.fromkeys(r["run_id"] for r in raw_rows))
        log.info("Loading DB results for run_ids=%s", run_ids)
    except Exception as exc:
        log.warning("load_results_since_failed (%s): %s", type(exc).__name__, exc)
        return {}

    overall_counts: dict[str, int] = {"PASS": 0, "FAIL": 0, "ERROR": 0}
    by_metric: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"pass": 0, "fail": 0, "scores": []}
    )
    by_conversation: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"pass": 0, "fail": 0}
    )

    turns: list[dict[str, Any]] = []
    for r in raw_rows:
        result_val = str(r.get("result") or "").upper()
        if result_val not in overall_counts:
            result_val = "ERROR"
        overall_counts[result_val] += 1

        metric = str(r.get("metric_identifier") or "unknown")
        if result_val == "PASS":
            by_metric[metric]["pass"] += 1
        elif result_val == "FAIL":
            by_metric[metric]["fail"] += 1
        try:
            by_metric[metric]["scores"].append(float(r.get("score") or 0))
        except (TypeError, ValueError):
            pass

        conv_id = str(r.get("conversation_group_id") or "unknown")
        if result_val == "PASS":
            by_conversation[conv_id]["pass"] += 1
        elif result_val == "FAIL":
            by_conversation[conv_id]["fail"] += 1

        turns.append(
            {
                k: (
                    v.isoformat()
                    if isinstance(v, datetime)
                    else str(v)
                    if isinstance(v, float)
                    else v
                )
                for k, v in r.items()
            }
        )

    total = len(raw_rows)
    overall_with_rates: dict[str, Any] = dict(overall_counts)
    overall_with_rates["pass_rate"] = (
        round(overall_counts["PASS"] / total, 3) if total else 0.0
    )
    overall_with_rates["fail_rate"] = (
        round(overall_counts["FAIL"] / total, 3) if total else 0.0
    )
    overall_with_rates["error_rate"] = (
        round(overall_counts["ERROR"] / total, 3) if total else 0.0
    )

    by_metric_out: dict[str, Any] = {}
    for metric, data in by_metric.items():
        m_total = data["pass"] + data["fail"]
        by_metric_out[metric] = {
            "pass": data["pass"],
            "fail": data["fail"],
            "pass_rate": round(data["pass"] / m_total, 3) if m_total else 0.0,
        }

    summary: dict[str, Any] = {
        "total_evaluations": total,
        "summary_stats": {
            "overall": overall_with_rates,
            "by_metric": by_metric_out,
            "by_conversation": dict(by_conversation),
        },
    }

    return {"turns": turns, "summary": summary, "ls_run_ids": run_ids}


def get_results_by_run_id(run_id: str) -> list[dict[str, Any]]:
    """Return per-turn rows for a specific run_id. Returns [] if not found."""
    log.info("get_results_by_run_id: run_id=%s", run_id)
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT conversation_group_id, turn_id, metric_identifier, result, score, reason "
                    "FROM evaluation_results WHERE run_id = %s ORDER BY id",
                    (run_id,),
                )
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        log.error("get_results_by_run_id failed (%s): %s", type(exc).__name__, exc)
        raise
    log.info("get_results_by_run_id: run_id=%s returned %d rows", run_id, len(rows))
    return rows


# ── Dataset access (eval_datasets table written by agentpod) ──────────────────

from eval_cases import get_defaults_for_tag, get_tool_turn_metrics  # noqa: E402


def _dataset_to_eval_cases(dataset: dict) -> list[dict]:
    """Convert UI dataset JSON to the list[dict] format used by eval_cases.yaml.

    Uses get_defaults_for_tag() from eval_cases — the single source of truth
    for tag → metrics mapping. No duplicate mapping here.
    """
    cases = []
    for tc in dataset.get("cases", []):
        tag = tc.get("tag", "non_hitl")
        tag_defaults = get_defaults_for_tag(tag)
        metrics = tag_defaults.get("turn_metrics", ["custom:answer_correctness"])
        turns = []
        for i, turn in enumerate(tc.get("turns", []), 1):
            turn_data: dict[str, Any] = {
                "turn_id": turn.get("id", f"turn_{i}"),
                "query": turn.get("userMessage", ""),
                "expected_response": turn.get("expectedResponse", ""),
                "turn_metrics": list(metrics),
            }
            # For HITL cases, mark only the LAST turn as the HITL turn.
            # Earlier turns are normal info-gathering turns that don't need approval;
            # only the final action turn (e.g. "email results") triggers the interrupt.
            all_turns = tc.get("turns", [])
            is_last_turn = i == len(all_turns)
            if tag_defaults.get("hitl") and is_last_turn:
                turn_data["hitl"] = True
                turn_data["expected_intent"] = (
                    turn.get("expectedIntent")
                    or "request approval before taking action"
                )
            elif turn.get("expectedIntent"):
                turn_data["expected_intent"] = turn["expectedIntent"]
                # Auto-add intent_eval when user specifies an expected intent
                if "custom:intent_eval" not in turn_data["turn_metrics"]:
                    turn_data["turn_metrics"].append("custom:intent_eval")
            # expectedKeywords: list of AND-rows; each row is comma-separated OR values.
            # Filter blank rows, auto-add keywords_eval metric when present.
            raw_kw = [
                str(r).strip()
                for r in turn.get("expectedKeywords", [])
                if str(r).strip()
            ]
            if raw_kw:
                turn_data["expected_keywords"] = [
                    [v.strip() for v in row.split(",") if v.strip()] for row in raw_kw
                ]
                if "custom:keywords_eval" not in turn_data["turn_metrics"]:
                    turn_data["turn_metrics"].append("custom:keywords_eval")
            if turn.get("toolCallEnabled") and turn.get("expectedToolCalls"):
                tool_calls = []
                for c in turn["expectedToolCalls"]:
                    args = {
                        a["key"]: (str(a.get("value", "")).strip() or ".*")
                        for a in c.get("arguments", [])
                        if a.get("key")
                    }
                    tool_calls.append(
                        {
                            "tool_name": c.get("toolName", ""),
                            "arguments": args,
                        }
                    )

                if tool_calls:
                    # Format as N singleton sequences [[tc1], [tc2], ...] to match
                    # run_eval.py's actual format [[tc] for tc in all_tool_calls].
                    # _compare_tool_call_sequence requires len(expected)==len(actual)
                    # per sequence — singletons satisfy this and let full_match=False
                    # check that ALL expected tools appear (AND, not OR).
                    # Argument values support regex (e.g. .* matches any value).
                    turn_data["expected_tool_calls"] = [[tc] for tc in tool_calls]
                    if "custom:tool_eval" not in turn_data["turn_metrics"]:
                        turn_data["turn_metrics"].append("custom:tool_eval")
                    # tool_turn_metrics (from tag_metrics.yaml) only apply when tool
                    # calls are expected — e.g. ragas:faithfulness requires contexts
                    # (tool results) and errors when no tools are called.
                    for m in get_tool_turn_metrics():
                        if m not in turn_data["turn_metrics"]:
                            turn_data["turn_metrics"].append(m)
                    # delegation_compliance only makes sense when sub-agent tool calls
                    # are expected — delegation happens via tool calls, not inline text.
                    if (
                        tag == "multi_agent"
                        and "geval:delegation_compliance"
                        not in turn_data["turn_metrics"]
                    ):
                        turn_data["turn_metrics"].append("geval:delegation_compliance")
                    turn_data.setdefault("turn_metrics_metadata", {})
                    # ordered=True only reliable for orchestrator-level calls;
                    # subagent calls fetched from Postgres checkpoint_blobs have
                    # no guaranteed ordering — warn users via UI about this.
                    turn_data["turn_metrics_metadata"]["custom:tool_eval"] = {
                        "ordered": bool(turn.get("toolCallOrdered", False)),
                        "full_match": False,
                    }
            turns.append(turn_data)

        # If any turn expects tool calls, mark the case tag as tool_use so the
        # eval runner fetches subagent tool calls from Postgres checkpoint_blobs.
        # We preserve the original tag in the description for reference.
        any_tool_calls = any(
            t.get("toolCallEnabled") and t.get("expectedToolCalls")
            for t in tc.get("turns", [])
        )
        effective_tag = "tool_use" if any_tool_calls else tag

        case_entry: dict[str, Any] = {
            "conversation_group_id": tc.get("name") or tc.get("id", ""),
            "description": tc.get("description", f"tag:{tag}"),
            "tag": effective_tag,
            "turns": turns,
        }
        # Add conversation-level metrics if the tag defines them
        conv_metrics = tag_defaults.get("conversation_metrics", [])
        if conv_metrics:
            case_entry["conversation_metrics"] = list(conv_metrics)
        cases.append(case_entry)
    return cases


def fetch_dataset() -> tuple[list[dict] | None, str | None]:
    """Fetch eval cases and judge_model from eval_datasets in one query.

    Returns (cases, judge_model) — either may be None if no dataset stored or on error.
    """
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT dataset, judge_model FROM eval_datasets LIMIT 1")
                row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            return None, None

        raw, judge_model = row
        dataset = json.loads(raw) if isinstance(raw, str) else raw
        cases = _dataset_to_eval_cases(dataset)
        log.info("fetch_dataset: %d cases judge_model=%s", len(cases), judge_model)
        return (cases or None), (str(judge_model) if judge_model else None)

    except Exception as exc:
        log.warning("fetch_dataset failed: %s", exc)
        return None, None
