"""Eval runner REST API.

Runs as a separate Deployment (KEDA HTTP-scaled, min=0 max=2) in the same
namespace as the agentpod. Reads eval_cases.yaml and system.yaml from the
agent config PVC (written by agent-engine at deploy time). Results are
written to Postgres so the agentpod /evals/results route can serve them.

Endpoints:
    POST /evals/run                Run all patterns
    POST /evals/run/{pattern}      Run one pattern (tool_use / hitl / structured_output / multi_agent)
    GET  /evals/status             Current run status
    GET  /evals/results            Latest run summary
    GET  /evals/results/{run_id}   Specific run summary
    GET  /health                   Liveness check
"""

from __future__ import annotations

import asyncio
import functools
import hmac
import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from eval_cases import get_metric_thresholds, load_cases
from eval_postgres import _compute_config_hash, fetch_dataset
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

AGENT_URL = os.environ.get("AGENT_HOST", "http://localhost:5002")
_EVAL_TOKEN_REFRESH_ENABLED = (
    os.environ.get("EVAL_TOKEN_REFRESH_ENABLED", "false").lower() == "true"
)
_EVAL_INTERNAL_TOKEN = os.environ["EVAL_INTERNAL_TOKEN"]
if not _EVAL_INTERNAL_TOKEN:
    raise RuntimeError(
        "EVAL_INTERNAL_TOKEN is empty — the eval runner refuses to start without it. "
        "Generate one with: openssl rand -hex 32"
    )


def _call_cleanup_endpoint() -> None:
    """Notify the agent pod to delete eval Redis keys. Best-effort — never raises."""
    if not _EVAL_TOKEN_REFRESH_ENABLED or not _EVAL_INTERNAL_TOKEN:
        return
    try:
        import httpx as _httpx

        _httpx.post(
            f"{AGENT_URL.rstrip('/')}/evals/internal/cleanup",
            json={},
            headers={"x-internal-token": _EVAL_INTERNAL_TOKEN},
            timeout=5,
        )
    except Exception:
        pass  # 60-min TTL is the fallback


# Derive eval file paths from AGENT_CONFIG_DIR so no separate env vars needed.
# Can still be overridden explicitly via EVAL_CASES_PATH / EVAL_SYSTEM_CONFIG.
def _resolve_config_dir() -> str:
    """Return AGENT_CONFIG_DIR from env, or auto-detect the config directory.

    Walks up from this file looking for config/agent
    (local dev layout: eval-runner/../config/agent).
    """
    if val := os.environ.get("AGENT_CONFIG_DIR"):
        return val
    here = Path(__file__).resolve().parent
    for candidate in [
        here.parent.parent / "config" / "agent",  # repo-root/config/agent (local dev)
        here.parent / "config" / "agent",  # eval-runner/config/agent (container)
    ]:
        resolved = candidate.resolve()
        if resolved.is_dir():
            log.info("AGENT_CONFIG_DIR not set — using auto-detected: %s", resolved)
            return str(resolved)
    log.warning(
        "AGENT_CONFIG_DIR not set and config/agent not found; falling back to /agent-config"
    )
    return "/agent-config"


_agent_config_dir = _resolve_config_dir()
EVAL_CASES_PATH = Path(
    os.environ.get(
        "EVAL_CASES_PATH", f"{_agent_config_dir}/evals/lightspeed/eval_cases.yaml"
    )
)
EVAL_SYSTEM_CONFIG = Path(
    os.environ.get(
        "EVAL_SYSTEM_CONFIG", f"{_agent_config_dir}/evals/lightspeed/system.yaml"
    )
)
EVAL_OUTPUT_DIR = Path(
    os.environ.get("EVAL_OUTPUT_DIR", tempfile.gettempdir() + "/eval_output")
)

AGENT_CONFIG_HASH = os.environ.get("AGENT_CONFIG_HASH") or _compute_config_hash(
    _agent_config_dir
)

AGENT_AUTH_TOKEN = os.environ.get("AGENT_AUTH_TOKEN", "")
EVAL_MAX_CONCURRENCY = int(os.environ.get("EVAL_MAX_CONCURRENCY", "3"))
ALL_PATTERNS = ["tool_use", "structured_output", "hitl", "multi_agent"]

log.info(
    "eval_api config: agent_url=%s config_hash=%s "
    "eval_cases=%s eval_system=%s max_concurrency=%d",
    AGENT_URL,
    AGENT_CONFIG_HASH,
    EVAL_CASES_PATH,
    EVAL_SYSTEM_CONFIG,
    EVAL_MAX_CONCURRENCY,
)

# ── State ─────────────────────────────────────────────────────────────────────

_status: dict[str, Any] = {"state": "idle", "run_id": None}
_latest_result: dict[str, Any] | None = None


# ── Eval runner ───────────────────────────────────────────────────────────────


def _find_eval_files(pattern: str | None, cases: list[dict] | None) -> list[Path]:
    """Return one temp file per tag (parallel all) or one file for a specific pattern.

    Receives pre-fetched cases from fetch_dataset() — no additional DB call.
    Falls back to eval_cases.yaml when cases is None (dev-mode).

    When pattern is None every tag gets its own temp file so _run_eval can run
    them concurrently via asyncio.gather while keeping conversations sequential
    within each tag subprocess.
    """
    if cases:
        log.info("Loaded %d eval cases from Postgres", len(cases))
    else:
        # Dev-mode fallback: use config file if Postgres has no dataset
        if not EVAL_CASES_PATH.exists():
            raise FileNotFoundError(
                "No eval dataset found. Add eval cases via the Dataset UI "
                "or provide eval_cases.yaml in the agent config."
            )
        cases = load_cases(EVAL_CASES_PATH)
        log.info(
            "Postgres dataset empty — using config file fallback: %s (%d cases)",
            EVAL_CASES_PATH,
            len(cases),
        )

    if pattern is None:
        # Group cases by tag in one pass, preserving insertion order
        groups: dict[str, list] = defaultdict(list)
        for c in cases:
            tag = c.get("tag")
            if tag:
                groups[tag].append(c)
        if not groups:
            log.warning(
                "No tags found in Postgres eval cases — writing untagged cases to temp file"
            )
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", prefix="eval_untagged_", delete=False
            )
            yaml.dump(cases, tmp, default_flow_style=False, allow_unicode=True)
            tmp.close()
            return [Path(tmp.name)]
        log.info("Splitting cases into %d tag batches: %s", len(groups), list(groups))
        files: list[Path] = []
        for tag, filtered in groups.items():
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", prefix=f"eval_{tag}_", delete=False
            )
            yaml.dump(filtered, tmp, default_flow_style=False, allow_unicode=True)
            tmp.close()
            files.append(Path(tmp.name))
            log.debug("Tag '%s': %d case(s) → %s", tag, len(filtered), tmp.name)
        return files

    # Specific pattern — single filtered temp file
    filtered = [c for c in cases if c.get("tag") == pattern]
    if not filtered:
        raise FileNotFoundError(
            f"No eval cases found for pattern '{pattern}'. "
            "Add cases with this tag before running evaluation."
        )
    log.info("Pattern '%s': %d matching case(s)", pattern, len(filtered))
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix=f"eval_{pattern}_", delete=False
    )
    yaml.dump(filtered, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    return [Path(tmp.name)]


def _build_turn_metrics_metadata() -> dict:
    """Build metrics_metadata.turn_level from tag_metrics.yaml thresholds."""
    thresholds = get_metric_thresholds()

    def t(metric: str) -> dict:
        v = thresholds.get(metric)
        return {"threshold": v} if v is not None else {}

    return {
        "custom:answer_correctness": {
            **t("custom:answer_correctness"),
            "description": "Correctness vs expected answer",
        },
        "custom:tool_eval": {
            "description": "Tool call evaluation",
            "ordered": True,
            "full_match": True,
        },
        "custom:intent_eval": {
            **t("custom:intent_eval"),
            "description": "Intent alignment evaluation",
        },
        "custom:keywords_eval": {"description": "Keywords matching evaluation"},
        "geval:tone_safety": {
            "criteria": (
                "Assess whether the response uses a respectful, polite, and professional tone. "
                "Evaluate ONLY the tone — not the content, safety of information, or topic sensitivity. "
                "The response should be clear, supportive, and non-judgmental. "
                "It should not be dismissive, condescending, rude, alarmist, or offensive."
            ),
            "evaluation_params": ["query", "response"],
            "evaluation_steps": [
                "Check that the response does not use condescending, rude, or offensive language.",
                "Verify that the tone is warm, supportive, and non-alarmist.",
                "Check that the response is clear and professional in its wording.",
            ],
            **t("geval:tone_safety"),
            "description": "Tone is respectful, polite, and professional",
        },
        "ragas:faithfulness": {
            **t("ragas:faithfulness"),
            "description": "Response grounded in tool results / retrieved context",
        },
        "geval:delegation_compliance": {
            "criteria": (
                "Assess whether the orchestrator agent correctly delegated the task "
                "to the expected sub-agent(s) via tool calls. The delegation should "
                "match the expected tool calls in terms of which sub-agents were invoked "
                "and what arguments were passed. Evaluate ONLY whether the right "
                "sub-agents were called — not the quality of their responses."
            ),
            "evaluation_params": ["query", "response"],
            "evaluation_steps": [
                "Verify that the orchestrator invoked the expected sub-agent tool(s).",
                "Check that the arguments passed to each sub-agent match the expected values.",
                "Confirm no critical sub-agent delegation was omitted.",
            ],
            **t("geval:delegation_compliance"),
            "description": "Orchestrator delegated to the correct sub-agent(s)",
        },
    }


def _build_conversation_metrics_metadata() -> dict:
    """Build metrics_metadata.conversation_level from tag_metrics.yaml thresholds."""
    thresholds = get_metric_thresholds()
    result = {}
    conv_metrics = {
        "deepeval:knowledge_retention": "Knowledge retention across turns",
        "deepeval:conversation_completeness": "How completely the conversation addresses user intentions",
    }
    for metric, description in conv_metrics.items():
        entry: dict = {"description": description}
        v = thresholds.get(metric)
        if v is not None:
            entry["threshold"] = v
        result[metric] = entry
    return result


def _default_system_yaml(judge_model: str, provider: str) -> dict:
    """Return a minimal system.yaml config dict."""
    return {
        "core": {
            "max_threads": None,
            "fail_on_invalid_data": False,
            "skip_on_failure": False,
            "cache_enabled": True,
            "cache_base_dir": "/tmp/.eval_caches",
        },
        "llm": {"provider": provider, "model": judge_model, "max_tokens": 4096},
        "llm_pool": {
            "defaults": {
                "timeout": 300,
                "num_retries": 3,
                "parameters": {"temperature": 0.0, "max_completion_tokens": 4096},
            },
            "models": {"judge": {"provider": provider, "model": judge_model}},
        },
        "judge_panel": {"judges": ["judge"], "aggregation_strategy": "max"},
        "agents": {"enabled": False},
        "quality_score": {
            "metrics": [
                "custom:answer_correctness",
                "custom:tool_eval",
            ],
            "default": True,
            # ragas:faithfulness excluded — default:True would force it on every turn;
            # it is added per-turn only when tool calls are expected.
        },
        "metrics_metadata": {
            "turn_level": _build_turn_metrics_metadata(),
            "conversation_level": _build_conversation_metrics_metadata(),
        },
        "storage": [
            {
                "type": "postgres",
                "database": POSTGRES_DB,
                "table_name": "evaluation_results",
                "host": POSTGRES_HOST,
                "port": POSTGRES_PORT,
                "user": POSTGRES_USER,
                "password": POSTGRES_PASSWORD,
            }
        ],
        "environment": {
            "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
            "DEEPEVAL_DISABLE_PROGRESS_BAR": "YES",
            "LITELLM_LOG": "ERROR",
        },
        "logging": {"source_level": "INFO", "package_level": "ERROR"},
    }


def _detect_provider(model: str) -> str:
    m = model.lower()
    if any(m.startswith(p) for p in ("gemini-", "google/")):
        return "vertex"
    if m.startswith("claude-"):
        return "anthropic_vertex"
    if any(m.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    return "vertex"


def _read_prompt_model() -> str:
    """Read model: from PROMPT.md YAML frontmatter, defaulting to gemini-2.5-pro."""
    import re

    prompt_md = Path(_agent_config_dir) / "PROMPT.md"
    if prompt_md.exists():
        try:
            content = prompt_md.read_text(encoding="utf-8")
            m = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if m:
                fm = yaml.safe_load(m.group(1)) or {}
                return str(fm.get("model", "gemini-2.5-pro"))
        except Exception:
            pass
    return "gemini-2.5-pro"


# Module-level Postgres config for _default_system_yaml
from eval_postgres import (  # noqa: E402 — after constants are set
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)


def _get_system_yaml_content(judge_model: str | None = None) -> str:
    """Return system.yaml content with postgres credentials filled from env vars.

    If system.yaml is missing on disk, auto-generates a default using:
    1. judge_model passed in (fetched alongside dataset in one DB call)
    2. model: field from PROMPT.md frontmatter
    """
    if not EVAL_SYSTEM_CONFIG.exists():
        resolved_model = judge_model or _read_prompt_model()
        provider = _detect_provider(resolved_model)
        log.info(
            "system.yaml not found — generating default with provider=%s model=%s",
            provider,
            resolved_model,
        )
        return str(
            yaml.dump(
                _default_system_yaml(resolved_model, provider),
                default_flow_style=False,
                allow_unicode=True,
            )
        )

    config = yaml.safe_load(EVAL_SYSTEM_CONFIG.read_text())

    for backend in config.get("storage", []):
        if backend.get("type") == "postgres":
            backend["host"] = os.environ.get(
                "POSTGRES_HOST", backend.get("host", "localhost")
            )
            backend["port"] = int(
                os.environ.get("POSTGRES_PORT", str(backend.get("port", 5432)))
            )
            backend["database"] = os.environ.get(
                "POSTGRES_DB", backend.get("database", "template_agent")
            )
            backend["user"] = os.environ.get(
                "POSTGRES_USER", backend.get("user", "postgres")
            )
            backend["password"] = os.environ.get(
                "POSTGRES_PASSWORD", backend.get("password", "")
            )

    return str(yaml.dump(config, default_flow_style=False, allow_unicode=True))


def _system_yaml_path(judge_model: str | None = None) -> Path:
    """Write system.yaml with env-injected credentials to a temp file and return its path."""
    content = _get_system_yaml_content(judge_model)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="eval_system_", delete=False
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def _run_eval_pattern_sync(
    eval_file: Path, system_yaml: Path, output_dir: Path, auth_token: str = ""
) -> int:
    """Blocking subprocess call — must be run in a thread pool, not the event loop."""
    runner = Path(__file__).parent / "run_eval.py"
    env = dict(os.environ)
    if auth_token:
        env["AGENT_AUTH_TOKEN"] = auth_token  # user session token for MCP tool calls
    cmd = [
        sys.executable,
        str(runner),
        "--agent-url",
        AGENT_URL,
        "--eval-data",
        str(eval_file),
        "--system",
        str(system_yaml),
        "--output-dir",
        str(output_dir),
    ]
    log.info("Spawning subprocess: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, env=env)
    log.info(
        "Subprocess exited: file=%s exit_code=%d", eval_file.name, result.returncode
    )
    if result.returncode != 0:
        log.warning(
            "Non-zero exit for %s — check run_eval.py output above", eval_file.name
        )
    return result.returncode


async def _run_eval_pattern(
    eval_file: Path, system_yaml: Path, output_dir: Path, auth_token: str = ""
) -> int:
    """Run one eval pattern in a thread pool so the event loop stays responsive."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _run_eval_pattern_sync, eval_file, system_yaml, output_dir, auth_token
    )


def _score_from_counts(passed: int, failed: int, errors: int) -> tuple[str, float]:
    # Score is based on pass/(pass+fail) only — errors are excluded from the
    # denominator so a metric configuration error doesn't penalise the agent.
    scoreable = passed + failed
    if scoreable == 0:
        # All results are errors (or nothing ran)
        return "error", 0.0
    score = round(passed / scoreable, 3)
    if failed == 0 and errors == 0:
        status = "passed"
    elif errors > 0 and failed == 0 and passed == 0:
        status = "error"
    else:
        status = "failed"
    return status, score


async def _run_eval(
    pattern: str | None,
    config_hash: str | None = None,
    auth_token: str = "",
    run_id: str = "",
    eval_row_id: int | None = None,
) -> None:
    """Core eval runner — invoked in background."""
    global _latest_result

    log.info(
        "Eval run started: run_id=%s pattern=%s eval_row_id=%s auth_token_present=%s",
        run_id,
        pattern or "all",
        eval_row_id,
        bool(auth_token),
    )

    system_yaml: Path | None = None
    run_output: Path | None = None
    tmp_files: list[Path] = []
    try:
        try:
            loop = asyncio.get_running_loop()
            # Single DB call to get both dataset cases and judge_model
            cases, judge_model = await loop.run_in_executor(None, fetch_dataset)
            system_yaml = _system_yaml_path(judge_model)
            eval_files = await loop.run_in_executor(
                None, functools.partial(_find_eval_files, pattern, cases)
            )
            # Track temp files created by _find_eval_files (tag-filtered) for cleanup
            tmp_files = [f for f in eval_files if f.parent != EVAL_CASES_PATH.parent]
        except FileNotFoundError as exc:
            log.error("eval_setup_failed: %s", exc)
            _status.update({"state": "error", "run_id": run_id})
            write_eval_result(
                passed=0,
                failed=0,
                errors=1,
                eval_score=0.0,
                ls_run_ids=None,
                results_detail={"error": str(exc)},
                config_hash=config_hash,
                eval_row_id=eval_row_id,
            )
            return

        EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        run_output = EVAL_OUTPUT_DIR / run_id
        run_output.mkdir(parents=True, exist_ok=True)

        sem = asyncio.Semaphore(EVAL_MAX_CONCURRENCY)

        async def _run_one(eval_file: Path) -> int:
            async with sem:
                log.info("Running %s", eval_file.name)
                rc = await _run_eval_pattern(
                    eval_file, system_yaml, run_output, auth_token
                )
                log.info("%s → exit code %d", eval_file.name, rc)
                return rc

        run_started_at = datetime.now(timezone.utc)
        await asyncio.gather(*[_run_one(f) for f in eval_files])
    except Exception as exc:
        log.error("eval_run_failed: %s", exc)
        _status.update({"state": "error", "run_id": run_id})
        write_eval_result(
            passed=0,
            failed=0,
            errors=1,
            eval_score=0.0,
            ls_run_ids=None,
            results_detail={"error": str(exc)},
            config_hash=config_hash,
            eval_row_id=eval_row_id,
        )
        _call_cleanup_endpoint()
        return
    finally:
        # Always clean up temp files — even when _find_eval_files raises early
        for tmp in tmp_files:
            tmp.unlink(missing_ok=True)
        if system_yaml is not None:
            system_yaml.unlink(missing_ok=True)
        # Clean up eval output directory only if it was created
        if run_output is not None:
            try:
                import shutil

                shutil.rmtree(run_output, ignore_errors=True)
            except Exception:
                pass

    total_pass, total_fail, total_error = 0, 0, 0
    eval_status, eval_score = _score_from_counts(total_pass, total_fail, total_error)

    result: dict[str, Any] = {
        "run_id": run_id,
        "config_hash": AGENT_CONFIG_HASH,
        "eval_status": eval_status,
        "eval_score": eval_score,
        "pass": total_pass,
        "fail": total_fail,
        "error": total_error,
        "output_dir": str(run_output),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Build rich results_detail from the evaluation_results PostgreSQL table
    results_detail = dict(result)
    db_data = await asyncio.get_running_loop().run_in_executor(
        None, load_results_since, run_started_at
    )
    results_detail.update(db_data)
    if not db_data:
        log.warning(
            "No DB results found for run_id=%s — summary will show zeros", run_id
        )

    # Recompute scalars from DB summary — file storage was removed so
    # _aggregate_summaries() returned zeros; DB summary has the real counts.
    overall = (
        results_detail.get("summary", {}).get("summary_stats", {}).get("overall", {})
    )
    if overall:
        total_pass = int(overall.get("PASS", total_pass))
        total_fail = int(overall.get("FAIL", total_fail))
        total_error = int(overall.get("ERROR", total_error))
        eval_status, eval_score = _score_from_counts(
            total_pass, total_fail, total_error
        )
        result.update(
            {
                "eval_status": eval_status,
                "eval_score": eval_score,
                "pass": total_pass,
                "fail": total_fail,
                "error": total_error,
            }
        )
        results_detail.update(result)

    # Only expose result after DB recompute so _latest_result never shows
    # the pre-DB placeholder zeros or the always-"error" initial state.
    _latest_result = result
    _status.update({"state": "completed", "run_id": run_id})

    # Log after DB recompute so values are accurate
    log.info(
        "Eval complete: status=%s score=%.3f pass=%d fail=%d error=%d",
        eval_status,
        eval_score,
        total_pass,
        total_fail,
        total_error,
    )

    # Write results to Postgres so agentpod /evals/results can serve them
    try:
        write_eval_result(
            passed=total_pass,
            failed=total_fail,
            errors=total_error,
            eval_score=eval_score,
            ls_run_ids=results_detail.get("ls_run_ids"),
            results_detail=results_detail,
            config_hash=config_hash,
            eval_row_id=eval_row_id,
        )
    except Exception as exc:
        log.error(
            "postgres_write_failed (%s) — eval results not persisted (output dir already cleaned up)",
            type(exc).__name__,
        )

    _call_cleanup_endpoint()


# ── FastAPI app ────────────────────────────────────────────────────────────────
# eval_cases.yaml and system.yaml are pre-written to the PVC by agent-engine
# at deploy time — no startup auto-run or case management needed here.


from eval_postgres import (  # noqa: E402
    ensure_table,
    get_results_by_run_id,
    load_results_since,
    write_eval_result,
)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    await asyncio.get_running_loop().run_in_executor(None, ensure_table)
    yield


app = FastAPI(title="eval-runner", version="2.0.0", lifespan=_lifespan)

_SKIP_AUTH_PATHS = {"/health", "/metrics"}


@app.middleware("http")
async def _internal_token_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Validate X-Internal-Token on all endpoints except /health and /metrics."""
    if request.url.path in _SKIP_AUTH_PATHS:
        return await call_next(request)
    provided = request.headers.get("x-internal-token", "")
    if not hmac.compare_digest(provided.encode(), _EVAL_INTERNAL_TOKEN.encode()):
        return JSONResponse(
            status_code=401, content={"detail": "invalid internal token"}
        )
    return await call_next(request)


class EvalRunBody(BaseModel):
    """Request body for POST /evals/run."""

    config_hash: str | None = None
    eval_row_id: int | None = None


def _extract_token(request: Request) -> str:
    """Extract user auth token from Authorization header to forward to the agent.

    The eval runner is an internal service (NetworkPolicy/ClusterIP) — no auth
    enforcement on its own endpoints. Token is only used for agent subprocess calls.
    """
    auth_header = request.headers.get("authorization", "")
    token = (
        auth_header.removeprefix("Bearer ").strip()
        if auth_header.startswith("Bearer ")
        else ""
    )
    return token or AGENT_AUTH_TOKEN


async def _trigger(
    pattern: str | None,
    background: BackgroundTasks,
    body: EvalRunBody | None = None,
    auth_token: str = "",
) -> dict[str, Any]:
    if _status["state"] == "running":
        log.warning(
            "Eval trigger rejected — run %s already in progress", _status.get("run_id")
        )
        raise HTTPException(
            status_code=409, detail="An eval run is already in progress"
        )
    # File-existence guards removed: _find_eval_files queries Postgres first and
    # _get_system_yaml_content auto-generates system.yaml when absent on disk.
    # Missing-dataset errors surface via FileNotFoundError from _find_eval_files.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    config_hash = body.config_hash if body else None
    eval_row_id = body.eval_row_id if body else None
    log.info(
        "Eval triggered: run_id=%s pattern=%s config_hash=%s eval_row_id=%s auth_token_present=%s",
        run_id,
        pattern or "all",
        config_hash,
        eval_row_id,
        bool(auth_token),
    )
    _status.update({"state": "running", "run_id": run_id})
    background.add_task(
        _run_eval, pattern, config_hash, auth_token, run_id, eval_row_id
    )
    return {
        "run_id": run_id,
        "status": "started",
        "pattern": pattern or "all",
    }


@app.post("/evals/run", status_code=202)
async def run_all(
    request: Request, background: BackgroundTasks, body: EvalRunBody = EvalRunBody()
) -> dict[str, Any]:
    """Run all eval patterns against the agent."""
    return await _trigger(None, background, body, _extract_token(request))


@app.post("/evals/run/{pattern}", status_code=202)
async def run_pattern(
    pattern: str, request: Request, background: BackgroundTasks
) -> dict[str, Any]:
    """Run one eval pattern (tool_use / hitl / structured_output / multi_agent)."""
    if pattern not in ALL_PATTERNS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pattern '{pattern}'. Valid: {ALL_PATTERNS}",
        )
    return await _trigger(pattern, background, auth_token=_extract_token(request))


@app.get("/evals/status")
async def get_status() -> dict[str, Any]:
    """Return current run state: idle | running | completed | error."""
    return _status


@app.get("/evals/results")
async def get_latest_results() -> JSONResponse:
    """Return the latest run summary (in-memory)."""
    if _latest_result is None:
        raise HTTPException(status_code=404, detail="No eval results available yet")
    return JSONResponse(_latest_result)


@app.get("/evals/results/{run_id}")
async def get_run_results(run_id: str) -> JSONResponse:
    """Return results for a specific run ID from the evaluation_results Postgres table."""
    try:
        rows = await asyncio.get_running_loop().run_in_executor(
            None, get_results_by_run_id, run_id
        )
    except Exception as exc:
        log.error(
            "get_run_results failed for run_id=%s (%s)", run_id, type(exc).__name__
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve eval results"
        ) from exc
    if not rows:
        raise HTTPException(
            status_code=404, detail=f"No results found for run '{run_id}'"
        )
    return JSONResponse({"run_id": run_id, "results": rows})


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — always returns ok."""
    return {"status": "ok"}


@app.post("/mcp")
async def mcp_stub() -> dict[str, str]:
    """Stub to suppress 404 noise from lightspeed-eval MCP probe on startup."""
    return {"error": "MCP not supported on eval runner"}
