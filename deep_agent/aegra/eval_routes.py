"""Eval endpoint for the template agent.

POST /v1/eval/run
    Runs the agent via the LangGraph Platform's internal thread API (self-call).
    Returns the full final state — tool calls at every depth including MCP tools
    called inside subagents (calculate_bmi, validate_email, send_email, etc.).
    HITL interrupts are auto-approved when auto_approve=True.

Uses the Platform's own /threads + /runs endpoints so auth context, graph cache,
and checkpointer all work correctly without side effects on other requests.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

_bearer = HTTPBearer(auto_error=False)


def _require_bearer(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """Reject requests that carry no Bearer token.

    Both /v1/eval/run and /v1/eval/thread-tool-calls/{thread_id} invoke the
    production agent or read checkpoint blobs that may contain user PII.
    Callers (run_eval.py subprocess) always forward the session token; any
    request without one is unauthenticated and must be rejected.
    """
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Bearer token required")
    return str(creds.credentials)


async def _check_mcp_auth(request: Request) -> list[dict]:
    """Pre-flight check: return list of MCP servers that need authentication.

    Checks servers where enabled=true, auth=true, auth_mode in ("dcr", "oauth").
    For each:
      A) checks Redis token exists
      B) does a lightweight ping to validate the token is still accepted

    Returns list of dicts with {name, connect_url} for servers needing auth.
    Empty list means all required auth is in place.
    """
    from deep_agent.aegra.mcp_token_store import McpTokenStore
    from deep_agent.src.agent.config import agent_config
    from deep_agent.src.settings import settings as _settings

    user_id: str = _extract_sub(request) or ""
    log.info("mcp_auth_check: user_id=%r", user_id)
    if not user_id:
        return []  # no user identity — can't check per-user tokens

    mcp_servers: dict = agent_config.get_mcp_servers()
    # Must match settings.agent_deployment_id used by mcp_oauth_handlers when storing tokens
    agent_name: str = _settings.agent_deployment_id
    log.info("mcp_auth_check: agent_name=%r", agent_name)

    dcr_servers = {
        name: cfg
        for name, cfg in mcp_servers.items()
        if cfg.get("enabled", True)
        and cfg.get("auth", False)
        and cfg.get("auth_mode", "") in ("dcr", "oauth")
    }
    log.info("mcp_auth_check: dcr_servers=%s", list(dcr_servers.keys()))

    if not dcr_servers:
        return []

    store = McpTokenStore(_settings.database_uri)
    missing: list[dict] = []

    for name, cfg in dcr_servers.items():
        token = await store.get_token(agent_name, user_id, name)
        needs_auth = token is None
        log.info("mcp_auth_check: mcp=%r token_found=%s", name, token is not None)

        # B) Token expiry check — re-auth if expired OR expiring within the
        # eval minimum TTL window so the token doesn't expire mid-run.
        if not needs_auth and token:
            from datetime import timedelta, timezone

            min_ttl = timedelta(minutes=_EVAL_TOKEN_MIN_TTL_MINUTES)
            if (
                token.expires_at
                and token.expires_at < datetime.now(timezone.utc) + min_ttl
            ):
                log.info(
                    "mcp_auth_check: mcp=%r token expires at %s (within %d-min buffer)",
                    name,
                    token.expires_at,
                    _EVAL_TOKEN_MIN_TTL_MINUTES,
                )
                needs_auth = True

        if needs_auth:
            missing.append(
                {
                    "name": name,
                    "connect_url": f"/mcp/{name}/connect",
                }
            )

    log.info("mcp_auth_check: missing=%s", [m["name"] for m in missing])
    return missing


# Minimum token TTL required before starting an eval. If the token expires
# sooner than this, the user is asked to re-authenticate upfront rather than
# having the MCP call fail mid-run.
_EVAL_TOKEN_MIN_TTL_MINUTES: int = int(
    os.environ.get("EVAL_TOKEN_MIN_TTL_MINUTES", "30")
)

# ── Eval token refresh helpers (feature-flagged) ──────────────────────────────
_EVAL_TOKEN_REFRESH_ENABLED: bool = (
    os.environ.get("EVAL_TOKEN_REFRESH_ENABLED", "false").lower() == "true"
)
_EVAL_INTERNAL_TOKEN: str = os.environ.get("EVAL_INTERNAL_TOKEN", "")
_EVAL_KEY_TTL = 3600  # 60-min safety-net; explicit cleanup via /evals/internal/cleanup


def _extract_sub(request: Request) -> str | None:
    """Extract sub from the Bearer JWT in the request without signature verification."""
    from deep_agent.aegra.auth import _decode_sub_unverified

    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return _decode_sub_unverified(auth[7:])


def _write_eval_redis(sub: str, refresh_token: str) -> None:
    """Write the three eval Redis keys at trigger time. Best-effort."""
    if not _EVAL_TOKEN_REFRESH_ENABLED or not sub:
        return
    try:
        from deep_agent.aegra.mcp_crypto import encrypt_secret
        from deep_agent.aegra.redis import cache_set

        cache_set(f"eval:active:{sub}", "1", _EVAL_KEY_TTL)
        if refresh_token:
            encrypted = encrypt_secret(refresh_token)
            if encrypted:
                cache_set(f"eval:refresh:{sub}", encrypted, _EVAL_KEY_TTL)
        cache_set("eval:trigger_sub", sub, _EVAL_KEY_TTL)
    except Exception:
        pass  # never block the eval trigger


_EVAL_RUNNER_URL = os.environ.get("EVAL_RUNNER_URL", "")


# SYNC: this algorithm must stay identical to _compute_config_hash() in
# eval-runner/src/eval_postgres.py — both services hash the same config dir
# and share the result via the `evals` Postgres table. Any change here must
# be mirrored there (same extensions, same exclude dirs, same truncation).
_HASH_EXTENSIONS = {".md", ".yaml", ".json"}
_HASH_EXCLUDE_DIRS = {"evals", "deployment"}


def _compute_config_hash() -> str:
    """SHA256 of behavior-relevant config files (prompts, skills, runtime, tools)."""
    config_dir = Path(os.environ.get("AGENT_CONFIG_DIR", "config/agent"))
    h = hashlib.sha256()
    if config_dir.exists():
        for fpath in sorted(config_dir.rglob("*")):
            if not fpath.is_file():
                continue
            if fpath.suffix not in _HASH_EXTENSIONS:
                continue
            if any(
                part in _HASH_EXCLUDE_DIRS
                for part in fpath.relative_to(config_dir).parts
            ):
                continue
            h.update(str(fpath.relative_to(config_dir)).encode())
            h.update(fpath.read_bytes())
    return h.hexdigest()[:16]


log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/eval", tags=["eval"])

_INTERNAL_TOOLS = {
    "write_todos",
    "task",  # framework internals
    "read_file",
    "write_file",  # filesystem
    "ls",
    "grep",
    "glob",  # filesystem
    "execute_command",  # shell
    "compact_conversation",  # memory management
}
MAX_HITL_APPROVALS = 50
AGENT_BASE_URL = "http://localhost:5002"
ASSISTANT_ID = "agent"


class EvalRunRequest(BaseModel):
    """Request body for a single eval run turn."""

    query: str
    conversation_id: str | None = None
    auto_approve: bool = True


class EvalRunResponse(BaseModel):
    """Response from a single eval run turn."""

    response: str
    pre_approval_response: str | None
    was_interrupted: bool
    tool_calls: list[dict[str, Any]]
    contexts: list[str]
    conversation_id: str


def _extract_from_messages(messages: list[dict]) -> tuple[str, list[dict], list[str]]:
    """Extract response, MCP tool calls, and contexts from state messages (plain dicts)."""
    final_response = ""
    tool_calls: list[dict[str, Any]] = []
    contexts: list[str] = []

    for msg in messages:
        msg_type = msg.get("type", "")

        if msg_type == "ai":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                final_response = content.strip()
            elif isinstance(content, list):
                text = " ".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ).strip()
                if text:
                    final_response = text

            for tc in msg.get("tool_calls", []):
                name = tc.get("name", "")
                if name and name not in _INTERNAL_TOOLS:
                    tool_calls.append(
                        {
                            "tool_name": name,
                            "arguments": tc.get("args", {}),
                        }
                    )

        elif msg_type == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                contexts.append(content.strip())

    return final_response, tool_calls, contexts


async def _create_thread(client: httpx.AsyncClient) -> str:
    resp = await client.post(f"{AGENT_BASE_URL}/threads", json={})
    resp.raise_for_status()
    return resp.json()["thread_id"]  # type: ignore[no-any-return]


async def _run_and_get_state(
    client: httpx.AsyncClient,
    thread_id: str,
    body: dict,
    timeout: float = 120.0,
) -> dict:
    """/runs/wait — returns state with messages at top level + __interrupt__ at top level."""
    resp = await client.post(
        f"{AGENT_BASE_URL}/threads/{thread_id}/runs/wait",
        json={**body, "stream_mode": "values"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def _extract_tool_calls_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Extract non-internal tool calls from a list of LangChain messages."""
    tool_calls: list[dict[str, Any]] = []
    for msg in messages:
        msg_dict = (
            msg
            if isinstance(msg, dict)
            else (msg.__dict__ if hasattr(msg, "__dict__") else {})
        )
        if msg_dict.get("type") != "ai":
            continue
        # Format 1: tool_calls list (newer LangChain)
        for tc in msg_dict.get("tool_calls", []):
            name = (
                tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            )
            args = (
                tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
            )
            if name and name not in _INTERNAL_TOOLS:
                tool_calls.append({"tool_name": name, "arguments": args})
        # Format 2: additional_kwargs.function_call (Gemini format)
        if not tool_calls or not any(
            tc["tool_name"] not in _INTERNAL_TOOLS for tc in tool_calls
        ):
            fc = msg_dict.get("additional_kwargs", {}).get("function_call")
            if fc:
                name = fc.get("name", "")
                if name and name not in _INTERNAL_TOOLS:
                    args_raw = fc.get("arguments", "{}")
                    try:
                        args = (
                            json.loads(args_raw)
                            if isinstance(args_raw, str)
                            else args_raw
                        )
                    except Exception:
                        args = {}
                    tool_calls.append({"tool_name": name, "arguments": args})
    return tool_calls


async def _collect_subagent_tool_calls_via_remote_graph(
    thread_id: str,
) -> list[dict[str, Any]]:
    """Fetch subagent tool calls via RemoteGraph.get_state(subgraphs=True).

    This uses the LangGraph SDK's official API to retrieve all subgraph states,
    recursively collecting tool calls from every subagent namespace.
    """
    from langgraph.pregel.remote import RemoteGraph

    tool_calls: list[dict[str, Any]] = []

    try:
        graph = RemoteGraph(ASSISTANT_ID, url=AGENT_BASE_URL)
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(config, subgraphs=True)

        def _collect(state_snapshot: Any) -> None:
            tasks = getattr(state_snapshot, "tasks", [])
            for task in tasks:
                substate = getattr(task, "state", None)
                if substate is None:
                    continue
                msgs = getattr(substate, "values", {})
                if isinstance(msgs, dict):
                    msgs = msgs.get("messages", [])
                tool_calls.extend(_extract_tool_calls_from_messages(msgs))
                _collect(substate)

        _collect(snapshot)
    except Exception as exc:
        log.warning(
            "RemoteGraph subgraph fetch failed: %s — falling back to Postgres", exc
        )
        return await _collect_subagent_tool_calls_from_postgres(thread_id)

    return tool_calls


async def _collect_subagent_tool_calls_from_postgres(
    thread_id: str,
) -> list[dict[str, Any]]:
    """Fallback: read subagent tool calls directly from Postgres checkpoint store.

    Blob format (LangGraph checkpoint serde):
        list of [module_path, class_name, message_dict] per message.

    The Gemini thought-signature fields use msgpack ext types — the ext_hook
    returns raw bytes for unknown types instead of raising, so those fields are
    ignored without aborting the entire blob parse.
    """
    import msgpack
    import psycopg

    from deep_agent.src.settings import settings

    def _ext_hook(code: int, data: bytes) -> Any:
        # Return raw bytes for unknown ext types instead of raising —
        # prevents Gemini thought-signature binary payloads from aborting parse.
        try:
            return msgpack.unpackb(data, raw=False, ext_hook=_ext_hook)
        except Exception:
            return data

    tool_calls: list[dict[str, Any]] = []

    try:
        async with await psycopg.AsyncConnection.connect(settings.database_uri) as conn:
            rows = await conn.execute(
                """
                SELECT DISTINCT ON (checkpoint_ns) blob
                FROM checkpoint_blobs
                WHERE thread_id = %s
                  AND checkpoint_ns != ''
                  AND channel = 'messages'
                ORDER BY checkpoint_ns, version DESC
                """,
                (thread_id,),
            )
            async for (blob,) in rows:
                if blob is None:
                    continue
                try:
                    items = msgpack.unpackb(bytes(blob), raw=False, ext_hook=_ext_hook)
                    if not isinstance(items, (list, tuple)):
                        continue
                    for item in items:
                        # Each item: [module_path, class_name, message_dict]
                        if not isinstance(item, (list, tuple)) or len(item) < 3:
                            continue
                        msg = item[2]
                        if not isinstance(msg, dict) or msg.get("type") != "ai":
                            continue
                        tool_calls.extend(_extract_tool_calls_from_messages([msg]))
                except Exception as exc:
                    log.debug("Skipping blob for %s: %s", thread_id, exc)
                    continue
    except Exception as exc:
        log.warning("Postgres subagent tool call extraction failed: %s", exc)

    return tool_calls


def _messages_from_run(run_state: dict) -> list[dict]:
    """Extract messages from /runs/wait response (top-level) or thread state (under values)."""
    if "messages" in run_state:
        return run_state["messages"]  # type: ignore[no-any-return]
    return run_state.get("values", {}).get("messages", [])  # type: ignore[no-any-return]


def _detect_interrupt(run_state: dict) -> bool:
    """Return True if the run state has a pending HITL interrupt."""
    if run_state.get("__interrupt__"):
        return True
    next_nodes = run_state.get("next", [])
    values = run_state.get("values", {})
    return bool(values.get("__interrupt__")) or any(
        "__interrupt__" in str(n) for n in next_nodes
    )


@router.get("/thread-tool-calls/{thread_id}")
async def get_thread_tool_calls(
    thread_id: str,
    request: Request,
    _token: str = Depends(_require_bearer),
) -> dict[str, Any]:
    """Return all subagent tool calls for a completed thread.

    Reads directly from Postgres checkpoint_blobs across all subagent namespaces
    since LangGraph's HTTP API only exposes subgraph state during interrupts.
    """
    await _verify_thread_ownership(thread_id, request)
    tool_calls = await _collect_subagent_tool_calls_from_postgres(thread_id)
    return {"thread_id": thread_id, "tool_calls": tool_calls}


async def _verify_thread_ownership(thread_id: str, request: Request) -> None:
    """Raise 403 unless the authenticated user owns the thread."""
    sub = _extract_sub(request)
    if not sub:
        raise HTTPException(
            status_code=403, detail="Could not identify user from token"
        )

    try:
        async with await _pg_conn() as conn:
            row = await conn.execute(
                "SELECT 1 FROM thread WHERE thread_id = %s AND user_id = %s",
                (thread_id, sub),
            )
            if await row.fetchone() is None:
                raise HTTPException(
                    status_code=403, detail="Thread not owned by authenticated user"
                )
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("Thread ownership check failed: %s", exc)
        raise HTTPException(
            status_code=403, detail="Could not verify thread ownership"
        ) from exc


@router.post("/run", response_model=EvalRunResponse)
async def eval_run(
    body: EvalRunRequest,
    _token: str = Depends(_require_bearer),
) -> EvalRunResponse:
    """Run one conversation turn for eval purposes.

    Returns ALL tool calls from the full state including MCP tools called
    inside subagents (calculate_bmi, validate_email, send_email, etc.).
    HITL interrupts are auto-approved when auto_approve=True.
    """
    conversation_id = body.conversation_id or str(uuid.uuid4())
    pre_approval_response: str | None = None
    was_interrupted = False

    async with httpx.AsyncClient(timeout=120) as client:
        thread_id = await _create_thread(client)

        run_body = {
            "assistant_id": ASSISTANT_ID,
            "input": {"messages": [{"role": "human", "content": body.query}]},
        }

        # Initial run — wait for completion or interrupt
        run_state = await _run_and_get_state(client, thread_id, run_body)
        messages = _messages_from_run(run_state)

        if _detect_interrupt(run_state):
            was_interrupted = True
            # Capture what the agent said when it paused for approval
            for msg in reversed(messages):
                if msg.get("type") == "ai":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        pre_approval_response = content.strip()
                        break

            if body.auto_approve:
                for _ in range(MAX_HITL_APPROVALS):
                    log.info("Auto-approving HITL interrupt for thread %s", thread_id)
                    approve_body = {
                        "assistant_id": ASSISTANT_ID,
                        "command": {"resume": {"decisions": [{"type": "approve"}]}},
                    }
                    run_state = await _run_and_get_state(
                        client, thread_id, approve_body
                    )
                    if not _detect_interrupt(run_state):
                        break
                else:
                    log.warning(
                        "HITL approval limit reached for thread %s — run may be incomplete",
                        thread_id,
                    )

                messages = _messages_from_run(run_state)

        # Collect subagent tool calls via RemoteGraph (falls back to Postgres on failure)
        subagent_tool_calls = await _collect_subagent_tool_calls_via_remote_graph(
            thread_id
        )

    final_response, orchestrator_tool_calls, contexts = _extract_from_messages(messages)
    # Merge: orchestrator-level tool calls first, then all subagent tool calls
    tool_calls = orchestrator_tool_calls + subagent_tool_calls

    log.info(
        "Eval run complete: thread=%s tools=%d response_len=%d interrupted=%s",
        thread_id,
        len(tool_calls),
        len(final_response),
        was_interrupted,
    )

    return EvalRunResponse(
        response=final_response,
        pre_approval_response=pre_approval_response,
        was_interrupted=was_interrupted,
        tool_calls=tool_calls,
        contexts=contexts,
        conversation_id=conversation_id,
    )


# ---------------------------------------------------------------------------
# Eval management routes — trigger / status / results
# Called by Template UI via agentpod (cache check + atomic in_progress guard)
# ---------------------------------------------------------------------------

eval_mgmt_router = APIRouter(prefix="/evals", tags=["eval-management"])

_EVALS_DDL_STATEMENTS = [
    """
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
    )
    """,
    "CREATE INDEX IF NOT EXISTS evals_config_hash ON evals (config_hash)",
    "CREATE INDEX IF NOT EXISTS evals_status ON evals (eval_status)",
    "ALTER TABLE evals ADD COLUMN IF NOT EXISTS ls_run_ids TEXT[]",
    "ALTER TABLE evals DROP COLUMN IF EXISTS ls_run_id",
    "DROP INDEX IF EXISTS evals_ls_run_id",
]


async def _pg_conn() -> Any:
    import psycopg

    from deep_agent.src.settings import settings

    return await psycopg.AsyncConnection.connect(settings.database_uri)


async def _ensure_evals_table() -> None:
    import psycopg.errors

    conn = await _pg_conn()
    try:
        await conn.set_autocommit(True)
        await conn.execute("SET statement_timeout = '5s'")
        for stmt in _EVALS_DDL_STATEMENTS:
            try:
                await conn.execute(stmt)
            except psycopg.errors.UniqueViolation:
                # Concurrent workers racing to create the same index/object — the
                # object now exists, which is the desired state. Safe to continue.
                log.debug("DDL skipped (already exists): %.120s", stmt.strip())
            except Exception as exc:
                log.warning(
                    "evals DDL statement failed: %s | stmt: %.120s", exc, stmt.strip()
                )
                raise
    finally:
        await conn.close()


_table_ensured = False


async def _ensure_evals_table_once() -> None:
    global _table_ensured
    if _table_ensured:
        return
    await _ensure_evals_table()
    _table_ensured = True


async def _run_ddl_once(
    ddl: str,
    migrations: list[str],
    flag_attr: str,
    label: str,
) -> None:
    """Generic once-only DDL runner. Updates the module-level bool named flag_attr.

    The flag is only set on SUCCESS so that transient failures can be retried.
    set_autocommit is inside the try/finally so the connection is always closed.
    """
    import sys as _sys

    mod = _sys.modules[__name__]
    if getattr(mod, flag_attr):
        return
    conn = await _pg_conn()
    try:
        await conn.set_autocommit(True)
        await conn.execute("SET statement_timeout = '5s'")
        await conn.execute(ddl)
        for stmt in migrations:
            try:
                await conn.execute(stmt)
            except Exception as exc:
                log.warning("%s migration skipped (may already exist): %s", label, exc)
        setattr(mod, flag_attr, True)  # only on success
    except Exception as exc:
        log.warning("%s DDL failed: %s", label, exc)
        raise  # propagate so callers know the table is not ready
    finally:
        await conn.close()


def _pg_row_to_dict(row: Any, cursor: Any) -> dict[str, Any]:
    return dict(zip([d.name for d in cursor.description], row))


async def _atomic_set_in_progress(
    config_hash: str, force: bool
) -> tuple[dict | None, bool]:
    """Set eval to in_progress in PostgreSQL.

    Returns (doc, is_new):
      - (None, False)         → in_progress exists on force; don't interrupt
      - (existing_doc, False) → already in_progress on normal trigger
      - (new_doc, True)       → fresh row inserted; start eval pod
    """
    await _ensure_evals_table_once()
    now = datetime.now(UTC)

    async with await _pg_conn() as conn:
        # Check for an existing in_progress run first (cheap read before locking).
        row = await conn.execute(
            "SELECT * FROM evals WHERE config_hash=%s "
            "AND eval_status='in_progress' ORDER BY created_at DESC LIMIT 1",
            (config_hash,),
        )
        existing = await row.fetchone()
        if existing is not None:
            if force:
                return None, False
            return _pg_row_to_dict(existing, row), False

        # Atomic insert — only succeeds if no in_progress row exists at insert
        # time, closing the SELECT+INSERT TOCTOU window when two pods race.
        cur = await conn.execute(
            """INSERT INTO evals
               (config_hash, eval_status, eval_score, pass, fail, error,
                results_detail, force_reeval, created_at, updated_at, completed_at)
               SELECT %(config_hash)s,'in_progress',
                      NULL,0,0,0,NULL,%(force)s,%(now)s,%(now)s,NULL
               WHERE NOT EXISTS (
                   SELECT 1 FROM evals
                   WHERE config_hash=%(config_hash)s
                   AND eval_status='in_progress'
               )
               RETURNING *""",
            {
                "config_hash": config_hash,
                "force": force,
                "now": now,
            },
        )
        new_row = await cur.fetchone()
        if new_row is None:
            # Another pod inserted in_progress between our SELECT and this INSERT
            row2 = await conn.execute(
                "SELECT * FROM evals WHERE config_hash=%s "
                "AND eval_status='in_progress' ORDER BY created_at DESC LIMIT 1",
                (config_hash,),
            )
            existing2 = await row2.fetchone()
            if existing2 is not None:
                return _pg_row_to_dict(existing2, row2), False
            return None, False
        doc = _pg_row_to_dict(new_row, cur)
        return doc, True


def _get_config_hash() -> str:
    """Get config hash from env var or compute from config dir."""
    env_hash = os.environ.get("AGENT_CONFIG_HASH")
    if env_hash:
        return env_hash
    return _compute_config_hash()


async def _mark_eval_error(config_hash: str, reason: str, created_at: datetime) -> None:
    """Set the specific in_progress eval record to error.

    Uses created_at to target the exact row so a later trigger's fresh
    in_progress record is never accidentally marked as error.
    """
    try:
        async with await _pg_conn() as conn:
            await conn.execute(
                "UPDATE evals SET eval_status='error', completed_at=NOW(), updated_at=NOW(), "
                "results_detail=%s::jsonb "
                "WHERE config_hash=%s "
                "AND eval_status='in_progress' AND created_at=%s",
                (
                    json.dumps({"error": reason}),
                    config_hash,
                    created_at,
                ),
            )
    except Exception as pg_exc:
        log.warning("Could not mark eval as error: %s", pg_exc)


async def _fire_eval_run(
    config_hash: str,
    auth_token: str = "",
    created_at: datetime | None = None,
    eval_row_id: int | None = None,
) -> str | None:
    """Call the eval runner pod.

    Returns None on success, or an error message string on failure.
    The caller is responsible for rolling back the in_progress row when an
    error is returned — this function no longer writes to Postgres so the
    trigger endpoint can surface the failure directly to the UI.
    """
    if not _EVAL_RUNNER_URL:
        msg = "EVAL_RUNNER_URL not set — eval runner is not deployed"
        log.warning(msg)
        return msg
    try:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = (
                auth_token
                if auth_token.startswith("Bearer ")
                else f"Bearer {auth_token}"
            )
        if _EVAL_INTERNAL_TOKEN:
            headers["X-Internal-Token"] = _EVAL_INTERNAL_TOKEN
        body: dict[str, Any] = {"config_hash": config_hash}
        if eval_row_id is not None:
            body["eval_row_id"] = eval_row_id
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{_EVAL_RUNNER_URL}/evals/run",
                json=body,
                headers=headers,
            )
            if resp.status_code >= 400:
                msg = f"Eval runner returned {resp.status_code}"
                log.warning("eval_runner_error status=%d", resp.status_code)
                return msg
            log.info("eval_runner_called status=%s", resp.status_code)
            return None
    except Exception as exc:
        msg = f"Could not reach eval runner at {_EVAL_RUNNER_URL}: {exc}"
        log.warning("eval_runner_call_failed: %s", msg)
        return msg


_DATASETS_DDL = """
    CREATE TABLE IF NOT EXISTS eval_datasets (
        id          SERIAL PRIMARY KEY,
        dataset     JSONB NOT NULL,
        judge_model TEXT,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
"""
# Migration for tables created before judge_model was added
_DATASETS_DDL_MIGRATIONS = [
    "ALTER TABLE eval_datasets ADD COLUMN IF NOT EXISTS judge_model TEXT",
]

_datasets_table_ensured = False


async def _ensure_datasets_table_once() -> None:
    await _run_ddl_once(
        _DATASETS_DDL,
        _DATASETS_DDL_MIGRATIONS,
        "_datasets_table_ensured",
        "eval_datasets",
    )


async def _has_postgres_dataset() -> bool:
    """Return True if eval_datasets has a row with at least one case."""
    try:
        await _ensure_datasets_table_once()
        async with await _pg_conn() as conn:
            row = await conn.execute(
                "SELECT 1 FROM eval_datasets "
                "WHERE jsonb_array_length(dataset->'cases') > 0 LIMIT 1",
            )
            return await row.fetchone() is not None
    except Exception as exc:
        log.warning("Could not check Postgres dataset: %s", exc)
        return False


async def _require_eval_files() -> None:
    """Validate that eval can run, then let the eval runner handle all file I/O.

    The agentpod no longer writes any files to the config volume.
    The eval runner pulls the dataset from Postgres directly and writes to a
    temporary file scoped to the eval run.

    eval_cases: Postgres must have a dataset, OR dev-mode config file fallback.
    system.yaml: eval runner handles generation/selection — agentpod does not touch it.
    """
    from fastapi import HTTPException

    has_pg = await _has_postgres_dataset()
    if not has_pg:
        is_dev = os.environ.get("ENVIRONMENT", "development").lower() == "development"
        config_dir = os.environ.get("CONFIG_PATH", "config/agent")
        eval_cases = Path(f"{config_dir}/evals/lightspeed/eval_cases.yaml")
        if is_dev and eval_cases.exists():
            log.info(
                "Dev mode: no Postgres dataset — eval runner will use eval_cases.yaml from config. "
                "Save cases via the Dataset UI to switch to Postgres-managed evals."
            )
        else:
            config_hint = (
                " A config file (eval_cases.yaml) exists but is ignored in production — "
                "save your dataset via the Dataset UI."
                if eval_cases.exists()
                else ""
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    "No eval dataset saved. Add or import test cases via the Dataset UI "
                    f"before running evaluation.{config_hint}"
                ),
            )


@eval_mgmt_router.post("/trigger", response_model=None)
async def trigger_eval(request: Request) -> Any:
    """Cache-first eval trigger. Returns cached result or sets in_progress."""
    if not _EVAL_RUNNER_URL:
        raise HTTPException(
            status_code=503,
            detail="EVAL_RUNNER_URL not set — eval runner is not deployed",
        )

    await _require_eval_files()

    config_hash = _get_config_hash()

    await _ensure_evals_table_once()
    async with await _pg_conn() as conn:
        row = await conn.execute(
            "SELECT * FROM evals WHERE config_hash=%s "
            "AND eval_status='completed' AND completed_at > NOW() - INTERVAL '24 hours' "
            "ORDER BY completed_at DESC LIMIT 1",
            (config_hash,),
        )
        existing = await row.fetchone()
        if existing:
            doc = _pg_row_to_dict(existing, row)
            # Bypass the cache when the dataset source has changed since the last eval:
            # 1. Dataset was updated after the cached eval completed (edit scenario).
            # 2. Dataset was cleared from Postgres (the data source switched to config
            #    file — the cached result is stale regardless of config_hash match).
            try:
                ds_row = await conn.execute(
                    "SELECT created_at FROM eval_datasets LIMIT 1",
                )
                ds = await ds_row.fetchone()
                if ds is None:
                    # Postgres dataset cleared — always run fresh so config file is used
                    log.info(
                        "eval_datasets is empty — bypassing cache to pick up config file"
                    )
                    existing = None
                elif ds[0] and doc.get("completed_at") and ds[0] > doc["completed_at"]:
                    log.info(
                        "Dataset updated after last eval (dataset=%s eval=%s) — bypassing cache",
                        ds[0].isoformat(),
                        doc["completed_at"],
                    )
                    existing = None
            except Exception as exc:
                log.warning("Could not check dataset recency for cache bypass: %s", exc)

        if existing:
            doc = _pg_row_to_dict(existing, row)
            doc.pop("id", None)
            return {**doc, "cached": True}

    # Pre-flight checks before inserting in_progress row — failures here surface
    # immediately to the UI without leaving a stuck record in Postgres.
    missing_auth = await _check_mcp_auth(request)
    if missing_auth:
        return JSONResponse(
            status_code=403,
            content={
                "message": "Connect required MCP servers before running eval",
                "auth_required": missing_auth,
            },
        )

    record, is_new = await _atomic_set_in_progress(config_hash, force=False)
    if not is_new:
        return {"eval_status": "in_progress", "message": "evaluation already running"}
    return await _queue_eval_run(config_hash, request, record)


@eval_mgmt_router.post("/force-trigger", response_model=None)
async def force_trigger_eval(request: Request) -> Any:
    """Force a fresh eval run, bypassing cache."""
    if not _EVAL_RUNNER_URL:
        raise HTTPException(
            status_code=503,
            detail="EVAL_RUNNER_URL not set — eval runner is not deployed",
        )

    missing_auth = await _check_mcp_auth(request)
    if missing_auth:
        return JSONResponse(
            status_code=403,
            content={
                "message": "Connect required MCP servers before running eval",
                "auth_required": missing_auth,
            },
        )

    await _require_eval_files()

    config_hash = _get_config_hash()

    record, is_new = await _atomic_set_in_progress(config_hash, force=True)
    if not is_new:
        return {"eval_status": "in_progress", "message": "evaluation already running"}
    return await _queue_eval_run(config_hash, request, record, forced=True)


async def _queue_eval_run(
    config_hash: str,
    request: Request,
    record: dict | None,
    *,
    forced: bool = False,
) -> dict[str, Any]:
    auth_token = request.headers.get("authorization", "")
    sub = _extract_sub(request)
    _write_eval_redis(sub or "", request.headers.get("x-refresh-token", ""))
    row_ts = record.get("created_at") if record else None
    eval_row_id: int | None = record.get("id") if record else None

    error_msg = await _fire_eval_run(config_hash, auth_token, row_ts, eval_row_id)
    if error_msg:
        # Roll back the in_progress row so the UI gets a clean error on trigger,
        # not a stuck in_progress that the status poll has to recover.
        await _mark_eval_error(config_hash, error_msg, row_ts or datetime.now(UTC))
        raise HTTPException(status_code=503, detail=error_msg)

    result: dict[str, Any] = {
        "eval_status": "in_progress",
        "queued": True,
        "config_hash": config_hash,
    }
    if forced:
        result["forced"] = True
    return result


_EVAL_STALE_TIMEOUT_MINUTES = int(os.environ.get("EVAL_STALE_TIMEOUT_MINUTES", "30"))


@eval_mgmt_router.get("/status")
async def eval_status() -> dict[str, Any]:
    """Return the latest eval record for this agent.

    Returns {"eval_status": "no_dataset"} immediately when no dataset is
    configured, without creating or querying the evals table.

    Auto-expires in_progress rows older than EVAL_STALE_TIMEOUT_MINUTES (default 30)
    so a crashed eval run does not leave the UI stuck in Evaluating forever.
    """
    has_pg = await _has_postgres_dataset()
    if not has_pg:
        config_dir = os.environ.get("CONFIG_PATH", "config/agent")
        eval_cases = Path(f"{config_dir}/evals/lightspeed/eval_cases.yaml")
        if not eval_cases.exists():
            return {
                "eval_status": "no_dataset",
                "message": "No eval dataset configured. Add test cases via the Dataset UI before running evaluation.",
            }

    await _ensure_evals_table_once()
    try:
        async with await _pg_conn() as conn:
            await conn.execute(
                "UPDATE evals SET eval_status='error', completed_at=NOW(), updated_at=NOW() "
                "WHERE eval_status='in_progress' "
                "AND created_at < NOW() - make_interval(mins => %s)",
                (_EVAL_STALE_TIMEOUT_MINUTES,),
            )
            row = await conn.execute(
                "SELECT * FROM evals ORDER BY created_at DESC LIMIT 1",
            )
            result = await row.fetchone()
            if not result:
                return {"eval_status": "not_started", "message": "no eval runs yet"}
            doc = _pg_row_to_dict(result, row)
            doc.pop("id", None)
            return doc
    except Exception as exc:
        import psycopg

        if isinstance(exc, psycopg.errors.UndefinedTable):
            global _table_ensured
            _table_ensured = False
            log.warning(
                "evals table does not exist yet — no eval results available: %s", exc
            )
            return {
                "eval_status": "no_dataset",
                "message": "No eval results exist yet. Run an evaluation first.",
            }
        raise


@eval_mgmt_router.get("/results")
async def eval_results(request: Request) -> dict[str, Any]:
    """Return a completed eval report.

    Optional query param ``completed_at`` fetches a specific run by its
    completion timestamp.  Without it the most recent run is returned.
    """
    from fastapi import HTTPException

    completed_at = request.query_params.get("completed_at")
    await _ensure_evals_table_once()
    try:
        async with await _pg_conn() as conn:
            if completed_at:
                row = await conn.execute(
                    "SELECT * FROM evals "
                    "WHERE eval_status='completed' AND completed_at=%s "
                    "LIMIT 1",
                    (completed_at,),
                )
            else:
                row = await conn.execute(
                    "SELECT * FROM evals WHERE eval_status='completed' "
                    "ORDER BY completed_at DESC LIMIT 1",
                )
            result = await row.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="no completed eval results")
            doc = _pg_row_to_dict(result, row)
            doc.pop("id", None)
            return doc
    except HTTPException:
        raise
    except Exception as exc:
        import psycopg

        if isinstance(exc, psycopg.errors.UndefinedTable):
            global _table_ensured
            _table_ensured = False
            log.warning(
                "evals table does not exist yet — no eval results available: %s", exc
            )
            raise HTTPException(status_code=404, detail="no completed eval results")
        raise


@eval_mgmt_router.get("/history")
async def eval_history(request: Request) -> dict[str, Any]:
    """Return historical completed eval runs (scalars only, no results_detail)."""
    try:
        limit = min(int(request.query_params.get("limit", "20")), 100)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="'limit' must be a positive integer"
        )
    await _ensure_evals_table_once()

    try:
        async with await _pg_conn() as conn:
            cur = await conn.execute(
                "SELECT eval_score, pass, fail, error, config_hash, "
                "       created_at, completed_at, "
                "       COUNT(*) OVER() AS total_count "
                "FROM evals "
                "WHERE eval_status='completed' "
                "ORDER BY completed_at DESC LIMIT %s",
                (limit,),
            )
            cols = [d.name for d in cur.description]
            runs = []
            total = 0
            async for row in cur:
                d = dict(zip(cols, row))
                total = d.pop("total_count", 0)
                for k in ("created_at", "completed_at"):
                    if d.get(k) is not None:
                        d[k] = (
                            d[k].isoformat()
                            if hasattr(d[k], "isoformat")
                            else str(d[k])
                        )
                runs.append(d)
    except Exception as exc:
        import psycopg

        if isinstance(exc, psycopg.errors.UndefinedTable):
            global _table_ensured
            _table_ensured = False
            log.warning(
                "evals table does not exist yet — no eval history available: %s", exc
            )
            return {"runs": [], "total": 0}
        raise

    return {"runs": runs, "total": total}


@eval_mgmt_router.get("/trends")
async def eval_trends(request: Request) -> dict[str, Any]:
    """Return per-metric score trends across historical eval runs."""
    try:
        limit = min(int(request.query_params.get("limit", "20")), 100)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="'limit' must be a positive integer"
        )
    await _ensure_evals_table_once()

    try:
        async with await _pg_conn() as conn:
            cur = await conn.execute(
                "SELECT results_detail->'summary'->'summary_stats'->'by_metric' as by_metric, "
                "       eval_score, completed_at "
                "FROM evals "
                "WHERE eval_status='completed' "
                "  AND results_detail IS NOT NULL "
                "ORDER BY completed_at DESC LIMIT %s",
                (limit,),
            )

            metrics: dict[str, list[dict]] = {}
            overall: list[dict] = []

            async for row in cur:
                by_metric_raw, eval_score, completed_at = row
                ts = (
                    completed_at.isoformat()
                    if hasattr(completed_at, "isoformat")
                    else str(completed_at)
                )
                overall.append({"completed_at": ts, "eval_score": eval_score})

                by_metric = by_metric_raw if isinstance(by_metric_raw, dict) else {}
                for metric_name, stats in by_metric.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(
                        {
                            "completed_at": ts,
                            "pass_rate": stats.get("pass_rate")
                            if isinstance(stats, dict)
                            else None,
                        }
                    )
    except Exception as exc:
        import psycopg

        if isinstance(exc, psycopg.errors.UndefinedTable):
            global _table_ensured
            _table_ensured = False
            log.warning(
                "evals table does not exist yet — no eval trends available: %s", exc
            )
            return {"metrics": {}, "overall": []}
        raise

    return {"metrics": metrics, "overall": overall}


# ---------------------------------------------------------------------------
# Dataset management — store / retrieve the test-case dataset in Postgres
# ---------------------------------------------------------------------------


def _collect_agent_models() -> list[dict[str, Any]]:
    """Read model: from PROMPT.md + all subagent .md frontmatters.

    Returns a list of unique model entries: [{model, source, default}].
    """
    import re

    import yaml as _yaml  # noqa: PLC0415

    config_dir = os.environ.get("CONFIG_PATH", "config/agent")
    models: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _parse_model(path: Path, source: str, default: bool = False) -> None:
        if not path.exists():
            return
        try:
            content = path.read_text(encoding="utf-8")
            m = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if not m:
                return
            fm = _yaml.safe_load(m.group(1)) or {}
            model = str(fm.get("model", "")).strip()
            if model and model not in seen:
                models.append({"model": model, "source": source, "default": default})
                seen.add(model)
        except Exception as exc:
            log.warning("Could not parse model from %s: %s", path, exc)

    _parse_model(Path(f"{config_dir}/PROMPT.md"), "orchestrator", default=True)

    subagents_dir = Path(f"{config_dir}/subagents")
    if subagents_dir.exists():
        for md_file in sorted(subagents_dir.glob("*.md")):
            fm_source = f"subagent:{md_file.stem}"
            _parse_model(md_file, fm_source)

    return models


@eval_mgmt_router.get("/models")
async def get_eval_models() -> dict[str, Any]:
    """Return available LLM models for evaluation (orchestrator + subagents)."""
    return {"models": _collect_agent_models()}


class DatasetUpsertRequest(BaseModel):
    """Dataset payload sent from the UI."""

    cases: list[dict[str, Any]]
    judge_model: str | None = None


@eval_mgmt_router.post("/dataset")
async def upsert_dataset(body: DatasetUpsertRequest) -> dict[str, Any]:
    """Upsert the eval dataset for this agent.

    Only one row is kept — DELETE + INSERT ensures the latest submission always wins.
    """
    await _ensure_datasets_table_once()
    now = datetime.now(UTC)
    dataset_json = json.dumps({"cases": body.cases})

    async with await _pg_conn() as conn:
        await conn.execute("DELETE FROM eval_datasets")
        await conn.execute(
            "INSERT INTO eval_datasets (dataset, judge_model, created_at) "
            "VALUES (%s::jsonb, %s, %s)",
            (dataset_json, body.judge_model, now),
        )

    log.info(
        "Dataset upserted: cases=%d judge_model=%s", len(body.cases), body.judge_model
    )
    return {"status": "ok", "case_count": len(body.cases)}


@eval_mgmt_router.get("/dataset")
async def get_dataset() -> dict[str, Any]:
    """Return the stored eval dataset for this agent."""
    from fastapi import HTTPException

    await _ensure_datasets_table_once()

    async with await _pg_conn() as conn:
        row = await conn.execute(
            "SELECT dataset, judge_model, created_at FROM eval_datasets LIMIT 1",
        )
        result = await row.fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="no dataset found")

    dataset, judge_model, created_at = result
    if isinstance(dataset, str):
        dataset = json.loads(dataset)

    return {
        "dataset": dataset,
        "judge_model": judge_model,
        "created_at": (
            created_at.isoformat()
            if hasattr(created_at, "isoformat")
            else str(created_at)
        ),
    }


@eval_mgmt_router.post("/internal/cleanup")
async def cleanup_eval_redis(request: Request) -> dict[str, Any]:
    """Delete eval Redis keys when an eval run completes.

    Called by the eval pod via POST with X-Internal-Token header.
    Returns 200 always — best-effort, never blocks the eval pod.
    Security: constant-time token comparison prevents timing attacks.
    """
    if not _EVAL_TOKEN_REFRESH_ENABLED:
        return {"status": "disabled"}

    provided = request.headers.get("x-internal-token", "")
    if not _EVAL_INTERNAL_TOKEN or not hmac.compare_digest(
        provided.encode(), _EVAL_INTERNAL_TOKEN.encode()
    ):
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="invalid internal token")

    try:
        from deep_agent.aegra.redis import cache_delete, cache_get

        sub = cache_get("eval:trigger_sub")
        cache_delete("eval:trigger_sub")
        if sub:
            cache_delete(f"eval:active:{sub}")
            cache_delete(f"eval:access:{sub}")
            cache_delete(f"eval:refresh:{sub}")
        log.info("eval_redis_cleanup keys_deleted=%d", 4 if sub else 1)
    except Exception as exc:
        log.warning("eval_redis_cleanup failed (TTL will handle): %s", exc)

    return {"status": "ok"}
