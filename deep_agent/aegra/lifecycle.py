"""Lifecycle state persistence — Postgres-based recovery.

When a pod is terminated mid-run, Aegra's LeaseReaper detects the
expired lease in Postgres and re-enqueues the run to the worker
queue. The new worker resumes from the LangGraph checkpoint.

Flow:
    1. ``persist_inflight_runs`` — called during shutdown, expires
       leases on active runs in Postgres so LeaseReaper recovers them.
    2. ``resume_interrupted_runs`` — called during startup, picks up
       interrupted runs using ``FOR UPDATE SKIP LOCKED`` to avoid
       duplicate processing across replicas.

Environment variables:
    POD_NAME: Kubernetes pod name (defaults to hostname).

Token encryption uses ``SSO_CLIENT_SECRET`` from settings (already
available as an env var in every pod).
"""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# ── Module constants ─────────────────────────────────────────────

POD_ID: str = os.environ.get(
    "POD_NAME",
    os.environ.get("HOSTNAME", f"local-{os.getpid()}"),
)


# ── Pydantic model ───────────────────────────────────────────────


class RunExecutionContext(BaseModel):
    """Snapshot of everything needed to resume an interrupted run."""

    model_name: str = Field(description="LLM model identifier")
    config_hash: str = Field(description="Hash of the graph config at build time")
    assistant_id: Optional[str] = Field(default=None, description="Aegra assistant ID")
    user_id: Optional[str] = Field(default=None, description="Authenticated user ID")
    encrypted_refresh_token: Optional[str] = Field(
        default=None, description="Fernet-encrypted refresh token"
    )
    mcp_server_names: list[str] = Field(
        default_factory=list, description="MCP servers active at time of run"
    )
    orchestrator_config_snapshot: dict[str, Any] = Field(
        default_factory=dict, description="Orchestrator config at time of run"
    )


# ── Token encryption ─────────────────────────────────────────────


def _derive_fernet_key(secret: str) -> bytes:
    """Derive a 32-byte Fernet key from an arbitrary secret string.

    Uses SHA-256 to produce exactly 32 bytes, then base64-urlsafe
    encodes the result as required by Fernet.
    """
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _get_encryption_secret() -> str:
    """Get the encryption secret from SSO_CLIENT_SECRET in settings.

    Falls back to env var, then empty string (which disables encryption).
    Uses SSO_CLIENT_SECRET because it is already available as an env
    var in every pod and provides sufficient entropy for Fernet key
    derivation.
    """
    try:
        from deep_agent.src.settings import settings

        if settings.SSO_CLIENT_SECRET:
            return settings.SSO_CLIENT_SECRET
    except Exception:
        pass
    return os.environ.get("SSO_CLIENT_SECRET", "")


def encrypt_token(token: str, secret: str) -> str:
    """Encrypt a token string using Fernet with the derived key.

    Args:
        token: Plaintext token to encrypt.
        secret: Shared secret for key derivation.

    Returns:
        Base64-encoded Fernet ciphertext as a string.
    """
    from cryptography.fernet import Fernet

    key = _derive_fernet_key(secret)
    f = Fernet(key)
    return str(f.encrypt(token.encode("utf-8")).decode("utf-8"))


def decrypt_token(encrypted: str, secret: str) -> str:
    """Decrypt a Fernet-encrypted token string.

    Args:
        encrypted: Base64-encoded Fernet ciphertext.
        secret: Shared secret for key derivation (must match encrypt).

    Returns:
        Plaintext token string.

    Raises:
        cryptography.fernet.InvalidToken: If the key is wrong or
            the ciphertext is corrupted.
    """
    from cryptography.fernet import Fernet

    key = _derive_fernet_key(secret)
    f = Fernet(key)
    return str(f.decrypt(encrypted.encode("utf-8")).decode("utf-8"))


# ── Persist and resume ───────────────────────────────────────────


def _get_active_runs() -> dict[str, Any]:
    """Get active runs from the Aegra API core module.

    Returns an empty dict if the module is unavailable (e.g. in
    tests or when Aegra is not running).
    """
    try:
        from aegra_api.core.active_runs import active_runs

        return dict(active_runs)
    except ImportError:
        logger.debug("aegra_api.core.active_runs not available")
        return {}
    except Exception as exc:
        logger.debug("Failed to get active runs: %s", exc)
        return {}


def _update_run_status_postgres(
    run_id: str,
    status: str,
    claimed_by: Optional[str] = None,
) -> bool:
    """Update the run status in PostgreSQL using sync psycopg.

    Uses a direct connection (not async) since this is called
    during shutdown when the event loop may be draining.

    Also updates ``claimed_by`` when provided (used during
    shutdown to record which pod was running this).

    Args:
        run_id: The run to update.
        status: New status string (e.g. ``"interrupted"``).
        claimed_by: Pod ID to set on the ``claimed_by`` column.

    Returns:
        True if the row was updated, False on error.
    """
    try:
        import psycopg

        from deep_agent.src.settings import settings

        with psycopg.connect(settings.database_uri) as conn:
            with conn.cursor() as cur:
                if claimed_by is not None:
                    cur.execute(
                        "UPDATE runs SET status = %s, claimed_by = %s, "
                        "updated_at = NOW() "
                        "WHERE run_id = %s AND status = 'running'",
                        (status, claimed_by, run_id),
                    )
                else:
                    cur.execute(
                        "UPDATE runs SET status = %s, updated_at = NOW() "
                        "WHERE run_id = %s AND status = 'running'",
                        (status, run_id),
                    )
                updated = bool(cur.rowcount > 0)
            conn.commit()

        if updated:
            logger.info("Updated run %s status to '%s' in Postgres", run_id, status)
        return updated
    except Exception as exc:
        logger.warning("Failed to update run %s status in Postgres: %s", run_id, exc)
        return False


def persist_inflight_runs() -> int:
    """Expire leases on active runs so Aegra's LeaseReaper recovers them.

    Reads the in-memory ``active_runs`` dict from Aegra for the
    authoritative list of active runs (instant, no network call).

    For each active run, sets ``lease_expires_at`` to the past so
    Aegra's built-in LeaseReaper finds them on the next scan cycle
    and re-enqueues them to the worker queue. The worker then
    resumes from the LangGraph checkpoint through the normal
    execution path — no custom resume code needed.

    Returns:
        Number of runs whose leases were expired.
    """
    active = _get_active_runs()
    if active:
        run_ids = list(active.keys())
        logger.info(
            "Read %d active run(s) from in-memory active_runs dict", len(run_ids)
        )
    else:
        run_ids = []

    if not run_ids:
        logger.info("No inflight runs to persist")
        return 0

    persisted = _expire_run_leases_batch(run_ids)

    logger.info(
        "Expired leases on %d/%d inflight runs — LeaseReaper will re-enqueue them",
        persisted,
        len(run_ids),
    )
    return persisted


def _expire_run_leases_batch(run_ids: list[str]) -> int:
    """Set lease_expires_at to the past for all given runs in a single query."""
    if not run_ids:
        return 0
    try:
        import psycopg

        from deep_agent.src.settings import settings

        with psycopg.connect(settings.database_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE runs SET lease_expires_at = NOW() - INTERVAL '1 second', "
                    "execution_params = COALESCE(execution_params, '{}'::jsonb) || "
                    "jsonb_build_object('started_by_pod', %s, 'killed_at', NOW()::text), "
                    "updated_at = NOW() "
                    "WHERE run_id = ANY(%s) AND status = 'running'",
                    (POD_ID, run_ids),
                )
                updated = cur.rowcount or 0
            conn.commit()

        if updated:
            logger.info(
                "Expired leases for %d run(s) (started_by=%s) — reaper will recover",
                updated,
                POD_ID,
            )
        return updated
    except Exception as exc:
        logger.warning("Failed to expire leases for runs %s: %s", run_ids, exc)
        return 0


async def resume_interrupted_runs(
    db_uri: str,
    resume_fn: Callable,
) -> dict[str, str]:
    """Resume runs marked as interrupted in Postgres.

    Uses ``FOR UPDATE SKIP LOCKED`` to prevent multiple replicas
    from resuming the same run concurrently. Claims each run with
    ``claimed_by`` and ``lease_expires_at`` before processing.

    Queries runs where status is 'running' or 'interrupted' AND
    the lease has expired (or was never set), which handles both
    graceful shutdown and pod crash scenarios.

    Args:
        db_uri: PostgreSQL connection URI.
        resume_fn: Async callable that accepts ``(run_id, thread_id)``
            and resumes the run.

    Returns:
        Dict mapping run_id to outcome (``"resumed"``, ``"skipped"``,
        or ``"error: ..."``)
    """
    try:
        from deep_agent.src.settings import settings as app_settings

        max_batch = app_settings.LIFECYCLE_MAX_RESUME_BATCH
        lease_seconds = app_settings.LIFECYCLE_LEASE_SECONDS
    except Exception:
        max_batch = 10
        lease_seconds = 300

    results: dict[str, str] = {}

    try:
        import psycopg

        async with await psycopg.AsyncConnection.connect(db_uri) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT run_id, thread_id, assistant_id, execution_params::text "
                    "FROM runs "
                    "WHERE status IN ('running', 'interrupted') "
                    "AND (lease_expires_at IS NULL "
                    "     OR lease_expires_at < NOW()) "
                    "ORDER BY updated_at ASC "
                    "LIMIT %s "
                    "FOR UPDATE SKIP LOCKED",
                    (max_batch,),
                )
                rows = await cur.fetchall()

                for row in rows:
                    run_id, thread_id = row[0], row[1]

                    await cur.execute(
                        "UPDATE runs SET claimed_by = %s, "
                        "lease_expires_at = NOW() + %s * INTERVAL '1 second', "
                        "status = 'running', "
                        "updated_at = NOW() "
                        "WHERE run_id = %s",
                        (POD_ID, lease_seconds, run_id),
                    )

                    try:
                        await resume_fn(run_id, thread_id)
                        await cur.execute(
                            "UPDATE runs SET status = 'success', "
                            "updated_at = NOW() WHERE run_id = %s",
                            (run_id,),
                        )
                        results[run_id] = "resumed"
                        logger.info("Resumed interrupted run %s", run_id)
                    except Exception as exc:
                        await cur.execute(
                            "UPDATE runs SET status = 'error', "
                            "updated_at = NOW() WHERE run_id = %s",
                            (run_id,),
                        )
                        results[run_id] = f"error: {exc}"
                        logger.warning("Failed to resume run %s: %s", run_id, exc)

            await conn.commit()

    except Exception as exc:
        logger.warning("Failed to query interrupted runs: %s", exc)

    return results


# ── Execution context builder ────────────────────────────────────


def build_execution_context(
    model_name: str,
    config_hash: str,
    assistant_id: Optional[str] = None,
    user_id: Optional[str] = None,
    refresh_token: Optional[str] = None,
    mcp_server_names: Optional[list[str]] = None,
    orchestrator_config: Optional[dict[str, Any]] = None,
    encryption_secret: Optional[str] = None,
) -> dict:
    """Build a serializable execution context for inflight persistence.

    Encrypts the refresh token if provided. Uses SSO_CLIENT_SECRET
    from settings for key derivation (falls back to explicit secret
    parameter if provided, for testing).

    Args:
        model_name: LLM model identifier.
        config_hash: Graph config fingerprint.
        assistant_id: Aegra assistant ID.
        user_id: Authenticated user ID.
        refresh_token: SSO refresh token (will be encrypted).
        mcp_server_names: Active MCP server names.
        orchestrator_config: Orchestrator config snapshot.
        encryption_secret: Explicit secret override (for testing).
            Falls back to ``SSO_CLIENT_SECRET`` from settings.

    Returns:
        Dict representation of ``RunExecutionContext``.
    """
    encrypted = None
    if refresh_token:
        secret = encryption_secret or _get_encryption_secret()
        if secret:
            try:
                encrypted = encrypt_token(refresh_token, secret)
            except Exception as exc:
                logger.warning("Failed to encrypt refresh token: %s", exc)
        else:
            logger.warning(
                "Refresh token encryption skipped — SSO_CLIENT_SECRET not configured"
            )

    ctx = RunExecutionContext(
        model_name=model_name,
        config_hash=config_hash,
        assistant_id=assistant_id,
        user_id=user_id,
        encrypted_refresh_token=encrypted,
        mcp_server_names=mcp_server_names or [],
        orchestrator_config_snapshot=orchestrator_config or {},
    )
    return dict(ctx.model_dump())
