"""Startup orchestrator — coordinated initialization on process boot.

Runs once when the agent process starts. Ensures all subsystems
are initialized in the correct order before the server accepts
traffic.

Startup sequence:
    1. Validate configuration
    2. Initialize Aegra database manager (Postgres checkpointer + store)
    3. Ensure database tables exist
    4. Warm caches (if enabled)
    5. Start memory scheduler (if enabled)
    6. Set up Langfuse tracing (if configured)
    7. Log readiness

This module is idempotent — calling ``run_startup()`` multiple
times is safe (each step guards against double-init).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import psycopg

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_startup_complete = False


async def run_startup() -> dict[str, str]:
    """Execute the startup sequence. Returns a status dict.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _startup_complete  # noqa: PLW0603

    if _startup_complete:
        logger.debug("Startup already complete — skipping")
        return {"status": "already_complete"}

    t0 = time.monotonic()
    results: dict[str, str] = {}

    results["config"] = await _validate_config()
    results["aegra_db"] = await _init_aegra_db()
    results["database"] = await _ensure_database()
    _check_mcp_encryption_key()
    results["resume"] = await _resume_interrupted_runs()
    results["mcp_apps"] = _setup_mcp_apps_capability()
    results["cache"] = await _warm_caches()
    results["scheduler"] = await _start_scheduler()
    results["otel"] = _setup_otel()
    results["telemetry"] = _setup_telemetry()

    _upgrade_signal_handlers()

    elapsed = round((time.monotonic() - t0) * 1000, 1)
    _startup_complete = True

    logger.info(
        "Startup complete in %.1fms: %s",
        elapsed,
        results,
    )
    return results


def _upgrade_signal_handlers() -> None:
    """Upgrade to loop-aware signal handlers for async drain."""
    try:
        from deep_agent.aegra.shutdown import register_signal_handlers

        register_signal_handlers()
    except Exception:
        logger.warning("Failed to register signal handlers", exc_info=True)


async def _init_aegra_db() -> str:
    """Initialize Aegra's DatabaseManager for Postgres checkpointing.

    When running under raw uvicorn (production Containerfile), the
    ``aegra dev`` startup path that normally calls
    ``db_manager.initialize()`` is bypassed.  This step ensures the
    Postgres connection pool, checkpointer, and store are created
    before the first graph request.
    """
    try:
        from aegra_api.core.database import db_manager

        if db_manager.engine is not None:
            return "ok: already initialized"

        await db_manager.initialize()
        logger.info("Aegra DatabaseManager initialized (Postgres checkpointer ready)")
        return "ok"
    except Exception as exc:
        logger.warning("Aegra DB init failed (falling back to in-memory): %s", exc)
        return f"warning: {exc}"


def _clear_stale_done_keys(run_ids: list[str]) -> None:
    """Remove Redis done/counter/cache keys for runs being re-enqueued.

    Aegra's run_executor sets ``aegra:run:done:<id>`` even for interrupted
    runs. If these keys survive into the next pod, the LeaseReaper skips
    the run thinking it's already handled.
    """
    try:
        from deep_agent.aegra.redis import get_redis_client

        client = get_redis_client()
        if client is None:
            return
        prefix = "aegra:run:"
        keys_to_delete = []
        for rid in run_ids:
            keys_to_delete.extend(
                [
                    f"{prefix}done:{rid}",
                    f"{prefix}counter:{rid}",
                    f"{prefix}cache:{rid}",
                ]
            )
        if keys_to_delete:
            client.delete(*keys_to_delete)
            logger.info(
                "Cleared %d stale Redis keys for %d re-enqueued runs",
                len(keys_to_delete),
                len(run_ids),
            )
    except Exception as exc:
        logger.warning("Failed to clear stale Redis keys: %s", exc)


async def _clear_input_for_checkpoint_resume(
    conn: "psycopg.AsyncConnection[Any]",
    run_ids: list[str],
) -> int:
    """Clear input_data and command from runs so they resume from checkpoint only.

    When a run is re-enqueued after a crash, _resolve_input() returns
    the stored input_data or command. Re-sending the original input
    causes LangGraph to re-process it on top of the checkpoint state,
    which creates duplicate messages and re-triggers HITL approvals.

    Clearing both fields makes _resolve_input() return None, so the
    graph resumes purely from its checkpoint — the correct behavior
    for crash recovery.
    """
    if not run_ids:
        return 0
    async with conn.cursor() as cur:
        await cur.execute(
            "UPDATE runs SET "
            "input = NULL, "
            "execution_params = jsonb_set("
            "  jsonb_set("
            "    execution_params, "
            "    '{execution,input_data}', 'null'::jsonb"
            "  ), "
            "  '{execution,command}', 'null'::jsonb"
            ") "
            "WHERE run_id = ANY(%s)",
            (run_ids,),
        )
        updated = cur.rowcount or 0
    if updated:
        logger.info(
            "Cleared input/command for %d run(s) — will resume from checkpoint only",
            updated,
        )
    return updated


async def _resume_interrupted_runs() -> str:
    """Reset interrupted runs so Aegra's LeaseReaper can recover them.

    Aegra's own shutdown may mark active runs as ``interrupted``
    (via executor.stop), but the LeaseReaper only scans for
    ``status='running'`` with expired leases. This step resets
    ``interrupted`` runs back to ``running`` with an expired lease
    so the reaper picks them up on its next scan cycle.
    """
    try:
        from deep_agent.src.settings import settings as app_settings

        if not app_settings.LIFECYCLE_PERSISTENCE_ENABLED:
            return "skipped: lifecycle persistence disabled"
        if not app_settings.LIFECYCLE_RESUME_ON_STARTUP:
            return "skipped: resume on startup disabled"

        import psycopg

        recovered = 0
        recovered_run_ids: list[str] = []
        async with await psycopg.AsyncConnection.connect(
            app_settings.database_uri
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT run_id FROM runs WHERE status = 'interrupted' "
                    "ORDER BY updated_at ASC LIMIT %s",
                    (app_settings.LIFECYCLE_MAX_RESUME_BATCH,),
                )
                recovered_run_ids = [row[0] for row in await cur.fetchall()]

                if recovered_run_ids:
                    await cur.execute(
                        "UPDATE runs SET status = 'running', "
                        "lease_expires_at = NOW() - INTERVAL '1 second', "
                        "claimed_by = NULL, "
                        "updated_at = NOW() "
                        "WHERE run_id = ANY(%s) AND status = 'interrupted'",
                        (recovered_run_ids,),
                    )
                recovered = len(recovered_run_ids)

                await cur.execute(
                    "SELECT run_id, thread_id FROM runs "
                    "WHERE status = 'running' "
                    "AND lease_expires_at IS NOT NULL "
                    "AND lease_expires_at < NOW() "
                    "ORDER BY updated_at ASC LIMIT 20"
                )
                stale = await cur.fetchall()
                stale_run_ids = [row[0] for row in stale]

            all_recoverable = list(set(recovered_run_ids + stale_run_ids))
            if all_recoverable:
                await _clear_input_for_checkpoint_resume(conn, all_recoverable)

            await conn.commit()

        if recovered > 0:
            _clear_stale_done_keys(recovered_run_ids)
            logger.info(
                "Reset %d interrupted run(s) to running — "
                "LeaseReaper will re-enqueue them",
                recovered,
            )

        for run_id, thread_id in stale:
            logger.info(
                "Stale run awaiting LeaseReaper: run=%s thread=%s",
                run_id,
                thread_id,
            )

        total = recovered + len(stale)
        if total == 0:
            return "ok: no recoverable runs"
        return f"ok: {recovered} reset, {len(stale)} stale — LeaseReaper will recover"
    except Exception as exc:
        logger.warning("Resume check failed: %s", exc)
        return f"warning: {exc}"


async def _validate_config() -> str:
    """Validate core settings."""
    try:
        from deep_agent.src.settings import settings, validate_config

        validate_config(settings)

        from deep_agent.aegra.middleware import validate_auth_config

        validate_auth_config()
        return "ok"
    except Exception as exc:
        logger.error("Config validation failed: %s", exc)
        raise  # Re-raise to fail startup


def _check_mcp_encryption_key() -> None:
    """Warn if any MCP server uses oauth/dcr but MCP_TOKEN_ENCRYPTION_KEY is not set."""
    try:
        from deep_agent.src.agent.config import agent_config
        from deep_agent.src.settings import settings

        servers = agent_config.get_mcp_servers()
        dcr_enabled = settings.MCP_DCR_ENABLED
        check_modes = {"oauth", "dcr"} if dcr_enabled else {"oauth"}
        needs_key = any(
            s.get("auth_mode") in check_modes
            for s in servers.values()
            if isinstance(s, dict) and s.get("enabled", False)
        )
        if needs_key and not os.environ.get("MCP_TOKEN_ENCRYPTION_KEY"):
            logger.error(
                "MCP_TOKEN_ENCRYPTION_KEY is not set but one or more MCP servers "
                "use auth_mode 'oauth' or 'dcr'. Token encryption will fail."
            )
    except Exception:
        logger.debug("MCP encryption key check skipped", exc_info=True)


def _setup_mcp_apps_capability() -> str:
    """Ensure MCP initialize advertises the Apps UI extension (SEP-1865)."""
    try:
        from deep_agent.aegra.mcp_apps import ensure_mcp_apps_capability_advertised

        newly_installed = ensure_mcp_apps_capability_advertised()
        return "ok" if newly_installed else "already_installed"
    except Exception as exc:
        logger.error("MCP Apps capability setup failed: %s", exc)
        return f"error: {exc}"


async def _ensure_database() -> str:
    """Create personalization, feedback, and token budget tables if they don't exist."""
    try:
        from deep_agent.src.feedback.repository import FeedbackRepository
        from deep_agent.src.personalization.repository import (
            PersonalizationRepository,
        )
        from deep_agent.src.settings import settings

        setup_tasks = []

        if settings.database_uri:
            personalization_repo = PersonalizationRepository(settings.database_uri)
            feedback_repo = FeedbackRepository(settings.database_uri)
            setup_tasks.append(personalization_repo.ensure_tables())
            setup_tasks.append(feedback_repo.ensure_table())

            from deep_agent.aegra.mcp_token_store import McpTokenStore

            mcp_token_store = McpTokenStore(settings.database_uri)
            setup_tasks.append(mcp_token_store.ensure_tables())

        if settings.MONGODB_URI:
            from deep_agent.src.token_budget.mongo_repository import (
                TokenUsageMongoRepository,
            )

            mongo_repo = TokenUsageMongoRepository(
                settings.MONGODB_URI,
                db_name=settings.MONGODB_DB,
            )
            setup_tasks.append(mongo_repo.ensure_indexes())

        if not setup_tasks:
            return "skipped: no database configured"

        await asyncio.gather(*setup_tasks)
        return "ok"
    except Exception as exc:
        logger.error("Database setup failed: %s", exc)
        raise


async def _warm_caches() -> str:
    """Pre-populate caches if caching is enabled."""
    try:
        from deep_agent.src.cache.config import cache_settings

        if not cache_settings.CACHE_ENABLED:
            return "skipped: caching disabled"

        from deep_agent.src.cache.warming import warm_caches

        warm_caches()
        return "ok"
    except Exception as exc:
        logger.warning("Cache warming failed: %s", exc)
        return f"warning: {exc}"


async def _start_scheduler() -> str:
    """Start background memory scheduler if enabled."""
    try:
        from deep_agent.src.memory.config import memory_settings

        if not memory_settings.MEMORY_CONSOLIDATION_ENABLED:
            return "skipped: memory consolidation disabled"

        from deep_agent.src.memory.scheduler import start_scheduler
        from deep_agent.src.settings import settings

        started = await start_scheduler(settings.database_uri)
        return "ok" if started else "skipped: already running"
    except Exception as exc:
        logger.warning("Scheduler start failed: %s", exc)
        return f"warning: {exc}"


def _setup_otel() -> str:
    """Initialize OpenTelemetry metrics and tracing."""
    try:
        from deep_agent.aegra.otel import initialize_telemetry

        initialize_telemetry()
        return "ok"
    except Exception as exc:
        logger.warning("OTEL setup failed: %s", exc)
        return f"warning: {exc}"


def _setup_telemetry() -> str:
    """Register PII middleware, Langfuse tracing, token budget, and Guardian."""
    try:
        from deep_agent.aegra.telemetry import (
            setup_guardian_guardrails,
            setup_langfuse_tracing,
            setup_pii_middleware,
            setup_token_budget_tracking,
        )

        setup_pii_middleware()  # must be first — Langfuse handler depends on the scrubber
        setup_langfuse_tracing()
        setup_token_budget_tracking()
        setup_guardian_guardrails()
        return "ok"
    except Exception as exc:
        logger.warning("Telemetry setup failed: %s", exc)
        return f"warning: {exc}"


def is_ready() -> bool:
    """Return True if startup has completed."""
    return _startup_complete
