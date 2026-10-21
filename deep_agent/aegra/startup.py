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
# Strong references to background tasks; asyncio keeps only weak ones.
# `Any` rather than `Task[None]`: the sweeper returns None and the
# warm-up returns a count, and both belong here.
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


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
    results["ldap"] = _init_ldap()
    results["aegra_db"] = await _init_aegra_db()
    results["run_wait"] = _install_fast_run_wait()
    results["database"] = await _ensure_database()
    _check_mcp_encryption_key()
    results["resume"] = await _resume_interrupted_runs()
    results["mcp_apps"] = _setup_mcp_apps_capability()
    results["cache"] = await _warm_caches()
    results["otel"] = _setup_otel()
    results["telemetry"] = _setup_telemetry()
    results["graph_warmup"] = _warmup_graph_imports()

    # Marked done *before* the graph is built, not after.
    #
    # `_warm_graph` calls `agent()`, which calls `_ensure_startup()`,
    # which called `run_startup()` again because this flag was still
    # False — and each nesting repeated the database init, the
    # interrupted-run resume, the cache warm and another graph build.
    # Measured on a live pod: 224 startups per boot, each one slower
    # than the last. Everything above this line is the part that must
    # not run twice; everything below is safe to re-enter.
    _startup_complete = True

    # Separate from the cache step on purpose: that one returns early
    # when CACHE_ENABLED is false, and building the graph is worth doing
    # whether or not the model cache is on.
    results["graph_build"] = await _warm_graph()
    results["inflight_metrics"] = _start_inflight_metrics()
    results["run_sweeper"] = _start_interrupted_run_sweeper()

    _upgrade_signal_handlers()

    elapsed = round((time.monotonic() - t0) * 1000, 1)

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


def _init_ldap() -> str:
    """Load PROMPT.md access config and validate LDAP settings."""
    try:
        from deep_agent.src.ldap.config import ldap_settings
        from deep_agent.src.ldap.prompt_config import get_prompt_access_config

        config = get_prompt_access_config()
        group_count = len(config.groups) if config.groups else 0

        if group_count == 0:
            logger.info(
                "LDAP access control: no groups in PROMPT.md — everyone gets users role"
            )
            return "ok: no groups configured"

        if not ldap_settings.LDAP_URL:
            logger.warning(
                "LDAP access control: %d group(s) defined but LDAP_URL not set — "
                "access will be denied (fail-closed) when accessibility=private",
                group_count,
            )
            return f"warning: {group_count} group(s) but no LDAP_URL"

        logger.info(
            "LDAP access control: %d group(s), accessibility=%s, url=%s",
            group_count,
            config.accessibility,
            ldap_settings.LDAP_URL,
        )
        return f"ok: {group_count} group(s), accessibility={config.accessibility}"
    except Exception as exc:
        logger.warning("LDAP init failed: %s", exc)
        return f"warning: {exc}"


def _install_fast_run_wait() -> str:
    """Stop `/runs/wait` adding up to two seconds to every turn."""
    from deep_agent.aegra.run_wait import install_fast_run_wait

    return install_fast_run_wait()


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
            # Aegra's own lifespan got there first. The pool still
            # needs its liveness check — skipping it here is how the
            # fix silently did nothing in the one configuration where
            # aegra initializes itself.
            _enable_pool_liveness_check(db_manager)
            return "ok: already initialized"

        await db_manager.initialize()
        _enable_pool_liveness_check(db_manager)
        logger.info("Aegra DatabaseManager initialized (Postgres checkpointer ready)")
        return "ok"
    except Exception as exc:
        logger.warning("Aegra DB init failed (falling back to in-memory): %s", exc)
        return f"warning: {exc}"


def _enable_pool_liveness_check(db_manager: "Any") -> None:
    """Make the LangGraph pool verify a connection before handing it out.

    Aegra builds ``lg_pool`` with psycopg's default ``check=None``, so
    a connection the server has already closed — proxy idle timeout,
    failover, network partition — is handed to the next borrower and
    fails that caller's query. After a failover this arrives as one
    error per pooled connection, all at once, looking like an
    application fault. The SQLAlchemy engine beside it already guards
    this with ``pool_pre_ping=True``; the LangGraph pool does not.

    ``_check`` is private, and setting it here is a patch over an
    upstream gap rather than a design. The real fix is a ``check``
    argument on aegra's pool construction; until that exists this is
    the only seam, so it is deliberately best-effort and never fatal.
    """
    pool = getattr(db_manager, "lg_pool", None)
    if pool is None:
        return
    try:
        from psycopg_pool import AsyncConnectionPool

        pool._check = AsyncConnectionPool.check_connection  # noqa: SLF001
        logger.info("LangGraph pool: liveness check enabled on checkout")
    except Exception as exc:
        logger.warning("Could not enable LangGraph pool liveness check: %s", exc)


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


async def _resume_interrupted_runs(
    min_age_seconds: int = 0, *, startup: bool = True
) -> str:
    """Reset interrupted runs so Aegra's LeaseReaper can recover them.

    ``min_age_seconds`` leaves very recent ones alone, so the periodic
    sweep does not grab a run a pod is still in the middle of shutting
    down. Startup passes zero: anything interrupted then is by
    definition from a process that is already gone.

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
        # Only the startup call. The periodic sweep answers a
        # different question — "did anything get stranded since?" —
        # and turning off resume-at-boot used to silence that too,
        # leaving an enabled sweep recovering nothing.
        if startup and not app_settings.LIFECYCLE_RESUME_ON_STARTUP:
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
                    "AND updated_at < NOW() - make_interval(secs => %s) "
                    "ORDER BY updated_at ASC LIMIT %s",
                    (min_age_seconds, app_settings.LIFECYCLE_MAX_RESUME_BATCH),
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


def _start_interrupted_run_sweeper() -> str:
    """Re-check for interrupted runs periodically, not only at boot.

    `_resume_interrupted_runs` runs once, during startup. A pod
    shutting down marks its in-flight runs `interrupted`, and when
    that write lands *after* the replacement pod has already scanned,
    nothing looks again until the next restart. Observed on a rolling
    update: the new pods scanned at 09:53:55 and 09:54:15, the old
    ones marked five runs interrupted at 09:54:31, and two threads sat
    "busy" in the sidebar with no one coming for them.
    """
    from deep_agent.src.settings import settings as app_settings

    interval = app_settings.LIFECYCLE_SWEEP_INTERVAL_SECONDS
    if not app_settings.LIFECYCLE_PERSISTENCE_ENABLED:
        return "skipped: lifecycle persistence disabled"
    if interval <= 0:
        return "skipped: sweep disabled"

    async def _sweep() -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                result = await _resume_interrupted_runs(
                    min_age_seconds=app_settings.LIFECYCLE_SWEEP_MIN_AGE_SECONDS,
                    startup=False,
                )
                if not result.startswith("ok: no recoverable"):
                    logger.info("Interrupted-run sweep: %s", result)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Interrupted-run sweep failed", exc_info=True)

    try:
        task = asyncio.create_task(_sweep())
        # Held so the task is not garbage collected mid-flight.
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)
    except RuntimeError:
        return "skipped: no running loop"
    return f"ok: every {interval}s"


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
            from deep_agent.src.projects import ProjectsRepository

            mcp_token_store = McpTokenStore(settings.database_uri)
            setup_tasks.append(mcp_token_store.ensure_tables())
            setup_tasks.append(ProjectsRepository.ensure_tables(settings.database_uri))

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


def _start_inflight_metrics() -> str:
    """Publish `aegra_runs_in_flight` for autoscaling to measure.

    Optional: an agent without Prometheus or KEDA does not need it, and
    a failure here must never block startup.
    """
    import os

    # On by default: the gauge drives autoscaling, so an operator who
    # forgets the flag gets working scale-down rather than a silent
    # fallback to CPU-only scaling.
    if os.getenv("ENABLE_PROMETHEUS_METRICS", "true").lower() not in ("true", "1"):
        return "skipped: metrics disabled"

    try:
        from deep_agent.aegra.inflight_metrics import start_inflight_metrics

        start_inflight_metrics()
    except Exception as exc:
        logger.warning("In-flight metrics unavailable: %s", exc)
        return f"error: {exc}"
    return "ok"


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


async def _warm_graph() -> str:
    """Build the graph once before any request arrives.

    The first request to a fresh pod otherwise pays for the full build —
    config load, tool resolution, MCP discovery — and measured ~500ms
    against ~45ms warm. That is paid on every deploy, every scale-up,
    and with the scale-to-zero component on *every* first request after
    an idle period, which is exactly when someone is watching.

    The cache key includes the caller's identity, because the compiled
    graph holds middleware built for one person. So warming with no
    user populates an entry no real request looks for, and the first
    request of every person still paid the full build. Worse, the
    anonymous build is the *cheap* one: with no token there is nothing
    to connect to, so it finished in milliseconds and looked like it
    had done the job. A user with a connected integration pays ~10s
    of MCP discovery, which is the cost worth moving off the critical
    path.

    So: the anonymous key first, then the key of every user who has a
    stored MCP token. Those are exactly the people whose build is
    slow, and they are already known at boot.
    """
    warmed = 0
    try:
        from deep_agent.aegra.graph import agent

        class _NoUser:
            """Minimal runtime: the factory skips MCP when user is None."""

            user = None

        await agent(_NoUser())
        warmed += 1
    except Exception as exc:
        # Never fatal. A pod that cannot pre-build will still serve,
        # just with the old first-request cost.
        logger.warning("Graph warm-up skipped: %s", exc)
        return f"graph not warmed: {exc}"

    # In the background, deliberately.
    #
    # Each connected user costs the ~10s of MCP discovery it saves
    # them later, and awaiting that here put it on the readiness
    # path: startup went from 7s to 67s, the pod failed its probe
    # window and the rolling update timed out. The whole point is to
    # move the cost off a path someone is waiting on, and readiness
    # is one of those.
    try:
        task = asyncio.create_task(_warm_connected_users())
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)
    except RuntimeError:
        logger.debug("No running loop for per-user graph warm-up")

    return f"graph warmed ({warmed} key(s), users warming in background)"


async def _warm_connected_users() -> int:
    """Pre-build for each user who has an MCP token stored.

    Bounded: a deployment with many users must not spend minutes of
    boot warming all of them, and the ones left out are no worse off
    than before this existed.
    """
    from deep_agent.aegra.graph import agent
    from deep_agent.aegra.redis import get_redis_client, run_redis
    from deep_agent.src.settings import settings as app_settings

    limit = app_settings.GRAPH_WARM_MAX_USERS
    if limit <= 0:
        return 0

    pattern = f"{app_settings.MCP_TOKEN_KEY_PREFIX}{app_settings.agent_deployment_id}:*"

    def _scan() -> list[str]:
        client = get_redis_client()
        if client is None:
            return []
        return [
            key.decode() if isinstance(key, bytes) else str(key)
            for key in client.scan_iter(match=f"*{pattern}", count=200)
        ]

    keys = await run_redis(_scan)
    users: list[str] = []
    for key in keys:
        # ...<prefix><agent>:<user>:<mcp>, and a user id can itself
        # contain a colon, so take the middle by position from the
        # right rather than splitting from the left.
        parts = key.split(":")
        if len(parts) < 3:
            continue
        user_id = parts[-2]
        if user_id and user_id not in users:
            users.append(user_id)
        if len(users) >= limit:
            break

    if not users:
        return 0

    class _User:
        def __init__(self, identity: str) -> None:
            self.identity = identity
            self.access_token = None
            self.refresh_token = None

    class _Runtime:
        def __init__(self, identity: str) -> None:
            self.user = _User(identity)

    warmed = 0
    for user_id in users:
        try:
            await agent(_Runtime(user_id))
            warmed += 1
        except Exception:
            logger.debug("Could not warm graph for a user", exc_info=True)
    logger.info("Pre-built the graph for %d connected user(s)", warmed)
    return warmed


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
            setup_usage_recording,
        )

        setup_pii_middleware()  # must be first — Langfuse handler depends on the scrubber
        setup_langfuse_tracing()
        setup_token_budget_tracking()
        setup_usage_recording()
        setup_guardian_guardrails()
        return "ok"
    except Exception as exc:
        logger.warning("Telemetry setup failed: %s", exc)
        return f"warning: {exc}"


def is_ready() -> bool:
    """Return True if startup has completed."""
    return _startup_complete


def _warmup_graph_imports() -> str:
    """Eagerly import the graph dependency tree to avoid first-request penalty.

    The agent() graph factory lazy-imports deepagents, langchain, langgraph,
    and provider SDKs. Pre-importing them during startup moves the ~18s
    module-load cost out of the user-facing request path.
    """
    try:
        import deepagents  # noqa: F401

        from deep_agent.aegra.mcp import get_mcp_tools  # noqa: F401
        from deep_agent.aegra.mcp_resource_tools import (
            get_mcp_resource_tools,  # noqa: F401
        )
        from deep_agent.aegra.mcp_tool_auth import wrap_mcp_tools_for_auth  # noqa: F401
        from deep_agent.src.agent.config import agent_config  # noqa: F401
        from deep_agent.src.agent.config.model import parse_model_config  # noqa: F401
        from deep_agent.src.agent.provider_factory import (
            create_model_from_spec,  # noqa: F401
        )
        from deep_agent.src.cache.model_cache import (
            get_or_create_model_from_spec,  # noqa: F401
        )
        from deep_agent.src.infrastructure.async_tasks import (
            build_async_middleware,  # noqa: F401
        )
        from deep_agent.src.infrastructure.backend import (
            get_configured_backend,  # noqa: F401
        )
        from deep_agent.src.infrastructure.middleware import (
            build_middleware_list,  # noqa: F401
        )
        from deep_agent.src.infrastructure.providers import (
            register_profiles_from_config,  # noqa: F401
        )
        from deep_agent.src.infrastructure.subagents import load_subagents  # noqa: F401

        return "ok"
    except Exception as exc:
        logger.warning("Graph import warmup failed: %s", exc)
        return f"warning: {exc}"
