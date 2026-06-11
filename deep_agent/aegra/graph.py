"""Graph factory for Aegra deployment.

This module exports an **async graph factory** that Aegra invokes
**per-request**.  The factory extracts the calling user's SSO token
from the ``ServerRuntime`` and passes it to MCP servers, so tool
calls are authenticated end-to-end with the user's own credentials.

``aegra.json`` references this as::

    "graphs": {"agent": "./deep_agent/aegra/graph.py:agent"}

Aegra detects the ``ServerRuntime`` parameter and classifies
``agent`` as a 1-param runtime factory.  On each request:

1. The auth handler validates the JWT and stores ``access_token``
   and ``refresh_token`` on the ``User`` model.
2. Aegra builds a ``ServerRuntime(user=user, …)`` and calls
   ``agent(runtime)`` → coroutine → ``await``-ed → compiled graph.
3. The graph is injected with Aegra's Postgres checkpointer/store
   before being used for the run.

For schema-only calls (LangGraph Studio, assistant listing) the
factory is invoked with ``user=None``; MCP tools are skipped and
the graph is built with only built-in tools.

Aegra automatically provides:

- Postgres-backed checkpointer (conversation persistence)
- Thread/run/assistant management API
- SSE streaming endpoint
- Worker architecture with Redis job queue
"""

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

from langgraph_sdk.runtime import ServerRuntime

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_REPO_ROOT = Path(__file__).resolve().parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("PYTHONPATH", str(_REPO_ROOT))

_startup_done = False  # noqa: E402

_graph_cache: dict[str, Any] = {}
_graph_cache_ts: dict[str, float] = {}


def _graph_fingerprint(
    model_name: str,
    system_prompt: str,
    tool_names: list[str],
) -> str:
    """Stable fingerprint for graph cache keying."""
    raw = f"{model_name}\0{system_prompt}\0{','.join(sorted(tool_names))}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


async def _ensure_startup() -> None:  # noqa: E402
    """Run startup orchestrator once (lazy, on first request)."""
    global _startup_done  # noqa: PLW0603
    if _startup_done:
        return
    from deep_agent.aegra.startup import run_startup

    await run_startup()
    _startup_done = True


async def agent(runtime: ServerRuntime) -> Any:
    """Async graph factory — invoked per-request by Aegra.

    Extracts the user's SSO token from the runtime and forwards it
    to MCP servers so external tool calls carry the user's identity.

    When ``runtime.user`` is ``None`` (schema-extraction calls), MCP
    tools are skipped and the graph is built with built-in tools only.

    Args:
        runtime: Aegra ``ServerRuntime`` containing the authenticated
            ``User`` with ``access_token`` / ``refresh_token`` extras.

    Returns:
        A compiled deep-agent graph (``CompiledStateGraph``).
    """
    await _ensure_startup()

    from deepagents import create_deep_agent

    from deep_agent.aegra.mcp import (
        get_mcp_tools,
        refresh_access_token,
        set_mcp_auth_context,
    )
    from deep_agent.src.agent.config import agent_config
    from deep_agent.src.infrastructure.async_tasks import build_async_middleware
    from deep_agent.src.infrastructure.backend import get_configured_backend
    from deep_agent.src.infrastructure.providers import (
        register_profiles_from_config,
        resolve_model_from_config,
    )
    from deep_agent.src.infrastructure.subagents import load_subagents

    user = getattr(runtime, "user", None)
    sso_token = getattr(user, "access_token", None) if user else None
    refresh_token = getattr(user, "refresh_token", None) if user else None

    if sso_token:
        sso_token = await refresh_access_token(sso_token, refresh_token)

    set_mcp_auth_context(sso_token, refresh_token)

    orchestrator_cfg = agent_config.get_orchestrator_config()
    agent_name = orchestrator_cfg.get("name", "orchestrator")
    model_name = orchestrator_cfg.get("model", "gemini-3.1-pro-preview")
    system_prompt = orchestrator_cfg.get("body", "")
    skill_paths = orchestrator_cfg.get("skill_paths", [])
    tool_names = orchestrator_cfg.get("tools", [])
    mcp_server_names = orchestrator_cfg.get("mcps", [])

    user_identity = getattr(user, "identity", None) if user else None
    if user_identity:
        try:
            from deep_agent.src.cache.personalization_cache import (
                get_personalization,
                set_personalization,
            )
            from deep_agent.src.memory.config import memory_settings
            from deep_agent.src.personalization.injector import inject_personalization
            from deep_agent.src.personalization.repository import (
                PersonalizationRepository,
            )
            from deep_agent.src.settings import settings as app_settings

            cached = await get_personalization(user_identity)
            if cached is not None:
                mem_contents = [m["content"] for m in cached[0]]
                rule_contents = [r["content"] for r in cached[1]]
            else:
                repo = PersonalizationRepository(app_settings.database_uri)
                max_inject = memory_settings.MEMORY_MAX_INJECT
                memories = await repo.list_top_memories(user_identity, limit=max_inject)
                rules = await repo.list_rules(user_identity, active_only=True)
                mem_contents = [m.content for m in memories]
                rule_contents = [r.content for r in rules]
                await set_personalization(
                    user_identity,
                    [{"content": m.content} for m in memories],
                    [{"content": r.content} for r in rules],
                )

            system_prompt = inject_personalization(
                system_prompt,
                mem_contents,
                rule_contents,
            )
            if mem_contents or rule_contents:
                logger.info(
                    "Personalization injected: %d memories, %d rules",
                    len(mem_contents),
                    len(rule_contents),
                )
        except Exception:
            logger.debug(
                "Personalization unavailable, continuing without", exc_info=True
            )

    logger.info(
        "Building agent '%s' (model=%s, mcp_auth=%s)",
        agent_name,
        model_name,
        bool(sso_token),
    )

    providers_config = agent_config.get_providers_config()
    register_profiles_from_config(providers_config)
    model = resolve_model_from_config(model_name, providers_config)

    mcp_tools = await get_mcp_tools(
        sso_token=sso_token,
        server_names=mcp_server_names or None,
    )
    tools = agent_config.resolve_tools(tool_names, mcp_tools, agent_name=agent_name)
    if not tools and not tool_names and mcp_server_names and mcp_tools:
        logger.info(
            "Agent '%s' declared MCP servers %s but no explicit tools; exposing all %d MCP tool(s)",
            agent_name,
            mcp_server_names,
            len(mcp_tools),
        )
        tools = mcp_tools

    cache_key = _graph_fingerprint(
        model_name,
        system_prompt,
        [t.name for t in tools],
    )
    now = time.time()
    graph_ttl = float(agent_config.get_cache_config().graph.ttl)
    cached = _graph_cache.get(cache_key)
    if cached is not None and (now - _graph_cache_ts.get(cache_key, 0)) < graph_ttl:
        age = now - _graph_cache_ts[cache_key]
        logger.warning("Graph cache HIT (age=%.1fs) — skipping rebuild", age)
        return cached

    logger.warning("Graph cache MISS — full rebuild")

    subagents = load_subagents(tools=mcp_tools)
    backend = get_configured_backend()

    from deep_agent.src.infrastructure.middleware import (
        build_middleware_list,
        resolve_memory_param,
    )

    middleware_overrides = orchestrator_cfg.get("middleware")
    resolved_mw = agent_config.resolve_agent_middleware(
        model_name, middleware_overrides
    )
    middleware = build_middleware_list(resolved_mw, model=model, backend=backend)
    memory = resolve_memory_param(resolved_mw)
    skills_param = skill_paths if resolved_mw.skills_enabled else None

    async_mw = build_async_middleware(subagents, providers_config.async_tasks)
    if async_mw is not None:
        middleware.append(async_mw)

    create_kwargs: dict[str, Any] = {
        "name": agent_name,
        "model": model,
        "system_prompt": system_prompt,
        "skills": skills_param,
        "tools": tools,
        "subagents": subagents,
        "backend": backend,
        "middleware": middleware,
        "memory": memory,
    }

    import inspect

    create_sig = inspect.signature(create_deep_agent)
    if "permissions" in create_sig.parameters:
        try:
            from deep_agent.src.infrastructure.permissions import build_permissions

            permissions = build_permissions(agent_config.get_filesystem_config())
            if permissions:
                create_kwargs["permissions"] = permissions
        except (ImportError, TypeError):
            pass

    compiled = create_deep_agent(**create_kwargs)

    _graph_cache[cache_key] = compiled
    _graph_cache_ts[cache_key] = time.time()

    tool_count = len(tools)
    sub_count = len(subagents) if subagents else 0
    logger.info(
        "Agent ready: %d tool(s), %d subagent(s), %d middleware, mcp_auth=%s",
        tool_count,
        sub_count,
        len(middleware),
        bool(sso_token),
    )

    return compiled
