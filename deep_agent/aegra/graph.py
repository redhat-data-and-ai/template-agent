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
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from langgraph_sdk.runtime import ServerRuntime

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_REPO_ROOT = Path(__file__).resolve().parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("PYTHONPATH", str(_REPO_ROOT))

_startup_done = False  # noqa: E402

_graph_cache: dict[str, Any] = {}
_graph_cache_ts: dict[str, float] = {}
_MAX_GRAPH_CACHE_SIZE = 16

_SAFETY_STOP_INSTRUCTION = """
## Content Safety (Enforced by Framework)

If any tool, subagent, or skill response contains the phrase "blocked due to content safety policy issue":
- **STOP ALL WORK IMMEDIATELY.**
- Do NOT retry the tool or subagent.
- Do NOT rephrase the request and try again.
"""


def _append_safety_stop_instruction(system_prompt: str) -> str:
    """Append the framework-level content safety stop instruction to any system prompt.

    Appended unconditionally when Guardian is enabled so every agent built with
    this framework gets the hard-stop rule without requiring users to add it to
    their own PROMPT.md.
    """
    return system_prompt.rstrip() + "\n" + _SAFETY_STOP_INSTRUCTION


_MEMORY_INSTRUCTIONS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "config"
    / "agent"
    / "runtime"
    / "memory_instructions.j2"
)

_memory_instructions_cache: str | None = None


def _load_memory_instructions() -> str:
    """Load the memory instructions J2 template (cached after first read)."""
    global _memory_instructions_cache  # noqa: PLW0603
    if _memory_instructions_cache is None:
        try:
            _memory_instructions_cache = _MEMORY_INSTRUCTIONS_PATH.read_text()
        except FileNotFoundError:
            logger.warning(
                "Memory instructions template not found at %s",
                _MEMORY_INSTRUCTIONS_PATH,
            )
            _memory_instructions_cache = ""
    return _memory_instructions_cache


def _append_memory_instructions(system_prompt: str) -> str:
    """Append memory instructions to the system prompt when memory is enabled."""
    instructions = _load_memory_instructions()
    if not instructions:
        return system_prompt
    return system_prompt.rstrip() + "\n\n---\n\n" + instructions


def invalidate_graph_cache() -> None:
    """Clear the compiled graph cache (e.g. after MCP OAuth connect)."""
    global _graph_cache, _graph_cache_ts  # noqa: PLW0603
    _graph_cache.clear()
    _graph_cache_ts.clear()


def _graph_fingerprint(
    model_name: str,
    system_prompt: str,
    tool_names: list[str],
    hitl_enabled: bool = False,
    hitl_mode: str = "all",
    hitl_exclude: list[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """Stable fingerprint for graph cache keying."""
    hitl_flag = (
        f"hitl={int(hitl_enabled)}"
        f",mode={hitl_mode}"
        f",exclude={','.join(sorted(hitl_exclude or []))}"
    )
    model_flag = f"temp={temperature},max_tokens={max_tokens}"
    raw = f"{model_name}\0{system_prompt}\0{','.join(sorted(tool_names))}\0{hitl_flag}\0{model_flag}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


async def _ensure_startup() -> None:  # noqa: E402
    """Run startup orchestrator once (lazy, on first request)."""
    global _startup_done  # noqa: PLW0603
    if _startup_done:
        return
    from deep_agent.aegra.startup import run_startup

    await run_startup()
    _startup_done = True


def _resolve_personalization_uid(user: Any, token: str | None) -> str | None:
    """Extract the user ID that the BFF proxy uses for personalization.

    The BFF sends ``X-User-ID: preferred_username`` while the Aegra auth
    handler sets ``identity`` to the JWT ``sub`` (a UUID).  Rules are stored
    under the BFF's value, so the graph builder must resolve the same ID.
    """
    if not token:
        return None
    try:
        import base64

        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        return (
            claims.get("preferred_username")
            or claims.get("sub")
            or getattr(user, "identity", None)
        )
    except Exception:
        return None


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
    from deep_agent.aegra.mcp_tool_auth import wrap_mcp_tools_for_auth
    from deep_agent.src.agent.config import agent_config
    from deep_agent.src.infrastructure.async_tasks import build_async_middleware
    from deep_agent.src.infrastructure.backend import get_configured_backend
    from deep_agent.src.infrastructure.providers import (
        register_profiles_from_config,
    )
    from deep_agent.src.infrastructure.subagents import load_subagents

    user = getattr(runtime, "user", None)
    sso_token = getattr(user, "access_token", None) if user else None
    refresh_token = getattr(user, "refresh_token", None) if user else None

    if sso_token:
        sso_token = await refresh_access_token(sso_token, refresh_token)

    user_identity = getattr(user, "identity", None) if user else None

    set_mcp_auth_context(sso_token, refresh_token, user_identity)
    orchestrator_cfg = agent_config.get_orchestrator_config()
    agent_name = orchestrator_cfg.get("name", "orchestrator")
    orch_model_raw = orchestrator_cfg.get("model", "gemini-3.1-pro-preview")
    system_prompt = orchestrator_cfg.get("body", "")
    skill_paths = orchestrator_cfg.get("skill_paths", [])
    tool_names = orchestrator_cfg.get("tools", [])
    mcp_server_names = orchestrator_cfg.get("mcps", [])

    # Resolve the personalization user ID to match what the BFF proxy
    # sends as X-User-ID (preferred_username from JWT, not sub UUID).
    personalization_uid = _resolve_personalization_uid(user, sso_token) or user_identity

    if not personalization_uid:
        logger.info("No user identity — skipping personalization rules")
    elif not settings.USER_RULES_ENABLED:
        logger.info("User rules feature disabled — skipping rule injection")
    else:
        try:
            from deep_agent.src.cache.personalization_cache import (
                get_rules,
                set_rules,
            )
            from deep_agent.src.personalization.injector import inject_rules
            from deep_agent.src.personalization.repository import (
                PersonalizationRepository,
            )
            from deep_agent.src.settings import settings as app_settings

            cached = await get_rules(personalization_uid)
            if cached is not None:
                rule_contents = [r["content"] for r in cached]
            else:
                repo = PersonalizationRepository(app_settings.database_uri)
                rules = await repo.list_rules(personalization_uid, active_only=True)
                rule_contents = [r.content for r in rules]
                await set_rules(
                    personalization_uid,
                    [{"content": r.content} for r in rules],
                )

            system_prompt = inject_rules(system_prompt, rule_contents)
            if rule_contents:
                logger.debug(
                    "Personalization: injected %d rule(s) for user",
                    len(rule_contents),
                )
            else:
                logger.debug("Personalization: no active rules found")
        except Exception:
            logger.warning(
                "Personalization unavailable, continuing without rules",
                exc_info=True,
            )

    # Parse orchestrator model to support provider
    from deep_agent.src.agent.config.model import parse_model_config
    from deep_agent.src.cache.model_cache import get_or_create_model_from_spec

    orch_spec = parse_model_config(orch_model_raw)
    model_name = orch_spec.name  # For logging and cache key

    model_params = agent_config.get_model_params(orch_spec.name)
    orch_temperature = model_params.get("temperature", 0.0)
    orch_max_tokens = model_params.get("max_tokens") or None

    logger.info(
        "Building agent '%s' (model=%s, provider=%s, temp=%s, max_tokens=%s, mcp_auth=%s)",
        agent_name,
        orch_spec.name,
        orch_spec.provider.value,
        orch_temperature,
        orch_max_tokens or "default",
        bool(sso_token),
    )

    model = get_or_create_model_from_spec(
        orch_spec,
        temperature=float(orch_temperature),
        max_output_tokens=int(orch_max_tokens) if orch_max_tokens else None,
    )

    providers_config = agent_config.get_providers_config()
    register_profiles_from_config(providers_config)

    mcp_tools = await get_mcp_tools(
        sso_token=sso_token,
        server_names=mcp_server_names or None,
        user_id=user_identity,
    )
    mcp_tools = wrap_mcp_tools_for_auth(mcp_tools)

    all_available_tools = list(mcp_tools)
    tools = agent_config.resolve_tools(
        tool_names, all_available_tools, agent_name=agent_name
    )
    if mcp_server_names and mcp_tools:
        resolved_names = {t.name for t in tools}
        extra = []
        for tool in mcp_tools:
            if tool.name not in resolved_names:
                resolved_names.add(tool.name)
                extra.append(tool)
        if extra:
            logger.info(
                "Agent '%s' adding %d MCP tool(s) from servers %s",
                agent_name,
                len(extra),
                mcp_server_names,
            )
            tools.extend(extra)

    from deep_agent.src.infrastructure.middleware import (
        build_middleware_list,
        resolve_memory_param,
    )

    middleware_overrides = orchestrator_cfg.get("middleware")
    resolved_mw = agent_config.resolve_agent_middleware(
        model_name, middleware_overrides
    )

    user_memory_enabled = settings.MEMORY_ENABLED
    if settings.MEMORY_ENABLED and personalization_uid:
        try:
            from deep_agent.src.personalization.repository import (
                PersonalizationRepository,
            )
            from deep_agent.src.settings import settings as app_settings

            pref_repo = PersonalizationRepository(app_settings.database_uri)
            user_prefs = await pref_repo.get_preferences(personalization_uid)
            user_memory_enabled = user_prefs.memory_enabled
            if not user_memory_enabled:
                logger.info(
                    "Memory disabled by user preference for %s", personalization_uid
                )
        except Exception:
            logger.debug(
                "Could not load user preferences, using global setting", exc_info=True
            )

    if user_memory_enabled:
        system_prompt = _append_memory_instructions(system_prompt)

    hitl = getattr(resolved_mw, "human_approval", None)

    cache_key = _graph_fingerprint(
        model_name,
        system_prompt,
        [t.name for t in tools],
        hitl_enabled=hitl.enabled if hitl else False,
        hitl_mode=hitl.mode if hitl else "",
        hitl_exclude=hitl.exclude if hitl else [],
        temperature=float(orch_temperature),
        max_tokens=int(orch_max_tokens) if orch_max_tokens else None,
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

    middleware = build_middleware_list(
        resolved_mw,
        model=model,
        backend=backend,
        mcp_tool_names=frozenset(t.name for t in mcp_tools),
    )
    memory = resolve_memory_param(resolved_mw) if user_memory_enabled else None

    if skill_paths and resolved_mw.skills_enabled:
        from deep_agent.src.agent.config.resolver import to_virtual_skill_paths

        skills_param = to_virtual_skill_paths(skill_paths)
    else:
        skills_param = None

    async_mw = build_async_middleware(subagents, providers_config.async_tasks)
    if async_mw is not None:
        middleware.append(async_mw)

    from deep_agent.src.settings import settings as app_settings

    guardrail_cfg = agent_config.get_guardrails_config()
    guardian_active = guardrail_cfg.enabled and bool(app_settings.GUARDIAN_API_BASE)

    if guardian_active:
        from deep_agent.src.guardrails.tool_proxy import wrap_tools

        tools = wrap_tools(tools)
        system_prompt = _append_safety_stop_instruction(system_prompt)

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

    if hitl and hitl.enabled and "interrupt_on" in create_sig.parameters:
        try:
            from deep_agent.src.agent.config.hitl import build_interrupt_on

            interrupt_on = build_interrupt_on(hitl, tools)
            if interrupt_on:
                create_kwargs["interrupt_on"] = interrupt_on
            else:
                logger.warning(
                    "HITL is enabled but interrupt_on is empty — "
                    "no tool calls will be interrupted (all tools may be excluded)"
                )
        except ImportError:
            logger.warning(
                "HITL is enabled but hitl module not available — "
                "upgrade deepagents to activate human-in-the-loop approval"
            )

    _inner_graph = create_deep_agent(**create_kwargs)

    compiled = _inner_graph

    from deep_agent.src.pii import get_scrubber

    if get_scrubber() is not None:
        from deep_agent.src.pii.runnable import PIIAwareRunnable

        compiled = PIIAwareRunnable(compiled)
        logger.info("graph_pii_enabled: wrapped with PIIAwareRunnable")

    if guardian_active:
        from deep_agent.aegra.safety import SafetyAwareRunnable

        compiled = SafetyAwareRunnable(compiled, outermost=True)
        logger.info(
            "graph_guardian_enabled: wrapped with SafetyAwareRunnable and GuardianToolProxy"
        )

    # ── Lifecycle persistence: hook register/deregister around invocation ──
    try:
        from deep_agent.src.settings import settings as app_settings

        if app_settings.LIFECYCLE_PERSISTENCE_ENABLED:
            from deep_agent.aegra.lifecycle import build_execution_context

            user_id_val = str(user_identity) if user_identity else ""
            assistant_id_val = getattr(runtime, "assistant_id", None) or ""

            _exec_ctx = build_execution_context(
                model_name=model_name,
                config_hash=cache_key,
                assistant_id=assistant_id_val,
                user_id=user_id_val,
                refresh_token=refresh_token,
                mcp_server_names=mcp_server_names,
                orchestrator_config=orchestrator_cfg,
            )

            setattr(compiled, "_lifecycle_enabled", True)
            setattr(compiled, "_lifecycle_config_hash", cache_key)
            setattr(compiled, "_lifecycle_user_id", user_id_val)
            setattr(compiled, "_lifecycle_assistant_id", assistant_id_val)
            setattr(compiled, "_lifecycle_exec_ctx", _exec_ctx)

            logger.debug("Lifecycle persistence enabled")
    except ImportError:
        pass

    _graph_cache[cache_key] = compiled
    _graph_cache_ts[cache_key] = time.time()

    if len(_graph_cache) > _MAX_GRAPH_CACHE_SIZE:
        oldest_key = min(_graph_cache_ts, key=_graph_cache_ts.get)  # type: ignore[arg-type]
        _graph_cache.pop(oldest_key, None)
        _graph_cache_ts.pop(oldest_key, None)

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
