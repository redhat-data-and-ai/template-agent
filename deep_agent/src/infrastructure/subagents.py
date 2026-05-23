"""Subagent loading from configuration files.

This module builds SubAgent instances from the markdown configuration files in
config/subagents/. It reads each subagent's config, resolves their tools
and skills, creates appropriate LLM instances, and returns ready-to-use SubAgent
objects for the orchestrator.

Supports three agent types via the ``type`` field in frontmatter:
    - ``default``: Standard SubAgent (in-process, synchronous delegation)
    - ``compiled``: CompiledSubAgent (pre-compiled graph, reused across requests)
    - ``async``: AsyncSubAgent (remote Agent Protocol server, background tasks)

Functions:
    load_subagents: Build all subagents from config/subagents/*.md
"""

from typing import Any

from deepagents import SubAgent
from deepagents.middleware.subagents import CompiledSubAgent

try:
    from deepagents.middleware.async_subagents import AsyncSubAgent
except ImportError:
    AsyncSubAgent = None

from deep_agent.src.agent.config import agent_config
from deep_agent.src.cache.model_cache import get_or_create_model
from deep_agent.src.exceptions import LLMError, SubAgentError
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

VALID_AGENT_TYPES = ("default", "compiled", "async")


def load_subagents(
    tools: list[Any],
) -> list[Any] | None:
    """Build subagents from pre-loaded configurations.

    Reads the ``type`` field from each subagent's frontmatter to determine
    which agent class to construct:
    - ``default`` / missing → SubAgent (standard in-process delegation)
    - ``compiled`` → CompiledSubAgent (pre-compiled graph as Runnable)
    - ``async`` → AsyncSubAgent (remote Agent Protocol server)

    Args:
        tools: List of available MCP tools.

    Returns:
        List of configured subagent instances, or None if no subagents configured.

    Raises:
        SubAgentError: If a subagent fails to build (missing model, bad config).
    """
    all_subagent_configs: dict[str, dict[str, Any]] = (
        agent_config.get_all_subagent_configs()
    )

    if not all_subagent_configs:
        logger.warning("No subagent configurations found")
        return None

    logger.info(f"Building {len(all_subagent_configs)} subagent(s)")

    subagents_list: list[Any] = []

    for name, agent_cfg in all_subagent_configs.items():
        try:
            sa = _build_single_subagent(name, agent_cfg, tools)
            subagents_list.append(sa)
        except (ValueError, LLMError) as e:
            raise SubAgentError(f"Failed to build subagent '{name}': {e}") from e
        except Exception as e:
            raise SubAgentError(
                f"Unexpected error building subagent '{name}': {e}"
            ) from e

    logger.info(f"Built {len(subagents_list)} subagent(s) successfully")
    return subagents_list


def _build_single_subagent(
    name: str,
    agent_cfg: dict[str, Any],
    tools: list[Any],
) -> Any:
    """Build a single subagent from its configuration.

    Dispatches to the appropriate builder based on the ``type`` field.

    Args:
        name: Subagent name (from config filename).
        agent_cfg: Parsed frontmatter config for this subagent.
        tools: Available MCP tools for tool resolution.

    Returns:
        Configured subagent instance (SubAgent, CompiledSubAgent, or AsyncSubAgent).

    Raises:
        ValueError: If required fields are missing or type is invalid.
        LLMError: If model creation fails.
    """
    agent_type: str = agent_cfg.get("type", "default")
    if agent_type not in VALID_AGENT_TYPES:
        raise ValueError(
            f"Subagent '{name}' has invalid type '{agent_type}'. "
            f"Valid types: {VALID_AGENT_TYPES}"
        )

    if agent_type == "async":
        return _build_async_subagent(name, agent_cfg)
    if agent_type == "compiled":
        return _build_compiled_subagent(name, agent_cfg, tools)
    return _build_default_subagent(name, agent_cfg, tools)


def _build_default_subagent(
    name: str,
    agent_cfg: dict[str, Any],
    tools: list[Any],
) -> SubAgent:
    """Build a standard SubAgent (in-process delegation)."""
    model_name: str | None = agent_cfg.get("model")
    if not model_name:
        raise ValueError(
            f"Subagent '{name}' is missing required 'model' field in frontmatter"
        )

    logger.info(f"Subagent '{name}' [default] using model: {model_name}")

    tool_names: list[str] = agent_cfg.get("tools", [])
    resolved_tools: list[Any] = (
        agent_config.resolve_tools(tool_names, tools, agent_name=name)
        if tool_names
        else []
    )

    skill_paths: list[str] = agent_cfg.get("skill_paths", [])

    subagent_params: dict[str, Any] = {
        "name": name,
        "model": get_or_create_model(model_name=model_name),
        "description": agent_cfg.get("description", ""),
        "system_prompt": agent_cfg.get("body", ""),
    }

    if resolved_tools:
        subagent_params["tools"] = resolved_tools
    if skill_paths:
        subagent_params["skills"] = skill_paths

    return SubAgent(**subagent_params)


def _build_compiled_subagent(
    name: str,
    agent_cfg: dict[str, Any],
    tools: list[Any],
) -> CompiledSubAgent:
    """Build a CompiledSubAgent (pre-compiled graph as Runnable).

    Creates a full deep agent graph for this subagent and wraps it as a
    CompiledSubAgent. The compiled graph is reused across requests,
    providing better performance for frequently-invoked subagents.
    """
    from deepagents import create_deep_agent

    from deep_agent.src.infrastructure.backend import get_configured_backend

    model_name: str | None = agent_cfg.get("model")
    if not model_name:
        raise ValueError(
            f"Subagent '{name}' (compiled) is missing required 'model' field"
        )

    logger.info(f"Subagent '{name}' [compiled] using model: {model_name}")

    tool_names: list[str] = agent_cfg.get("tools", [])
    resolved_tools: list[Any] = (
        agent_config.resolve_tools(tool_names, tools, agent_name=name)
        if tool_names
        else []
    )
    skill_paths: list[str] = agent_cfg.get("skill_paths", [])

    compiled_graph = create_deep_agent(
        name=name,
        model=get_or_create_model(model_name=model_name),
        system_prompt=agent_cfg.get("body", ""),
        tools=resolved_tools or None,
        skills=skill_paths or None,
        backend=get_configured_backend(),
    )

    return CompiledSubAgent(
        name=name,
        description=agent_cfg.get("description", ""),
        runnable=compiled_graph,
    )


def _build_async_subagent(
    name: str,
    agent_cfg: dict[str, Any],
) -> Any:
    """Build an AsyncSubAgent (remote Agent Protocol server).

    Requires ``graph_id`` in frontmatter. Optionally accepts ``url``
    for the remote endpoint.

    Auth headers are resolved from environment variables (OpenShift Secrets),
    never from frontmatter config. The env var name follows the convention:
    ``ASYNC_SUBAGENT_<NAME>_TOKEN`` (uppercased, hyphens → underscores).
    """
    if AsyncSubAgent is None:
        raise ValueError(
            f"Subagent '{name}' (async) requires deepagents with async support. "
            "Upgrade deepagents or remove this subagent config."
        )

    graph_id: str | None = agent_cfg.get("graph_id")
    if not graph_id:
        raise ValueError(
            f"Subagent '{name}' (async) is missing required 'graph_id' field"
        )

    logger.info(f"Subagent '{name}' [async] connecting to graph: {graph_id}")

    params: dict[str, Any] = {
        "name": name,
        "description": agent_cfg.get("description", ""),
        "graph_id": graph_id,
    }

    url: str | None = agent_cfg.get("url")
    if url:
        params["url"] = url

    headers = _resolve_async_headers(name)
    if headers:
        params["headers"] = headers

    return AsyncSubAgent(**params)


def _resolve_async_headers(name: str) -> dict[str, str] | None:
    """Resolve auth headers for an async subagent from environment.

    Convention: ASYNC_SUBAGENT_<NAME>_TOKEN env var → Authorization header.
    Secrets come from OpenShift Secrets mounted as env vars.
    """
    import os

    env_key = f"ASYNC_SUBAGENT_{name.upper().replace('-', '_')}_TOKEN"
    token = os.environ.get(env_key)
    if token:
        return {"Authorization": f"Bearer {token}"}
    return None
