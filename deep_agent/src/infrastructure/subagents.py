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
from deep_agent.src.agent.config.model import ModelSpec, infer_provider, parse_model_config
from deep_agent.src.cache.model_cache import get_or_create_model_from_spec
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

    Subagents that don't specify a ``model`` inherit the orchestrator's model.
    Subagents that don't specify ``mcps`` inherit the orchestrator's MCPs
    (which determines tool visibility).

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

    orchestrator_cfg = agent_config.get_orchestrator_config()

    logger.info(f"Building {len(all_subagent_configs)} subagent(s)")

    subagents_list: list[Any] = []

    for name, agent_cfg in all_subagent_configs.items():
        _inherit_from_orchestrator(agent_cfg, orchestrator_cfg, name)
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


_DEFAULT_FALLBACK_MODEL = "gemini-3.1-pro-preview"


def _inherit_from_orchestrator(
    agent_cfg: dict[str, Any],
    orchestrator_cfg: dict[str, Any],
    name: str,
) -> None:
    """Fill in missing model/mcps from the parent orchestrator config.

    Mutates *agent_cfg* in place. Model inheritance follows these rules:
    1. If subagent has no model → use orchestrator model (no fallback)
    2. If subagent has model but no fallback → use orchestrator model as fallback
    3. If subagent has model with fallback → keep as-is

    Falls back to _DEFAULT_FALLBACK_MODEL when neither the subagent nor the
    orchestrator specifies a model.
    """
    parent_model = orchestrator_cfg.get("model")
    subagent_model = agent_cfg.get("model")

    if not subagent_model:
        # Case 1: No subagent model → inherit orchestrator model (no fallback)
        if parent_model:
            logger.info(
                "Subagent '%s' inheriting model from orchestrator: %s",
                name,
                parent_model,
            )
            agent_cfg["model"] = parent_model
        else:
            logger.warning(
                "Subagent '%s' has no model and orchestrator has no model — "
                "falling back to default: %s",
                name,
                _DEFAULT_FALLBACK_MODEL,
            )
            agent_cfg["model"] = _DEFAULT_FALLBACK_MODEL
    elif parent_model:
        # Case 2 & 3: Subagent has model → inject orchestrator as fallback if missing
        agent_cfg["model"] = _inject_fallback_if_missing(
            subagent_model, parent_model, name
        )

    if not agent_cfg.get("mcps"):
        parent_mcps = orchestrator_cfg.get("mcps", [])
        if parent_mcps:
            logger.info(
                "Subagent '%s' inheriting %d MCP(s) from orchestrator",
                name,
                len(parent_mcps),
            )
            agent_cfg["mcps"] = list(parent_mcps)


def _normalize_model_to_dict(
    raw_model: str | dict[str, Any],
    strip_fallback: bool = False,
) -> dict[str, Any] | Any:
    """Normalize model config (string or dict) to dict format.

    Args:
        raw_model: Model config in string or dict format.
        strip_fallback: If True, remove fallback key from dict configs.

    Returns:
        Normalized dict with provider and name keys, or original value if invalid type.
    """
    if isinstance(raw_model, str):
        return {
            "provider": infer_provider(raw_model).value,
            "name": raw_model,
        }
    elif isinstance(raw_model, dict):
        result = dict(raw_model)  # Copy to avoid mutation
        if strip_fallback and "fallback" in result:
            del result["fallback"]
        return result
    else:
        # Invalid type - return as-is and let parse_model_config fail later
        logger.warning(
            "Invalid model config type: %s, letting parse_model_config handle error",
            type(raw_model).__name__,
        )
        return raw_model


def _inject_fallback_if_missing(
    subagent_model: str | dict[str, Any],
    parent_model: str | dict[str, Any],
    name: str,
) -> str | dict[str, Any]:
    """Inject orchestrator model as fallback if subagent model has no fallback.

    Args:
        subagent_model: Subagent's model config (string or dict).
        parent_model: Orchestrator's model config.
        name: Subagent name (for logging).

    Returns:
        Normalized model config dict with fallback injected if needed,
        or original value if invalid type (will fail in parse_model_config).
    """
    # Normalize subagent model to dict
    model_dict = _normalize_model_to_dict(subagent_model)
    if not isinstance(model_dict, dict):
        return model_dict  # Invalid type, will fail later in parse_model_config

    # Case 3: Subagent already has fallback → keep as-is
    if "fallback" in model_dict:
        return model_dict

    # Case 2: Subagent has no fallback → inject orchestrator as fallback
    logger.debug(
        "Subagent '%s' inheriting orchestrator model as fallback: %s",
        name,
        parent_model,
    )

    # Normalize parent model to dict for fallback (strip nested fallback)
    fallback_dict = _normalize_model_to_dict(parent_model, strip_fallback=True)
    if not isinstance(fallback_dict, dict):
        # Invalid parent, skip fallback injection
        return model_dict

    model_dict["fallback"] = fallback_dict
    return model_dict


def _create_primary_model(spec: ModelSpec) -> object:
    """Create only the primary BaseChatModel from a ModelSpec, without fallback wrapper.

    Uses the model cache for efficient reuse. Fallbacks should be handled via
    LangChain's ModelFallbackMiddleware.

    Args:
        spec: Parsed model specification (fallback config ignored).

    Returns:
        A BaseChatModel instance for the primary model only.
    """

    # Create a spec without fallback for the primary model
    primary_spec = ModelSpec(provider=spec.provider, name=spec.name, fallback=None)

    # Use the cache to get or create the model
    return get_or_create_model_from_spec(primary_spec)


def _resolve_subagent_model(agent_cfg: dict[str, Any]) -> object:
    """Parse frontmatter model config and return only the primary BaseChatModel.

    Strips any fallback configuration since deepagents doesn't support RunnableWithFallbacks.
    Fallback handling should be done via LangChain's ModelFallbackMiddleware instead.
    """
    raw_model = agent_cfg.get("model")
    if raw_model is None:
        raise ValueError("missing required 'model' field in frontmatter")

    # Parse model spec (may include fallback config)
    spec = parse_model_config(raw_model)

    # Create only the primary model using the cache
    return _create_primary_model(spec)


def _format_model_log(spec: ModelSpec) -> str:
    """Format model spec for log messages."""
    return spec.display_name()


def _build_fallback_middleware(spec: ModelSpec) -> list[Any]:
    """Build ModelFallbackMiddleware with BaseChatModel if spec has fallback configured.

    Creates the fallback model using get_or_create_model_from_spec to preserve custom
    initialization logic (MAAS base URLs, Vertex credentials, etc) and enable caching.

    Args:
        spec: Parsed model specification.

    Returns:
        List containing ModelFallbackMiddleware if fallback exists, empty list otherwise.
    """
    if spec.fallback is None:
        return []

    try:
        from langchain.agents.middleware import ModelFallbackMiddleware
    except ImportError:
        logger.warning("ModelFallbackMiddleware not available, skipping fallback configuration")
        return []

    # Create fallback model using the model cache
    fallback_model = _create_primary_model(spec.fallback)

    middleware = ModelFallbackMiddleware(fallback_model)
    logger.info(
        "Configured fallback middleware: %s -> %s",
        spec.display_name(),
        spec.fallback.display_name(),
    )
    return [middleware]


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
    if not agent_cfg.get("model"):
        raise ValueError(
            f"Subagent '{name}' is missing required 'model' field in frontmatter"
        )

    spec = parse_model_config(agent_cfg["model"])
    logger.info("Subagent '%s' [default] using model: %s", name, _format_model_log(spec))

    tool_names: list[str] = agent_cfg.get("tools", [])
    mcp_names: list[str] = agent_cfg.get("mcps", [])

    if tool_names:
        resolved_tools: list[Any] = agent_config.resolve_tools(
            tool_names, tools, agent_name=name
        )
    elif mcp_names and tools:
        logger.info(
            "Subagent '%s' declared MCP servers %s but no explicit tools; "
            "exposing all %d available MCP tool(s)",
            name,
            mcp_names,
            len(tools),
        )
        resolved_tools = list(tools)
    else:
        resolved_tools = []

    skill_paths: list[str] = agent_cfg.get("skill_paths", [])

    # Build fallback middleware if spec has fallback configured
    fallback_mw = _build_fallback_middleware(spec)
    subagent_middleware = [*fallback_mw]

    subagent_params: dict[str, Any] = {
        "name": name,
        "model": _resolve_subagent_model(agent_cfg),
        "description": agent_cfg.get("description", ""),
        "system_prompt": agent_cfg.get("body", ""),
    }

    if resolved_tools:
        subagent_params["tools"] = resolved_tools
    if skill_paths:
        subagent_params["skills"] = skill_paths
    if subagent_middleware:
        subagent_params["middleware"] = subagent_middleware

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

    if not agent_cfg.get("model"):
        raise ValueError(
            f"Subagent '{name}' (compiled) is missing required 'model' field"
        )

    spec = parse_model_config(agent_cfg["model"])
    logger.info(
        "Subagent '%s' [compiled] using model: %s", name, _format_model_log(spec)
    )

    tool_names: list[str] = agent_cfg.get("tools", [])
    mcp_names: list[str] = agent_cfg.get("mcps", [])

    if tool_names:
        resolved_tools: list[Any] = agent_config.resolve_tools(
            tool_names, tools, agent_name=name
        )
    elif mcp_names and tools:
        logger.info(
            "Subagent '%s' [compiled] declared MCP servers %s but no explicit tools; "
            "exposing all %d available MCP tool(s)",
            name,
            mcp_names,
            len(tools),
        )
        resolved_tools = list(tools)
    else:
        resolved_tools = []
    skill_paths: list[str] = agent_cfg.get("skill_paths", [])

    # Build fallback middleware if spec has fallback configured
    fallback_mw = _build_fallback_middleware(spec)
    compiled_middleware = [*fallback_mw]

    create_kwargs = {
        "name": name,
        "model": _resolve_subagent_model(agent_cfg),
        "system_prompt": agent_cfg.get("body", ""),
        "tools": resolved_tools or None,
        "skills": skill_paths or None,
        "backend": get_configured_backend(),
    }

    if compiled_middleware:
        create_kwargs["middleware"] = compiled_middleware

    compiled_graph = create_deep_agent(**create_kwargs)

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
