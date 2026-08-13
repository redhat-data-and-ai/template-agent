"""Middleware builder for deepagents integration.

Converts ResolvedMiddlewareConfig into a list of AgentMiddleware instances
that can be passed to create_deep_agent(middleware=...).

This module is the bridge between declarative YAML config and the deepagents
middleware API. Template-agent users never import or call this directly.
"""

from __future__ import annotations

import importlib
from typing import Any

from deep_agent.src.agent.config.middleware import ResolvedMiddlewareConfig
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


def build_audit_middleware(
    *,
    mcp_tool_names: frozenset[str] | None = None,
    subagent: str | None = None,
    agent: str | None = None,
) -> Any | None:
    """Return AuditMiddleware when platform audit is enabled, else None."""
    from deep_agent.src.audit.config import is_audit_enabled

    if not is_audit_enabled():
        return None

    from deep_agent.src.audit.middleware import AuditMiddleware

    return AuditMiddleware(
        mcp_tool_names=mcp_tool_names or frozenset(),
        subagent=subagent,
        agent=agent,
    )


def _mcp_tool_names_from_tools(tools: list[Any]) -> frozenset[str]:
    return frozenset(getattr(t, "name", "") for t in tools if getattr(t, "name", None))


def build_middleware_list(
    resolved: ResolvedMiddlewareConfig,
    *,
    model: Any | None = None,
    backend: Any | None = None,
    mcp_tool_names: frozenset[str] | None = None,
) -> list[Any]:
    """Build a list of middleware instances from resolved config.

    Only instantiates middleware that deepagents does NOT auto-include.
    The auto-included middleware (SubAgentMiddleware, SummarizationMiddleware,
    PatchToolCallsMiddleware, FilesystemMiddleware, TodoListMiddleware) are
    handled by create_deep_agent() itself.

    Args:
        resolved: Fully resolved middleware configuration for this agent.
        model: Chat model instance for SummarizationToolMiddleware.
        backend: Backend instance for SummarizationToolMiddleware.
        mcp_tool_names: MCP tool names for platform audit classification.

    Returns:
        List of middleware instances to pass as middleware= parameter.
        Empty list means only deepagents defaults apply.
    """
    middlewares: list[Any] = []

    audit_mw = build_audit_middleware(mcp_tool_names=mcp_tool_names)
    if audit_mw is not None:
        middlewares.append(audit_mw)

    if not settings.MIDDLEWARE_ENABLED:
        logger.info("Middleware disabled via MIDDLEWARE_ENABLED=false")
        return middlewares

    from deep_agent.src.agent.config import agent_config

    pii_cfg = agent_config.get_custom_pii_config()

    if pii_cfg.enabled:
        _append_if_built(middlewares, _build_custom_pii_middleware())

    if resolved.summarization_tool_enabled:
        _append_if_built(
            middlewares,
            _build_summarization_tool_middleware(model=model, backend=backend),
        )

    _append_guardrails(middlewares, resolved)

    for dotted_path in resolved.extra_middleware:
        _append_if_built(middlewares, _import_middleware(dotted_path))

    if middlewares:
        logger.info("Built %d extra middleware instance(s)", len(middlewares))
    return middlewares


def _append_if_built(target: list[Any], mw: Any | None) -> None:
    """Append middleware to list if it was built successfully."""
    if mw is not None:
        target.append(mw)


def _append_guardrails(target: list[Any], resolved: ResolvedMiddlewareConfig) -> None:
    """Build and append all production guardrail middleware."""
    if resolved.model_call_limit.enabled:
        _append_if_built(
            target, _build_model_call_limit(resolved.model_call_limit.run_limit)
        )

    if resolved.tool_call_limit.enabled:
        _append_if_built(
            target, _build_tool_call_limit(resolved.tool_call_limit.run_limit)
        )

    if resolved.model_retry.enabled:
        _append_if_built(target, _build_model_retry(resolved.model_retry))

    if resolved.model_fallback.enabled and resolved.model_fallback.fallback_model:
        _append_if_built(
            target, _build_model_fallback(resolved.model_fallback.fallback_model)
        )

    if resolved.tool_retry.enabled and resolved.tool_retry.tools:
        _append_if_built(target, _build_tool_retry(resolved.tool_retry))

    from deep_agent.src.agent.config import agent_config

    pii_cfg = agent_config.get_custom_pii_config()
    if pii_cfg.enabled and pii_cfg.rules:
        target.extend(_build_pii_middleware(pii_cfg))


def build_excluded_middleware(
    resolved: ResolvedMiddlewareConfig,
) -> list[str]:
    """Build the list of middleware to exclude from deepagents defaults.

    Used when registering HarnessProfiles or passing to create_deep_agent
    via profile configuration.

    Args:
        resolved: Resolved middleware config.

    Returns:
        List of middleware class names to exclude.
    """
    excluded: list[str] = list(resolved.excluded_middleware)

    if not resolved.patch_tool_calls_enabled:
        excluded.append("PatchToolCallsMiddleware")

    return excluded


def resolve_memory_param(
    resolved: ResolvedMiddlewareConfig,
) -> list[str] | None:
    """Resolve the memory= parameter for create_deep_agent().

    MemoryMiddleware is auto-included when memory= is provided.
    This function returns the namespaces list or None to disable.

    Args:
        resolved: Resolved middleware config.

    Returns:
        List of memory namespace strings, or None if memory is disabled.
    """
    if not settings.MIDDLEWARE_ENABLED:
        return None
    if not resolved.memory_enabled:
        return None
    return resolved.memory_namespaces or None


def _build_model_call_limit(run_limit: int) -> Any | None:
    """Build ModelCallLimitMiddleware to cap LLM calls per run."""
    try:
        from langchain.agents.middleware import ModelCallLimitMiddleware

        return ModelCallLimitMiddleware(run_limit=run_limit)
    except ImportError:
        logger.debug("ModelCallLimitMiddleware not available")
        return None


def _build_tool_call_limit(run_limit: int) -> Any | None:
    """Build ToolCallLimitMiddleware to cap tool calls per run."""
    try:
        from langchain.agents.middleware import ToolCallLimitMiddleware

        return ToolCallLimitMiddleware(run_limit=run_limit)
    except ImportError:
        logger.debug("ToolCallLimitMiddleware not available")
        return None


def _build_model_retry(config: Any) -> Any | None:
    """Build ModelRetryMiddleware for transient failure recovery."""
    try:
        from langchain.agents.middleware import ModelRetryMiddleware

        from deep_agent.src.guardrails import (
            ContentSafetyError,
            InputContentSafetyError,
            ToolContentSafetyError,
        )

        def _on_failure(exc: Exception) -> str:
            # Safety errors: return clean user-facing refusal instead of raw exception text.
            # retry_on already skips retrying these, so exc is the original exception.
            if isinstance(exc, ToolContentSafetyError):
                return "I wasn't able to complete this task due to a content safety policy issue."
            if isinstance(exc, (InputContentSafetyError, ContentSafetyError)):
                return "I can't help with that request due to content safety policy."
            # All other errors: preserve the standard format.
            exc_type = type(exc).__name__
            return f"Model call failed with {exc_type}: {exc}"

        return ModelRetryMiddleware(
            max_retries=config.max_retries,
            backoff_factor=config.backoff_factor,
            initial_delay=config.initial_delay,
            retry_on=lambda exc: not isinstance(exc, ContentSafetyError),
            on_failure=_on_failure,
        )
    except ImportError:
        logger.debug("ModelRetryMiddleware not available")
        return None


def _build_model_fallback(fallback_model: str) -> Any | None:
    """Build ModelFallbackMiddleware to switch models on primary failure."""
    try:
        from langchain.agents.middleware import ModelFallbackMiddleware

        return ModelFallbackMiddleware(fallback_model)
    except ImportError:
        logger.debug("ModelFallbackMiddleware not available")
        return None
    except Exception as e:
        logger.warning("ModelFallbackMiddleware init failed (check model auth): %s", e)
        return None


def _build_tool_retry(config: Any) -> Any | None:
    """Build ToolRetryMiddleware for specific tools."""
    try:
        from langchain.agents.middleware import ToolRetryMiddleware

        return ToolRetryMiddleware(
            max_retries=config.max_retries,
            tools=config.tools,
        )
    except ImportError:
        logger.debug("ToolRetryMiddleware not available")
        return None


def _build_custom_pii_middleware() -> Any | None:
    """Build the custom token-map PIIMiddleware from the global scrubber."""
    try:
        from deep_agent.src.pii.middleware import build_pii_middleware

        return build_pii_middleware()
    except ImportError:
        logger.debug("Custom PIIMiddleware not available")
        return None


def _build_pii_middleware(config: Any) -> list[Any]:
    """Route rules by provider.

    - provider: default → ParallelPIIMiddleware (stock langchain, one-way)
    - others            → handled by custom PIIMiddleware via global scrubber
    """
    try:
        from langchain.agents.middleware import AgentMiddleware, PIIMiddleware

        # Only default-provider rules go to the stock parallel middleware
        instances = []
        for rule in config.rules:
            if getattr(rule, "provider", "default") != "default":
                continue
            try:
                instances.append(
                    PIIMiddleware(
                        rule.name, strategy=rule.strategy, apply_to_input=True
                    )
                )
            except (ValueError, TypeError) as e:
                logger.warning("Skipping PII rule '%s': %s", rule.name, e)

        if not instances:
            return []

        class ParallelPIIMiddleware(AgentMiddleware):
            """Runs all stock PII type checks concurrently instead of as a sequential chain."""

            async def abefore_model(self, state: Any, runtime: Any) -> Any:
                import asyncio

                results = await asyncio.gather(
                    *(inst.abefore_model(state, runtime) for inst in instances),
                    return_exceptions=True,
                )
                for r in results:
                    if isinstance(r, BaseException):
                        raise r
                merged: dict = {}
                for r in results:
                    if isinstance(r, dict):
                        merged.update(r)
                return merged or None

            def before_model(self, state: Any, runtime: Any) -> Any:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor() as pool:
                    futures = [
                        pool.submit(inst.before_model, state, runtime)
                        for inst in instances
                    ]
                    results = [f.result() for f in futures]
                for r in results:
                    if isinstance(r, BaseException):
                        raise r
                merged: dict = {}
                for r in results:
                    if isinstance(r, dict):
                        merged.update(r)
                return merged or None

        return [ParallelPIIMiddleware()]

    except ImportError:
        logger.debug("PIIMiddleware not available")
        return []


def _build_summarization_tool_middleware(
    *,
    model: Any | None = None,
    backend: Any | None = None,
) -> Any | None:
    """Build SummarizationToolMiddleware instance.

    This gives the agent a tool to proactively trigger summarization
    at opportune moments (e.g., between tasks) rather than only at
    fixed token thresholds.
    """
    if model is None or backend is None:
        logger.warning("Summarization tool requires model and backend; skipping")
        return None
    try:
        from deepagents.middleware.summarization import (
            create_summarization_tool_middleware,
        )

        return create_summarization_tool_middleware(model, backend)
    except ImportError:
        logger.debug(
            "SummarizationToolMiddleware not available in this deepagents version"
        )
        return None
    except Exception as e:
        logger.warning("Failed to create SummarizationToolMiddleware: %s", e)
        return None


def _import_middleware(dotted_path: str) -> Any | None:
    """Import and instantiate a middleware from a dotted path.

    Format: "module.path:ClassName" or "module.path:factory_function"

    Args:
        dotted_path: Dotted import path with colon-separated attribute.

    Returns:
        Instantiated middleware, or None on failure.
    """
    try:
        if ":" not in dotted_path:
            logger.warning(
                "Invalid middleware path '%s' — expected 'module:Class'", dotted_path
            )
            return None

        module_path, attr_name = dotted_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        factory_or_class = getattr(module, attr_name)

        if callable(factory_or_class):
            return factory_or_class()
        return factory_or_class
    except Exception as e:
        logger.warning("Failed to import middleware '%s': %s", dotted_path, e)
        return None
