"""Async subagent middleware builder.

Auto-detects async subagents (type: async in frontmatter) from the loaded
subagent list and builds AsyncSubAgentMiddleware to wire background task
tools into the agent.

Template-agent users configure async subagents via Markdown frontmatter:
    type: async
    graph_id: my_graph
    url: https://my-deployment.example.com   # optional

This module handles the middleware wiring. Users never call it directly.
"""

from __future__ import annotations

from typing import Any

from deep_agent.src.agent.config.providers import AsyncTaskConfig
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def build_async_middleware(
    subagents: list[Any] | None,
    async_config: AsyncTaskConfig,
) -> Any | None:
    """Build AsyncSubAgentMiddleware if async subagents exist.

    Scans the loaded subagent list for AsyncSubAgent instances and
    wraps them in AsyncSubAgentMiddleware, which adds tools for
    launching, monitoring, and updating background tasks.

    Args:
        subagents: List of loaded subagent instances (SubAgent,
            CompiledSubAgent, AsyncSubAgent). Can be None.
        async_config: Async task settings from providers.yaml.

    Returns:
        AsyncSubAgentMiddleware instance, or None if no async subagents
        exist or the feature is disabled.
    """
    if not async_config.enabled:
        logger.debug("Async tasks disabled via config")
        return None

    if not subagents:
        return None

    async_subagents = _extract_async_subagents(subagents)
    if not async_subagents:
        return None

    try:
        from deepagents.middleware.async_subagents import AsyncSubAgentMiddleware

        kwargs: dict[str, Any] = {"async_subagents": async_subagents}
        if async_config.system_prompt is not None:
            kwargs["system_prompt"] = async_config.system_prompt

        middleware = AsyncSubAgentMiddleware(**kwargs)
        logger.info(
            "Built AsyncSubAgentMiddleware with %d async subagent(s)",
            len(async_subagents),
        )
        return middleware
    except ImportError:
        logger.warning("AsyncSubAgentMiddleware not available — async tasks disabled")
        return None
    except Exception as e:
        logger.warning("Failed to build AsyncSubAgentMiddleware: %s", e)
        return None


def _extract_async_subagents(subagents: list[Any]) -> list[Any]:
    """Filter the subagent list for AsyncSubAgent instances."""
    try:
        from deepagents.middleware.async_subagents import AsyncSubAgent

        return [s for s in subagents if isinstance(s, AsyncSubAgent)]
    except ImportError:
        return []
