"""Tool access control for subagents.

Provides filtering, denial, and approval wrapping for subagent tool sets.
Used by subagent builders to enforce per-subagent tool access policies
declared in frontmatter config.
"""

from __future__ import annotations

import inspect
from typing import Any

from langgraph.types import interrupt

from deep_agent.src.exceptions import AppException, ErrorCodes
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


def filter_denied_tools(
    tools: list[Any],
    denied_names: list[str],
    agent_name: str,
) -> list[Any]:
    """Remove denied tools from a resolved tool list.

    Args:
        tools: List of resolved tool objects (must have a ``.name`` attr).
        denied_names: Tool names to exclude.
        agent_name: Subagent name (for logging).

    Returns:
        New list with denied tools removed. Original order preserved.
    """
    if not denied_names:
        return tools

    denied_set = set(denied_names)
    result: list[Any] = []

    for tool in tools:
        if tool.name in denied_set:
            logger.info("DENIED tool removed", agent=agent_name, tool=tool.name)
        else:
            result.append(tool)

    return result


def apply_tool_approval(
    tools: list[Any],
    approval_names: list[str],
    agent_name: str,
) -> list[Any]:
    """Wrap tools requiring human approval with interrupt guards.

    Tools whose names appear in *approval_names* are wrapped so that
    invoking them triggers a LangGraph ``interrupt()`` before the
    underlying function executes.

    Args:
        tools: List of resolved tool objects (must have a ``.name`` attr).
        approval_names: Tool names that require approval.
        agent_name: Subagent name (for logging and interrupt payload).

    Returns:
        New list with matching tools wrapped. Non-matching tools
        are passed through unchanged.
    """
    if not approval_names:
        return tools

    approval_set = set(approval_names)
    tool_names_present = {t.name for t in tools}

    # Warn about approval names that reference tools not in the resolved set.
    for name in approval_names:
        if name not in tool_names_present:
            logger.warning(
                "APPROVAL references unknown tool", agent=agent_name, tool=name
            )

    result: list[Any] = []
    for tool in tools:
        if tool.name in approval_set:
            result.append(_wrap_tool_with_approval(tool, agent_name))
        else:
            result.append(tool)

    return result


def _build_hitl_payload(tool: Any, agent_name: str) -> dict[str, Any]:
    """Build an HITL-compatible interrupt payload for tool approval.

    Returns a dict matching the ``HITLInterruptValue`` schema expected by the
    frontend: ``{ action_requests: [...], review_configs: [...] }``.
    """
    description = (
        f"Tool approval required: subagent '{agent_name}' wants to call "
        f"'{tool.name}'. Approve or reject this tool call."
    )
    return {
        "action_requests": [
            {
                "name": tool.name,
                "args": {"agent": agent_name, "description": description},
            }
        ],
        "review_configs": [
            {
                "action_name": tool.name,
                "allowed_decisions": ["approve", "reject"],
            }
        ],
    }


def _is_approved(decision: Any) -> bool:
    """Check whether the HITL resume value signals approval.

    Handles both the structured format (list of dicts with ``type`` key)
    sent by the frontend and simple string values used in tests.
    """
    if isinstance(decision, list):
        return any(isinstance(d, dict) and d.get("type") == "approve" for d in decision)
    return str(decision).strip().lower() == "approved"


def _wrap_tool_with_approval(tool: Any, agent_name: str) -> Any:
    """Wrap a single tool so it calls ``interrupt()`` before execution.

    Follows the same wrapping pattern used in
    ``deep_agent.aegra.mcp_tool_auth._wrap_single_tool``: replaces the
    coroutine (async) or func (sync) with a wrapper that issues an
    interrupt, then proceeds only if the human approves.

    The interrupt payload uses the HITL dict format so the frontend's
    ``InterruptBanner`` can render it natively.

    Args:
        tool: A LangChain ``StructuredTool`` (or compatible) tool object.
        agent_name: Subagent name (for the interrupt payload message).

    Returns:
        A copy of the tool with its callable replaced by the approval wrapper.
    """
    payload = _build_hitl_payload(tool, agent_name)

    coroutine = getattr(tool, "coroutine", None)
    func = getattr(tool, "func", None)

    if inspect.iscoroutinefunction(coroutine):
        original_coroutine = coroutine

        async def wrapped_coroutine(**kwargs: Any) -> Any:
            decision = interrupt(payload)
            if _is_approved(decision):
                return await original_coroutine(**kwargs)
            return f"Tool '{tool.name}' was rejected by the user."

        try:
            return tool.model_copy(update={"coroutine": wrapped_coroutine})
        except Exception:
            tool.coroutine = wrapped_coroutine
            return tool

    if func is not None and inspect.isfunction(func):
        original_func = func

        def wrapped_func(**kwargs: Any) -> Any:
            decision = interrupt(payload)
            if _is_approved(decision):
                return original_func(**kwargs)
            return f"Tool '{tool.name}' was rejected by the user."

        try:
            return tool.model_copy(update={"func": wrapped_func})
        except Exception:
            tool.func = wrapped_func
            return tool

    # Tool has neither coroutine nor func — return as-is.
    return tool


def migrate_tools_field(config: dict[str, Any], agent_name: str) -> dict[str, Any]:
    """Migrate deprecated ``tools`` key to ``allowed_tools``.

    If the config contains a ``tools`` key but no ``allowed_tools``, the
    value is moved to ``allowed_tools`` and a deprecation warning is logged.

    Args:
        config: Parsed frontmatter config dict (mutated in place).
        agent_name: Subagent name (for logging).

    Returns:
        The same *config* dict (for chaining convenience).

    Raises:
        AppException: If both ``tools`` and ``allowed_tools`` are present.
    """
    has_tools = "tools" in config
    has_allowed = "allowed_tools" in config

    if has_tools and has_allowed:
        raise AppException(
            f"Subagent '{agent_name}': config contains both 'tools' and "
            f"'allowed_tools'. Remove the deprecated 'tools' field.",
            error_code=ErrorCodes.CONFIGURATION_VALIDATION_ERROR,
        )

    if has_tools and not has_allowed:
        config["allowed_tools"] = config.pop("tools")
        logger.info("COMPAT migrated tools->allowed_tools", agent=agent_name)

    return config
