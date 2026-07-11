"""Human-in-the-loop interrupt configuration builder.

Converts the ``human_approval`` section of ``agent.yaml`` into the
``interrupt_on`` dict expected by ``create_deep_agent()``.

The dict maps each tool name to ``True`` (use deepagents default
decisions: approve / edit / reject / respond).  When the feature is
disabled the function returns an empty dict, which signals to
``graph.py`` not to pass ``interrupt_on`` at all.

For ``mode: all``, both the caller-supplied tools (MCP / explicit) and
the deepagents built-in tools are included so that every tool call —
regardless of origin — pauses for human approval.

Example YAML config::

    middleware:
      human_approval:
        enabled: true
        mode: all
        exclude:
          - ls
          - read_file
          - glob
          - grep
"""

from __future__ import annotations

from typing import Any

from deep_agent.src.agent.config.middleware import HumanApprovalConfig
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Built-in tool names added by deepagents internally (FilesystemMiddleware,
# TodoListMiddleware, SubAgentMiddleware).  These are never present in the
# caller-supplied ``tools`` list, so they must be enumerated explicitly for
# ``interrupt_on`` to cover them.
_DEEPAGENTS_BUILTIN_TOOLS: frozenset[str] = frozenset(
    {
        # filesystem (FilesystemMiddleware)
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "execute",
        # todo list (TodoListMiddleware)
        "write_todos",
        # subagents (SubAgentMiddleware)
        "task",
        # conversation management
        "compact_conversation",
    }
)


def build_interrupt_on(
    config: HumanApprovalConfig,
    tools: list[Any],
) -> dict[str, Any]:
    """Build the ``interrupt_on`` dict for ``create_deep_agent()``.

    Args:
        config: Resolved ``human_approval`` config from ``agent.yaml``.
        tools: List of resolved tool objects (must have a ``.name`` attr).
            Typically MCP tools + any explicitly declared tools.  Built-in
            deepagents tools are added automatically when ``mode`` is ``"all"``.

    Returns:
        Dict mapping tool name → ``True`` for every tool that should
        trigger a human approval interrupt.  Returns ``{}`` when the
        feature is disabled or ``mode`` is ``"none"``.
    """
    if not config.enabled or config.mode == "none":
        logger.debug("HITL disabled (enabled=%s, mode=%s)", config.enabled, config.mode)
        return {}

    exclude = set(config.exclude)

    # Explicit / MCP tools passed by the caller
    explicit_names = {t.name for t in tools}

    # For mode=all, also cover the deepagents built-in tools so that
    # filesystem and todo calls are intercepted even when no MCP tools exist.
    all_names = explicit_names | _DEEPAGENTS_BUILTIN_TOOLS

    interrupt_on = {name: True for name in all_names if name not in exclude}

    if interrupt_on:
        excluded = (explicit_names | _DEEPAGENTS_BUILTIN_TOOLS) - set(interrupt_on)
        logger.info(
            "HITL enabled: %d tool(s) will require approval%s",
            len(interrupt_on),
            f" ({len(excluded)} excluded: {sorted(excluded)})" if excluded else "",
        )
    else:
        logger.debug(
            "HITL enabled but all tools excluded (explicit=%d, builtins=%d, exclude=%s)",
            len(explicit_names),
            len(_DEEPAGENTS_BUILTIN_TOOLS),
            exclude,
        )

    return interrupt_on
