"""Human-in-the-loop interrupt configuration builder.

Converts the ``human_approval`` section of ``agent.yaml`` into the
``interrupt_on`` dict expected by ``create_deep_agent()``.

The dict maps each tool name to ``True`` or an ``InterruptOnConfig``
(which may include a ``when`` predicate).  When the feature is
disabled the function returns an empty dict, which signals to
``graph.py`` not to pass ``interrupt_on`` at all.

For ``mode: all``, both the caller-supplied tools (MCP / explicit) and
the deepagents built-in tools are included so that every tool call —
regardless of origin — pauses for human approval.

File tools (``read_file``, ``write_file``, ``edit_file``) are
auto-approved only when the target path is under ``/memories/``.
All other paths still require human approval.

Example YAML config::

    middleware:
      human_approval:
        enabled: true
        mode: all
        exclude:
          - ls
          - glob
          - grep
"""

from __future__ import annotations

from typing import Any

from deep_agent.src.agent.config.middleware import HumanApprovalConfig
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_MEMORY_PATH_PREFIX = "/memories/"

_MEMORY_FILE_TOOLS: frozenset[str] = frozenset({"read_file", "write_file", "edit_file"})

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


def _is_non_memory_path(req: Any) -> bool:
    """Return True (interrupt) unless the file path is under /memories/."""
    args = req.tool_call.get("args", {})
    path = args.get("file_path") or args.get("path") or ""
    return not path.startswith(_MEMORY_PATH_PREFIX)


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
        Dict mapping tool name → ``True`` or ``InterruptOnConfig`` for
        every tool that should trigger a human approval interrupt.
        Returns ``{}`` when the feature is disabled or ``mode`` is
        ``"none"``.
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

    interrupt_on: dict[str, Any] = {}
    for name in all_names:
        if name in exclude:
            continue
        if name in _MEMORY_FILE_TOOLS:
            interrupt_on[name] = {
                "allowed_decisions": ["approve", "edit", "reject", "respond"],
                "when": _is_non_memory_path,
            }
        else:
            interrupt_on[name] = True

    if interrupt_on:
        excluded = (explicit_names | _DEEPAGENTS_BUILTIN_TOOLS) - set(interrupt_on)
        memory_auto = sorted(_MEMORY_FILE_TOOLS - exclude)
        logger.info(
            "HITL enabled: %d tool(s) will require approval%s%s",
            len(interrupt_on),
            f" ({len(excluded)} excluded: {sorted(excluded)})" if excluded else "",
            f" (memory-only auto-approve: {memory_auto})" if memory_auto else "",
        )
    else:
        logger.debug(
            "HITL enabled but all tools excluded (explicit=%d, builtins=%d, exclude=%s)",
            len(explicit_names),
            len(_DEEPAGENTS_BUILTIN_TOOLS),
            exclude,
        )

    return interrupt_on
