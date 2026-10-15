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

from fnmatch import fnmatch
from typing import Any

from deep_agent.src.agent.config.middleware import (
    ApprovalPolicy,
    HumanApprovalConfig,
)
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_MEMORY_PATH_PREFIX = "/memories/"

# Where skills are mounted in the agent's filesystem view, per the
# `filesystem.routes` entry in agent.yaml.
_SKILLS_PATH_PREFIX = "/skills/"

# Reading a skill is how the agent learns what a skill does. Prompting
# for it interrupts the agent mid-decision, before it even knows whether
# the skill is relevant, and trains people to approve without reading.
# Writing to a skill is a real change and still asks.
_READ_ONLY_FILE_TOOLS: frozenset[str] = frozenset({"read_file", "ls", "glob", "grep"})

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


def _tool_call_path(req: Any) -> str:
    """The path a filesystem tool call targets, or "" if it has none."""
    args = req.tool_call.get("args", {}) or {}
    return args.get("file_path") or args.get("path") or ""


def _is_non_memory_path(req: Any) -> bool:
    """Return True (interrupt) unless the file path is under /memories/."""
    return not _tool_call_path(req).startswith(_MEMORY_PATH_PREFIX)


def _is_non_exempt_path(req: Any) -> bool:
    """Return True (interrupt) unless the path is memory or a skill.

    Applied to read-only filesystem tools. Skill reads are exempt because
    the agent must read `SKILL.md` to decide whether a skill applies;
    asking first is asking about a decision that has not been made yet.
    """
    path = _tool_call_path(req)
    if path.startswith(_MEMORY_PATH_PREFIX):
        return False
    return not path.startswith(_SKILLS_PATH_PREFIX)


def _is_non_skill_read(req: Any) -> bool:
    """Return True (interrupt) unless the path is a skill."""
    return not _tool_call_path(req).startswith(_SKILLS_PATH_PREFIX)


def resolve_policy(
    tool_name: str,
    config: HumanApprovalConfig,
    *,
    effects: str = "live",
) -> ApprovalPolicy:
    """Return the policy governing one tool. First match wins, in order.

    ``exclude`` predates policies and still means "never prompt".

    Under ``effects="dry_run"`` anything that would pause for a human is
    refused instead: the agent can still read and plan, but cannot act.
    """
    policy = _resolve_policy_live(tool_name, config)

    if effects == "dry_run" and policy.decision == "ask":
        return ApprovalPolicy(
            tools=policy.tools,
            decision="deny",
            risk=policy.risk,
            reason=f"dry run — would have asked ({policy.reason or 'no reason given'})",
        )
    return policy


def _resolve_policy_live(tool_name: str, config: HumanApprovalConfig) -> ApprovalPolicy:
    """Policy resolution ignoring the effects axis."""
    for pattern in config.exclude:
        if fnmatch(tool_name, pattern):
            return ApprovalPolicy(
                tools=[pattern],
                decision="allow",
                risk="low",
                reason="Excluded from approval in config",
            )

    for policy in config.policies:
        if any(fnmatch(tool_name, pattern) for pattern in policy.tools):
            return policy

    return ApprovalPolicy(
        tools=[tool_name],
        decision=config.default_decision,
        reason="No matching policy; using default_decision",
    )


def _approval_description(tool_name: str, policy: ApprovalPolicy) -> str:
    """One line a human can act on, prefixed with severity.

    Machine-readable enough to parse (`[HIGH RISK]`) and readable enough
    to be useful if nothing parses it.
    """
    severity = {"high": "[HIGH RISK] ", "medium": "", "low": "[low risk] "}[policy.risk]
    # The default-policy text is internal bookkeeping, not something to
    # show someone deciding whether to approve.
    if policy.reason.startswith("No matching policy"):
        reason = "No approval policy covers this tool; approval required by default."
    else:
        reason = policy.reason or "No rationale configured for this tool."
    return f"{severity}{tool_name}: {reason}"


def describe_policies(
    tool_names: set[str],
    config: HumanApprovalConfig,
    *,
    effects: str = "live",
) -> dict[str, dict[str, Any]]:
    """Resolved decision, risk and rationale per tool.

    Lets an operator see the effect of the glob patterns rather than
    applying them mentally.
    """
    return {
        name: {
            "decision": (p := resolve_policy(name, config, effects=effects)).decision,
            "risk": p.risk,
            "reason": p.reason,
        }
        for name in sorted(tool_names)
    }


def denied_tool_names(
    config: HumanApprovalConfig, tools: list[Any], *, effects: str = "live"
) -> frozenset[str]:
    """Tools the policy refuses outright.

    Offering a tool that can only be refused wastes a turn and reads to
    the user as the agent failing.
    """
    if not config.enabled:
        return frozenset()

    names = {t.name for t in tools} | _DEEPAGENTS_BUILTIN_TOOLS
    return frozenset(
        name
        for name in names
        if resolve_policy(name, config, effects=effects).decision == "deny"
    )


def build_interrupt_on(
    config: HumanApprovalConfig,
    tools: list[Any],
    *,
    effects: str = "live",
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
    denied: list[str] = []
    for name in all_names:
        decision = resolve_policy(name, config, effects=effects).decision
        if decision == "allow":
            continue
        if decision == "deny":
            denied.append(name)
            continue
        # Read-only tools that can also touch memory need both exemptions;
        # the rest need whichever single one applies.
        if name in _MEMORY_FILE_TOOLS and name in _READ_ONLY_FILE_TOOLS:
            predicate = _is_non_exempt_path
        elif name in _MEMORY_FILE_TOOLS:
            predicate = _is_non_memory_path
        elif name in _READ_ONLY_FILE_TOOLS:
            predicate = _is_non_skill_read
        else:
            predicate = None

        policy = resolve_policy(name, config, effects=effects)
        entry: dict[str, Any] = {
            "allowed_decisions": ["approve", "edit", "reject", "respond"],
            # Carries risk and rationale to whatever renders the prompt.
            # Without it every approval looks equally urgent and people
            # learn to approve without reading.
            "description": _approval_description(name, policy),
        }
        if predicate is not None:
            entry["when"] = predicate
        interrupt_on[name] = entry

    if denied:
        logger.warning("Tools denied by approval policy: %s", sorted(denied))

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
