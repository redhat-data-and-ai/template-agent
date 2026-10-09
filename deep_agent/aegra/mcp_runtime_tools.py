"""Runtime attach of OAuth/DCR MCP tools after Connect, without rebuilding the graph.

Compile-time graphs bind a stable placeholder per oauth/dcr server. After the
user token is in Redis, this middleware lists the real tools and:

- ``awrap_model_call``: shows those tools to the model
- ``aafter_model``: Authenticate / Approve **before** any tool in the batch runs
- ``awrap_tool_call``: executes names that were not on the compiled ToolNode

HITL ``mode: all`` does not include runtime names in ``interrupt_on``. Those
calls are paused in ``aafter_model`` with the same HITL payload the UI already
understands, so an in-tools-node interrupt cannot replay sibling MCP calls.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command, interrupt

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_HITL_DECISIONS = ["approve", "edit", "reject", "respond"]


def _tool_call_name_and_id(tool_call: Any) -> tuple[str, str]:
    if isinstance(tool_call, dict):
        return str(tool_call.get("name") or ""), str(tool_call.get("id") or "")
    return str(getattr(tool_call, "name", "") or ""), str(
        getattr(tool_call, "id", "") or ""
    )


def _tool_call_args(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        args = tool_call.get("args") or {}
    else:
        args = getattr(tool_call, "args", {}) or {}
    return args if isinstance(args, dict) else {}


def _oauth_dcr_server_from_resource_call(
    tool_call: Any,
    scope: list[str] | frozenset[str] | None = None,
) -> str | None:
    """Return the oauth/dcr server in ``mcp_name`` for a host resource tool call."""
    from deep_agent.aegra.mcp import _fenced_oauth_dcr_servers
    from deep_agent.aegra.mcp_resource_tools import LIST_TOOL, READ_TOOL, TEMPLATES_TOOL

    name, _ = _tool_call_name_and_id(tool_call)
    if name not in {LIST_TOOL, TEMPLATES_TOOL, READ_TOOL}:
        return None
    mcp_name = _tool_call_args(tool_call).get("mcp_name")
    if not isinstance(mcp_name, str) or not mcp_name:
        return None
    if mcp_name in _fenced_oauth_dcr_servers(scope):
        return mcp_name
    return None


def _is_auth_continue(raw: Any) -> bool:
    if raw == "continue" or raw == {"type": "continue"}:
        return True
    return False


def _runtime_hitl_required(tool_name: str) -> bool:
    """Whether a dynamically attached MCP tool must pause for human approval."""
    try:
        from deep_agent.src.agent.config import agent_config

        orch = agent_config.get_orchestrator_config()
        resolved = agent_config.resolve_agent_middleware(orch.get("model") or "")
        hitl = resolved.human_approval
    except Exception:
        logger.warning(
            "Could not read HITL config for runtime MCP tool '%s' — pausing",
            tool_name,
            exc_info=True,
        )
        return True
    if not hitl.enabled or hitl.mode == "none":
        return False
    if tool_name in hitl.exclude:
        return False
    return hitl.mode == "all"


def _is_live_oauth_dcr_name(
    name: str,
    scope: list[str] | frozenset[str] | None = None,
) -> bool:
    from deep_agent.aegra.mcp import oauth_dcr_server_for_tool_name

    if not name or name.startswith("mcp__"):
        return False
    return oauth_dcr_server_for_tool_name(name, scope=scope) is not None


def _live_mcp_server(tool: Any) -> str | None:
    metadata = getattr(tool, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    server = metadata.get("mcp_server")
    return server if isinstance(server, str) and server else None


def _hitl_payload_for_calls(calls: list[Any]) -> dict[str, Any]:
    action_requests = []
    review_configs = []
    for call in calls:
        name, _ = _tool_call_name_and_id(call)
        action_requests.append(
            {
                "name": name,
                "args": _tool_call_args(call),
                "description": f"Approve MCP tool '{name}'",
            }
        )
        review_configs.append(
            {
                "action_name": name,
                "allowed_decisions": list(_HITL_DECISIONS),
            }
        )
    return {"action_requests": action_requests, "review_configs": review_configs}


def _interrupt_hitl(payload: dict[str, Any]) -> Any:
    """Pause for Approve. Auth-style ``continue`` is not an Approve."""
    raw = interrupt(payload)
    if _is_auth_continue(raw):
        raw = interrupt(payload)
    return raw


def _reject_decisions_for_payload(payload: Any) -> dict[str, Any]:
    """Fail-closed HITL payload when the resume is not ``{decisions: ...}``."""
    n = 1
    if isinstance(payload, dict):
        actions = payload.get("action_requests") or []
        if isinstance(actions, list) and actions:
            n = len(actions)
    missing = {"type": "reject", "message": "Missing HITL decision."}
    return {"decisions": [dict(missing) for _ in range(n)]}


def _interrupt_compiled_hitl(payload: Any) -> Any:
    """Compiled HITL: drain Connect ``continue``, reject an empty resume."""
    raw = _interrupt_hitl(payload if isinstance(payload, dict) else {})
    if isinstance(raw, dict) and isinstance(raw.get("decisions"), list):
        return raw
    return _reject_decisions_for_payload(payload)


_compiled_hitl_auth_resume_installed = False


def install_compiled_hitl_auth_resume() -> None:
    """Make compiled HITL ignore Authenticate's ``continue`` resume.

    LangChain HITL does ``interrupt(...)["decisions"]``. After Connect, that
    ``interrupt`` still returns ``"continue"``, which crashes. One retry
    pauses for a real Approve. An empty resume rejects instead of looping.
    """
    import langchain.agents.middleware.human_in_the_loop as hitl_mod

    hitl_mod.interrupt = _interrupt_compiled_hitl
    global _compiled_hitl_auth_resume_installed  # noqa: PLW0603
    _compiled_hitl_auth_resume_installed = True


def _decision_at(raw: Any, index: int) -> dict[str, Any]:
    missing = {"type": "reject", "message": "Missing HITL decision."}
    if not isinstance(raw, dict):
        return missing
    decisions = raw.get("decisions") or []
    if index >= len(decisions):
        return missing
    decision = decisions[index]
    if not isinstance(decision, dict) or not decision.get("type"):
        return missing
    return decision


def _apply_live_hitl_decisions(
    tool_calls: list[Any],
    live_indices: list[int],
    raw: Any,
) -> tuple[list[Any], list[ToolMessage]]:
    """Keep approved/edited live calls; rejected/responded calls become ToolMessages."""
    live_set = set(live_indices)
    revised: list[Any] = []
    messages: list[ToolMessage] = []
    decision_idx = 0
    for idx, call in enumerate(tool_calls):
        if idx not in live_set:
            revised.append(call)
            continue
        name, tool_call_id = _tool_call_name_and_id(call)
        decision = _decision_at(raw, decision_idx)
        decision_idx += 1
        kind = decision.get("type")
        if kind == "approve":
            revised.append(call)
            continue
        if kind == "reject":
            reason = str(decision.get("message") or "") or (
                "The user rejected this tool call."
            )
            messages.append(
                ToolMessage(
                    content=reason,
                    name=name,
                    tool_call_id=tool_call_id,
                    status="error",
                )
            )
            continue
        if kind == "respond":
            messages.append(
                ToolMessage(
                    content=str(decision.get("message") or ""),
                    name=name,
                    tool_call_id=tool_call_id,
                    status="success",
                )
            )
            continue
        if kind == "edit":
            edited = decision.get("edited_action") or {}
            new_call = (
                dict(call)
                if isinstance(call, dict)
                else {
                    "name": name,
                    "args": _tool_call_args(call),
                    "id": tool_call_id,
                }
            )
            if isinstance(edited, dict):
                if edited.get("name"):
                    new_call["name"] = edited["name"]
                if "args" in edited:
                    new_call["args"] = edited["args"]
            revised.append(new_call)
            continue
        messages.append(
            ToolMessage(
                content="Unknown HITL decision type.",
                name=name,
                tool_call_id=tool_call_id,
                status="error",
            )
        )
    return revised, messages


class McpRuntimeToolsMiddlewareSlot(AgentMiddleware):
    """No-op name slot so harness extra_middleware does not double-install."""

    name = "McpRuntimeToolsMiddleware"


class McpRuntimeToolsMiddleware(AgentMiddleware):
    """Attach authenticated OAuth/DCR MCP tools at call time (stable compiled graph)."""

    name = "McpRuntimeToolsMiddleware"

    def __init__(
        self,
        *,
        allowed_tool_names: Collection[str] | None = None,
        mcp_names: Collection[str] | None = None,
    ) -> None:
        """Store the yaml live-name allowlist and MCP server fence."""
        super().__init__()
        self._allowlist = (
            None if allowed_tool_names is None else frozenset(allowed_tool_names)
        )
        self._mcp_names = None if mcp_names is None else frozenset(mcp_names)

    def _allowed(self, name: str) -> bool:
        if self._allowlist is None:
            return True
        return name in self._allowlist

    def _placeholder_server_names(self, tools: list[Any]) -> list[str]:
        from deep_agent.aegra.mcp import _get_server_configs, placeholder_tool_name

        bound = {getattr(t, "name", "") for t in tools}
        names: list[str] = []
        for key, cfg in _get_server_configs().items():
            if not cfg.get("enabled", False):
                continue
            if cfg.get("auth_mode") not in ("oauth", "dcr"):
                continue
            if self._mcp_names is not None and key not in self._mcp_names:
                continue
            if placeholder_tool_name(key) in bound:
                names.append(key)
        return names

    async def _interrupt_missing_oauth_tokens(
        self, user_id: str, server_keys: list[str]
    ) -> None:
        from deep_agent.aegra.mcp import _get_server_configs, _resolve_connection_token
        from deep_agent.aegra.mcp_auth import (
            NeedsAuthorization,
            get_mcp_credential_resolver,
        )
        from deep_agent.aegra.mcp_tool_auth import _mcp_auth_interrupt_payload

        configs = _get_server_configs()
        seen: set[str] = set()
        ordered: list[str] = []
        for key in server_keys:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        while True:
            missing: str | None = None
            for key in ordered:
                entry = configs.get(key)
                if entry is None:
                    continue
                token = await _resolve_connection_token(key, entry, None, user_id)
                if not token:
                    missing = key
                    break
            if missing is None:
                return
            exc = NeedsAuthorization(
                missing,
                get_mcp_credential_resolver().connect_url(missing),
            )
            interrupt(_mcp_auth_interrupt_payload(exc))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Attach live OAuth/DCR tools to the model request when a token exists."""
        from deep_agent.aegra.mcp import (
            _current_user_id,
            _resolve_mcp_user_id,
            get_authenticated_oauth_mcp_tools,
            oauth_dcr_server_for_tool_name,
        )

        user_id = _resolve_mcp_user_id()
        if user_id:
            _current_user_id.set(user_id)
        if not user_id:
            return await handler(request)

        server_names = self._placeholder_server_names(list(request.tools or []))
        if not server_names:
            return await handler(request)

        try:
            live = await get_authenticated_oauth_mcp_tools(
                user_id, server_names=server_names
            )
        except Exception:
            logger.warning(
                "Authenticated MCP tool listing failed — continuing without runtime tools",
                exc_info=True,
            )
            return await handler(request)

        existing = {getattr(t, "name", "") for t in request.tools or []}
        bound = list(request.tools or [])
        scope = None if self._mcp_names is None else self._mcp_names
        extra = []
        for t in live:
            name = str(getattr(t, "name", "") or "")
            if not name or name in existing or not self._allowed(name):
                continue
            owner = oauth_dcr_server_for_tool_name(name, scope=scope)
            if owner is not None and _live_mcp_server(t) != owner:
                continue
            extra.append(t)
        live_servers = {
            server
            for tool in (*bound, *extra)
            if not str(getattr(tool, "name", "") or "").startswith("mcp__")
            and (server := _live_mcp_server(tool))
        }
        shown = []
        for tool in bound:
            name = str(getattr(tool, "name", "") or "")
            owner = oauth_dcr_server_for_tool_name(name, scope=scope)
            if name.startswith("mcp__") and owner in live_servers:
                continue
            shown.append(tool)
        if not extra and len(shown) == len(bound):
            return await handler(request)
        return await handler(request.override(tools=[*shown, *extra]))

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Pause for Connect and HITL before any live OAuth/DCR tool in the batch runs."""
        from deep_agent.aegra.mcp import (
            _current_user_id,
            _resolve_mcp_user_id,
            oauth_dcr_server_for_tool_name,
        )

        messages = state.get("messages") if isinstance(state, dict) else None
        if not messages:
            return None
        last_ai = next(
            (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None,
        )
        if last_ai is None or not last_ai.tool_calls:
            return None

        tool_calls = list(last_ai.tool_calls)
        server_keys: list[str] = []
        live_indices: list[int] = []
        scope = None if self._mcp_names is None else self._mcp_names
        for idx, call in enumerate(tool_calls):
            name, _ = _tool_call_name_and_id(call)
            key = oauth_dcr_server_for_tool_name(name, scope=scope)
            if not key:
                key = _oauth_dcr_server_from_resource_call(call, scope=scope)
            if not key:
                continue
            is_live = _is_live_oauth_dcr_name(name, scope=scope)
            if is_live and not self._allowed(name):
                continue
            server_keys.append(key)
            if is_live and _runtime_hitl_required(name):
                live_indices.append(idx)

        user_id = _resolve_mcp_user_id()
        if user_id:
            _current_user_id.set(user_id)
            if server_keys:
                await self._interrupt_missing_oauth_tokens(user_id, server_keys)

        if not live_indices:
            return None

        payload = _hitl_payload_for_calls([tool_calls[i] for i in live_indices])
        raw = _interrupt_hitl(payload)
        revised, artificial = _apply_live_hitl_decisions(tool_calls, live_indices, raw)
        last_ai.tool_calls = revised
        if not artificial:
            return {"messages": [last_ai]}
        return {"messages": [last_ai, *artificial]}

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Execute a live OAuth/DCR tool that was not bound on the compiled ToolNode."""
        from deep_agent.aegra.mcp import (
            _current_user_id,
            _resolve_mcp_user_id,
            get_authenticated_oauth_mcp_tools,
            oauth_dcr_server_for_tool_name,
        )

        name, tool_call_id = _tool_call_name_and_id(request.tool_call)
        if request.tool is not None:
            return await handler(request)
        if not self._allowed(name):
            return ToolMessage(
                content=f"Tool '{name}' is not allowed for this agent.",
                name=name,
                tool_call_id=tool_call_id,
                status="error",
            )

        scope = None if self._mcp_names is None else self._mcp_names
        owner = oauth_dcr_server_for_tool_name(name, scope=scope)
        if not owner:
            return ToolMessage(
                content=f"Tool '{name}' is not allowed for this agent.",
                name=name,
                tool_call_id=tool_call_id,
                status="error",
            )

        user_id = _resolve_mcp_user_id()
        if user_id:
            _current_user_id.set(user_id)
        if not user_id:
            return ToolMessage(
                content=f"Tool '{name}' is not connected.",
                name=name,
                tool_call_id=tool_call_id,
                status="error",
            )

        try:
            live = await get_authenticated_oauth_mcp_tools(
                user_id, server_names=[owner]
            )
        except Exception:
            logger.warning(
                "Authenticated MCP tool lookup failed for '%s'",
                name,
                exc_info=True,
            )
            return ToolMessage(
                content=f"Tool '{name}' could not be loaded.",
                name=name,
                tool_call_id=tool_call_id,
                status="error",
            )

        live_tool = next(
            (t for t in live if getattr(t, "name", "") == name),
            None,
        )
        if live_tool is None:
            return ToolMessage(
                content=f"Tool '{name}' is not connected.",
                name=name,
                tool_call_id=tool_call_id,
                status="error",
            )
        from deep_agent.src.guardrails.tool_proxy import wrap_tools

        return await handler(request.override(tool=wrap_tools([live_tool])[0]))


def runtime_mcp_attach_filters(
    declared_tools: Collection[str] | None = None,
    declared_mcps: Collection[str] | None = None,
) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    """Map yaml ``tools:`` / ``mcps:`` to runtime middleware filters.

    Non-empty ``tools:`` allowlists those live names. ``mcp__`` placeholders
    are Connect names, not live allowlist entries. Empty live ``tools:`` with
    ``mcps:`` attaches every live tool on the fence. Both empty attaches
    nothing.
    """
    tools = frozenset(
        n for n in (declared_tools or ()) if n and not str(n).startswith("mcp__")
    )
    mcps = frozenset(n for n in (declared_mcps or ()) if n)
    if not tools and not mcps:
        return frozenset(), frozenset()
    return (tools or None, mcps or None)


def build_mcp_runtime_tools_middleware(
    allowed_tool_names: Collection[str] | None = None,
    mcp_names: Collection[str] | None = None,
) -> McpRuntimeToolsMiddleware:
    """Factory used by the orchestrator, subagents, and harness extra_middleware."""
    return McpRuntimeToolsMiddleware(
        allowed_tool_names=allowed_tool_names,
        mcp_names=mcp_names,
    )


def build_mcp_runtime_tools_middleware_from_declared(
    declared_tools: Collection[str] | None = None,
    declared_mcps: Collection[str] | None = None,
) -> McpRuntimeToolsMiddleware:
    """Build runtime MCP middleware from an agent's yaml ``tools:`` / ``mcps:``."""
    allowed, names = runtime_mcp_attach_filters(declared_tools, declared_mcps)
    return build_mcp_runtime_tools_middleware(
        allowed_tool_names=allowed,
        mcp_names=names,
    )
