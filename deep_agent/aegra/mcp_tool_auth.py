"""Wrap MCP tools to raise LangGraph interrupts when OAuth is required."""

from __future__ import annotations

import contextvars
import inspect
import json
from typing import Any

from langgraph.types import interrupt

from deep_agent.aegra.mcp_auth import NeedsAuthorization
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _in_nested_graph() -> bool:
    """True when running inside a subagent rather than the main graph.

    LangGraph namespaces nested graphs: ``checkpoint_ns`` is empty at
    the top level and non-empty inside one. That distinction decides
    whether an authorization request can be answered at all.

    Unknown counts as "top level". Interrupting when we could have
    reported is recoverable; reporting when we should have paused
    loses a resumable run.
    """
    try:
        from langgraph.config import get_config

        configurable = (get_config() or {}).get("configurable") or {}
        return bool(configurable.get("checkpoint_ns"))
    except Exception:
        return False


# Authorization requests raised while answering the current turn.
#
# The marker in the tool result is for the model to read; it is not a
# reliable channel to the user, because the model paraphrases. Asked
# to list Jira issues with Atlassian disconnected, it wrote its own
# "please connect Jira in Settings" and dropped the marker — so the
# UI had nothing to render a button from.
#
# Recorded structurally instead, and attached to the message in
# `provenance.describe_message`, which is the same path the rest of
# the per-message metadata already takes.
_auth_requests: contextvars.ContextVar[list[dict[str, str]] | None] = (
    contextvars.ContextVar("_mcp_auth_requests", default=None)
)


def record_auth_request(exc: NeedsAuthorization) -> None:
    """Note that this turn could not reach an MCP server.

    Appends to the list; never calls ``set``. Context propagates
    *into* a nested task and not back out, so a subagent that
    allocated its own list would record into something the parent
    never sees — which is exactly what happened: the request was
    logged, the turn answered, and provenance came back empty.
    Mutating one list allocated per turn is visible from both.
    """
    pending = _auth_requests.get()
    if pending is None:
        # No turn in progress (startup probe, eval, direct call).
        # Nothing will read it, so there is nothing to record.
        return
    entry = {
        "mcp_name": exc.mcp_name,
        "reason": getattr(exc, "reason", "not_connected"),
        "connect_url": exc.connect_url,
    }
    # One server, one prompt — several tools can fail on the same one.
    if entry not in pending:
        pending.append(entry)


def pending_auth_requests() -> list[dict[str, str]]:
    """Authorization requests recorded during this turn."""
    return list(_auth_requests.get() or [])


def clear_auth_requests() -> None:
    """Start a turn with a fresh list.

    Allocated here, once, so every nested context — subagents
    included — appends to the same object.
    """
    _auth_requests.set([])


def _auth_reason_phrase(exc: NeedsAuthorization) -> str:
    """Wording that matches which of the two situations this is."""
    if getattr(exc, "reason", "not_connected") == "expired":
        return "your session for it has expired and could not be renewed"
    return "it is not connected for this user"


def _auth_prompt_key(mcp_name: str) -> str | None:
    """Redis key marking that this user has already been asked."""
    from deep_agent.aegra.mcp import _current_user_id

    user_id = _current_user_id.get()
    if not user_id:
        return None
    return f"mcp_auth_asked:{settings.agent_deployment_id}:{user_id}:{mcp_name}"


async def forget_auth_prompt(user_id: str, mcp_name: str) -> None:
    """Drop the cooldown once the integration is connected.

    Otherwise connecting, then later having the grant revoked, would
    be met with silence for the rest of the window — the one case
    where interrupting is exactly what the user wants.
    """
    if settings.MCP_AUTH_PROMPT_COOLDOWN_SECONDS <= 0 or not user_id:
        return
    key = f"mcp_auth_asked:{settings.agent_deployment_id}:{user_id}:{mcp_name}"
    try:
        from deep_agent.aegra.redis import cache_delete, run_redis

        await run_redis(cache_delete, key)
    except Exception:
        logger.debug("Could not clear auth prompt state", exc_info=True)


async def _already_asked(mcp_name: str) -> bool:
    """True when an unanswered prompt for this integration still stands.

    Interrupting is the right way to ask the first time: the run
    pauses, the user connects, and it carries on from where it was.
    But someone who simply does not want that integration gets asked
    again in the next chat, and the next — each time blocking a turn
    to do it. Within the cooldown the agent falls back to saying the
    integration is unavailable and offering the Connect button,
    which informs without stopping anything.
    """
    if settings.MCP_AUTH_PROMPT_COOLDOWN_SECONDS <= 0:
        return False
    key = _auth_prompt_key(mcp_name)
    if not key:
        return False
    try:
        from deep_agent.aegra.redis import cache_get, run_redis

        return bool(await run_redis(cache_get, key))
    except Exception:
        # Unknown means ask: a prompt the user can act on beats
        # silence because a cache was unreachable.
        logger.debug("Could not read auth prompt state", exc_info=True)
        return False


async def _remember_asked(mcp_name: str) -> None:
    """Record that this user has been asked, so the next turn does not."""
    ttl = settings.MCP_AUTH_PROMPT_COOLDOWN_SECONDS
    if ttl <= 0:
        return
    key = _auth_prompt_key(mcp_name)
    if not key:
        return
    try:
        from deep_agent.aegra.redis import cache_set, run_redis

        await run_redis(cache_set, key, "1", ttl_seconds=ttl)
    except Exception:
        logger.debug("Could not record auth prompt state", exc_info=True)


def _mcp_auth_message(exc: NeedsAuthorization) -> str:
    """What a subagent reports upward when it cannot reach a server.

    A subagent is not a conversation. ``interrupt()`` there pauses a
    graph nobody is watching: the pause bubbles out of the `task`
    tool as an error, the parent run ends `interrupted`, and
    ``/runs/wait`` answers 200 with an empty body. The user sees the
    agent stop for no stated reason — which is exactly what "it
    hangs on subagent calls" turned out to be.

    Returned as a tool result instead, the orchestrator reads it and
    tells the user which integration to connect.
    """
    # The marker is machine-readable so the UI can offer a connect
    # action in place of the prose, and the prose is still useful if
    # nothing parses it.
    #
    # No URL anywhere in this string. It is this deployment's own
    # address — `http://localhost:8081/...` in a Kind cluster — and
    # anything placed in a tool result is something the model can
    # copy into its answer, where it is wrong for every reader not
    # on the machine that rendered it. It did exactly that. The UI
    # builds the connect button from the server name alone, so
    # neither the marker nor the prose needs an address.
    return (
        f"[AUTH_REQUIRED mcp={exc.mcp_name} reason="
        f"{getattr(exc, 'reason', 'not_connected')}] "
        f"Cannot use {exc.mcp_name}: {_auth_reason_phrase(exc)}. "
        f"Tell the user to connect {exc.mcp_name} using the Connect "
        f"button shown with this message, then ask again. Never write "
        f"a URL for this — do not repeat any link from this message. "
        f"Do not retry this tool."
    )


def _mcp_auth_interrupt_payload(exc: NeedsAuthorization) -> str:
    return json.dumps(
        {
            "type": "mcp_auth_required",
            "mcp_name": exc.mcp_name,
            "connect_url": exc.connect_url,
            "reason": getattr(exc, "reason", "not_connected"),
            "message": f"Connect to {exc.mcp_name} to use these tools",
        }
    )


def _fix_stringified_json_args(tool: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Parse stringified JSON args when the tool schema expects object/array.

    Some models (notably Gemini) serialize nested objects as JSON strings
    instead of proper dicts when calling tools with complex input schemas.
    Parses args whose schema type is ``object``, ``array``, or untyped
    (``Any`` — no ``type`` key in the schema property). Args explicitly
    typed as ``string`` are never modified.
    """
    schema_props: dict[str, Any] = {}
    try:
        schema_props = getattr(tool, "args", {}) or {}
    except Exception:
        logger.debug(
            "Failed to read args schema for tool '%s' — skipping JSON arg fix",
            getattr(tool, "name", "?"),
            exc_info=True,
        )
        return kwargs

    if not schema_props:
        return kwargs

    fixed = dict(kwargs)
    for key, value in fixed.items():
        if not isinstance(value, str):
            continue
        prop = schema_props.get(key, {})
        expected_type = prop.get("type", "")
        if expected_type == "string" or (
            isinstance(expected_type, list) and "string" in expected_type
        ):
            continue
        union_schemas = prop.get("anyOf", []) + prop.get("oneOf", [])
        if any("string" in str(s.get("type", "")) for s in union_schemas):
            continue
        stripped = value.strip()
        if stripped and stripped[0] in ("{", "["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, (dict, list)):
                    fixed[key] = parsed
                    logger.debug(
                        "Fixed stringified JSON arg '%s' for tool '%s'",
                        key,
                        getattr(tool, "name", "?"),
                    )
            except (json.JSONDecodeError, ValueError, RecursionError):
                pass
    return fixed


def wrap_mcp_tools_for_auth(tools: list[Any]) -> list[Any]:
    """Wrap MCP tools so ``NeedsAuthorization`` becomes a resumable interrupt."""
    wrapped: list[Any] = []
    for tool in tools:
        wrapped.append(_wrap_single_tool(tool))
    return wrapped


def _make_safe_ainvoke(target_tool: Any) -> Any:
    """Build an ainvoke wrapper that catches MCP errors for *target_tool*."""
    original_ainvoke = target_tool.ainvoke

    async def safe_ainvoke(tool_input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Wrap ainvoke to catch auth interrupts and MCP errors."""
        from langchain_core.messages import ToolMessage
        from langgraph.errors import GraphBubbleUp

        # Same shape as `wrapped_coroutine` below, and for the same
        # reason: the cooldown belongs after a declined resume, not
        # before the pause. Recorded first, the very first ask
        # disqualified every ask after it, so func-based and
        # unwrapped tools never paused at all.
        paused = False
        while True:
            try:
                return await original_ainvoke(tool_input, config, **kwargs)
            except NeedsAuthorization as exc:
                nested_without_interrupt = (
                    _in_nested_graph() and not settings.MCP_AUTH_INTERRUPT_IN_SUBAGENTS
                )
                # Asked already and not answered: inform, do not block.
                asked_before = await _already_asked(exc.mcp_name)

                if nested_without_interrupt or asked_before or paused:
                    if paused:
                        # Came back from the Connect button still
                        # unconnected: they declined.
                        await _remember_asked(exc.mcp_name)
                    logger.info(
                        "MCP auth required for '%s' — reporting upward (%s)",
                        exc.mcp_name,
                        "declined"
                        if paused
                        else "already asked this user"
                        if asked_before
                        else "inside a subagent",
                    )
                    record_auth_request(exc)
                    return ToolMessage(
                        content=_mcp_auth_message(exc),
                        name=getattr(target_tool, "name", "unknown"),
                        tool_call_id=str(tool_input.get("id", ""))
                        if isinstance(tool_input, dict)
                        else "",
                        status="error",
                    )
                logger.info(
                    "MCP auth required for '%s' — interrupting run",
                    exc.mcp_name,
                )
                record_auth_request(exc)
                paused = True
                interrupt(_mcp_auth_interrupt_payload(exc))
            except GraphBubbleUp:
                raise
            except Exception as exc:
                tool_name = getattr(target_tool, "name", "unknown")
                tool_call_id = ""
                if isinstance(tool_input, dict):
                    tool_call_id = str(tool_input.get("id", ""))
                logger.warning("MCP tool '%s' failed: %s", tool_name, exc)
                return ToolMessage(
                    content=f"[TOOL_ERROR] {tool_name} failed: {exc}",
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    status="error",
                )

    return safe_ainvoke


def _wrap_single_tool(tool: Any) -> Any:
    coroutine = getattr(tool, "coroutine", None)
    func = getattr(tool, "func", None)

    if inspect.iscoroutinefunction(coroutine):

        async def wrapped_coroutine(**kwargs: Any) -> Any:
            # True once this call has paused and come back. On resume
            # LangGraph replays the node, so the flag starts false
            # again — but `interrupt()` returns instead of raising the
            # second time, which drops us back into the loop with it
            # set. That is what tells a decline apart from a first ask.
            paused = False
            while True:
                try:
                    kwargs = _fix_stringified_json_args(tool, kwargs)
                    return await coroutine(**kwargs)
                except NeedsAuthorization as exc:
                    nested_without_interrupt = (
                        _in_nested_graph()
                        and not settings.MCP_AUTH_INTERRUPT_IN_SUBAGENTS
                    )
                    asked_before = await _already_asked(exc.mcp_name)
                    if nested_without_interrupt or asked_before or paused:
                        if paused:
                            # Offered the Connect button, came back
                            # still unconnected: they declined. Only
                            # now is it right to stop asking.
                            #
                            # This used to be recorded before the
                            # pause, which meant the very first ask
                            # disqualified every ask after it — the
                            # cooldown suppressed the interrupt it was
                            # supposed to follow, and the run reported
                            # "not connected" in prose for ever after.
                            await _remember_asked(exc.mcp_name)
                        logger.info(
                            "[%s] MCP auth required (tool=%s) — reporting upward (%s)",
                            exc.mcp_name,
                            getattr(tool, "name", "?"),
                            "declined"
                            if paused
                            else "already asked this user"
                            if asked_before
                            else "inside a subagent",
                        )
                        record_auth_request(exc)
                        return _mcp_auth_message(exc)
                    logger.warning(
                        "[%s] MCP auth required — interrupting run (tool=%s)",
                        exc.mcp_name,
                        getattr(tool, "name", "?"),
                    )
                    record_auth_request(exc)
                    paused = True
                    interrupt(_mcp_auth_interrupt_payload(exc))

        try:
            wrapped = tool.model_copy(update={"coroutine": wrapped_coroutine})
        except Exception:
            logger.info(
                "model_copy failed for tool '%s' — patching coroutine directly",
                getattr(tool, "name", "?"),
            )
            tool.coroutine = wrapped_coroutine
            wrapped = tool
        object.__setattr__(wrapped, "ainvoke", _make_safe_ainvoke(wrapped))
        return wrapped

    if func is not None and inspect.isfunction(func):

        def wrapped_func(**kwargs: Any) -> Any:
            while True:
                try:
                    kwargs = _fix_stringified_json_args(tool, kwargs)
                    return func(**kwargs)
                except NeedsAuthorization as exc:
                    logger.warning(
                        "[%s] MCP auth required — interrupting run (tool=%s)",
                        exc.mcp_name,
                        getattr(tool, "name", "?"),
                    )
                    record_auth_request(exc)
                    interrupt(_mcp_auth_interrupt_payload(exc))

        try:
            wrapped = tool.model_copy(update={"func": wrapped_func})
        except Exception:
            logger.info(
                "model_copy failed for tool '%s' — patching func directly",
                getattr(tool, "name", "?"),
            )
            tool.func = wrapped_func
            wrapped = tool
        object.__setattr__(wrapped, "ainvoke", _make_safe_ainvoke(wrapped))
        return wrapped

    object.__setattr__(tool, "ainvoke", _make_safe_ainvoke(tool))
    return tool


def narrow_auth_requests(
    requests: list[dict[str, Any]], text: str
) -> list[dict[str, Any]]:
    """Keep only the integrations this answer is actually about.

    Discovery records every unconnected server, because every one of
    them contributed a placeholder tool. Stamped verbatim that is
    five Connect buttons under a reply about one of them, and working
    out which one is theirs is most of the work the button exists to
    remove.

    Strictly: an integration the answer does not mention gets no
    button. The first attempt kept the whole list when nothing
    matched, on the theory that one button too many beats none —
    which is precisely backwards. Connect the named server and ask
    again, and it drops out of the list, nothing matches, and the
    reader is offered Atlan, Atlassian, Datadog, GitLab, Gworkspace,
    Miro and unstructured-data: seven buttons, none of them the one
    the sentence above is about.

    An unattributed failure showing no button is the better error.
    The tool-failure text names the server it could not use, so the
    answer relaying it almost always names it too.
    """
    if not requests or not isinstance(text, str) or not text:
        return []
    haystack = text.lower()
    return [
        r
        for r in requests
        if isinstance(r, dict) and str(r.get("mcp_name", "")).lower() in haystack
    ]


def build_mcp_auth_notice_middleware() -> Any | None:
    """Stamp this turn's unreachable integrations onto the reply.

    The recording happens deep inside discovery and tool calls, in
    whatever context those run in. This copies it onto the message
    while that context is still live, because the message is what
    survives into the response the UI reads.
    """
    try:
        from langchain.agents.middleware.types import (
            AgentMiddleware,
            ModelRequest,
            ModelResponse,
        )
    except ImportError:
        return None

    class McpAuthNoticeMiddleware(AgentMiddleware):
        @staticmethod
        def _stamp(response: Any) -> Any:
            pending = pending_auth_requests()
            if not pending:
                return response
            try:
                message: Any = getattr(response, "result", None)
                if isinstance(message, list):
                    message = message[0] if message else None
                if message is None:
                    return response
                extra = getattr(message, "additional_kwargs", None)
                if not isinstance(extra, dict):
                    extra = {}
                    message.additional_kwargs = extra

                # Merge, do not overwrite. This used to assign, and
                # middleware order decided the outcome: whatever the
                # status middleware had narrowed to one button was
                # replaced by the full discovery list.
                existing = extra.get("auth_required")
                merged = list(existing) if isinstance(existing, list) else []
                seen = {r.get("mcp_name") for r in merged if isinstance(r, dict)}
                for request in pending:
                    if isinstance(request, dict) and request.get("mcp_name") in seen:
                        continue
                    merged.append(request)

                text = getattr(message, "text", "") or ""
                extra["auth_required"] = narrow_auth_requests(merged, text)
            except Exception:
                # A missing connect button is a worse answer; a failed
                # turn is no answer.
                logger.debug("Could not stamp auth requests", exc_info=True)
            return response

        def wrap_model_call(
            self, request: ModelRequest[Any], handler: Any
        ) -> ModelResponse[Any]:
            return self._stamp(handler(request))

        async def awrap_model_call(
            self, request: ModelRequest[Any], handler: Any
        ) -> ModelResponse[Any]:
            return self._stamp(await handler(request))

    return McpAuthNoticeMiddleware()
