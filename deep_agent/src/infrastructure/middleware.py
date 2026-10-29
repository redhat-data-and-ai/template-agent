"""Middleware builder for deepagents integration.

Converts ResolvedMiddlewareConfig into a list of AgentMiddleware instances
that can be passed to create_deep_agent(middleware=...).

This module is the bridge between declarative YAML config and the deepagents
middleware API. Template-agent users never import or call this directly.
"""

from __future__ import annotations

import contextvars
import importlib
import re
from datetime import datetime, timezone
from typing import Any

from deep_agent.src.agent.config.middleware import ResolvedMiddlewareConfig
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

_current_user_info: contextvars.ContextVar[dict[str, str] | None] = (
    contextvars.ContextVar("_current_user_info", default=None)
)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
_DELIMITER_RE = re.compile(r"[<>\[\]]")


def _sanitize_identity_value(value: str) -> str:
    """Strip control characters and delimiter characters from an identity value.

    Prevents a crafted JWT claim from injecting newlines or fake XML/markdown
    tags into the system message.
    """
    value = _CONTROL_CHAR_RE.sub("", value)
    value = _DELIMITER_RE.sub("", value)
    return value.strip()


def set_user_info(
    *,
    user_id: str | None = None,
    display_name: str | None = None,
    email: str | None = None,
) -> None:
    """Store the current user's identity fields for system-prompt injection.

    Values are sanitized at storage time to strip control characters and
    delimiter characters that could be used to inject fake instructions.
    """
    raw = {"user_id": user_id, "display_name": display_name, "email": email}
    info = {k: _sanitize_identity_value(v) for k, v in raw.items() if v}
    _current_user_info.set(info or None)


def get_current_datetime() -> str:
    """Get the current date and time (UTC) in a formatted string.

    Returns:
        The current date and time (UTC) formatted as "Month Day, Year, Hour:Minute:Second" (e.g., "September 25, 2026, 12:00:00").
    """
    return datetime.now(timezone.utc).strftime("%B %d, %Y, %H:%M:%S UTC")


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


def build_opa_middleware() -> Any | None:
    """Return OPAMiddleware when OPA authorization is enabled, else None."""
    from deep_agent.src.opa.config import is_opa_enabled

    if not is_opa_enabled():
        return None

    from deep_agent.src.opa.middleware import OPAMiddleware

    return OPAMiddleware()


def build_user_identity_middleware() -> Any | None:
    """Build middleware that injects the current user's identity into the system prompt.

    Reads from ``_current_user_info`` ContextVar at invocation time, so a
    cached graph still picks up the correct user per request.
    """
    try:
        from langchain.agents.middleware.types import (
            AgentMiddleware,
            ModelRequest,
            ModelResponse,
        )
    except ImportError:
        return None

    class UserIdentityMiddleware(AgentMiddleware):
        def _inject_identity(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
            info = _current_user_info.get()
            if not info:
                return request
            from deepagents.middleware.subagents import append_to_system_message

            parts = [f"  {k}: {v}" for k, v in info.items()]
            identity_block = (
                "<authenticated-user>\n" + "\n".join(parts) + "\n</authenticated-user>"
            )

            logger.info(
                "UserIdentityMiddleware: injected user identity into system prompt"
            )
            return request.override(
                system_message=append_to_system_message(
                    request.system_message, identity_block
                ),
            )

        def wrap_model_call(
            self, request: ModelRequest[Any], handler: Any
        ) -> ModelResponse[Any]:
            return handler(self._inject_identity(request))

        async def awrap_model_call(
            self, request: ModelRequest[Any], handler: Any
        ) -> ModelResponse[Any]:
            return await handler(self._inject_identity(request))

    return UserIdentityMiddleware()


#: Non-filesystem tools deepagents registers, so a name listed in
#: ``excluded_tools`` can be recognised rather than flagged as a typo.
#: Filesystem tool names are discovered from FilesystemMiddleware itself.
_KNOWN_NON_FS_TOOLS: frozenset[str] = frozenset(
    {"write_todos", "task", "compact_conversation"}
)


def _warn_unknown_excluded_tools(
    blocked: frozenset[str], fs_tool_names: list[str]
) -> None:
    """Warn about excluded names that match no known tool.

    A typo in ``excluded_tools`` would otherwise exclude nothing and say
    nothing, leaving the operator believing a tool is blocked when it is not.
    MCP tool names are not known here, so this warns rather than raises.

    Args:
        blocked: Tool names from config.
        fs_tool_names: Filesystem tools this deepagents version registers.
    """
    unknown = sorted(blocked - set(fs_tool_names) - _KNOWN_NON_FS_TOOLS)
    if unknown:
        logger.warning(
            "excluded_tools names match no built-in tool: %s. "
            "Check for a typo — built-ins are %s. "
            "(MCP tool names are not validated here.)",
            unknown,
            sorted(set(fs_tool_names) | _KNOWN_NON_FS_TOOLS),
        )


def build_restricted_filesystem_middleware(
    backend: Any, blocked: frozenset[str]
) -> Any | None:
    """Build a FilesystemMiddleware that never registers the blocked tools.

    ``create_deep_agent`` builds its own FilesystemMiddleware, but merges a
    caller-supplied one by ``.name`` — an instance named ``FilesystemMiddleware``
    replaces the internal one in place. Passing an explicit ``tools=`` opt-in
    list omits the rest entirely: per deepagents, a name outside that list
    "never reaches the dispatchable tool node". That is strictly stronger than
    hiding the tool from the model, which leaves it callable.

    Args:
        backend: Backend instance the filesystem tools operate on.
        blocked: Tool names to leave unregistered.

    Returns:
        Replacement middleware, or None when nothing filesystem-related is
        blocked (leave deepagents' own instance alone).
    """
    if not blocked:
        return None

    if backend is None:
        # Enforcement needs a backend to bind the replacement to. Say so
        # loudly: silently skipping it looks identical to having no policy.
        logger.warning(
            "excluded_tools %s NOT enforced for filesystem tools: no backend given",
            sorted(blocked),
        )
        return None

    try:
        from deepagents.middleware.filesystem import FilesystemMiddleware
    except ImportError:
        logger.warning("FilesystemMiddleware unavailable — excluded_tools not applied")
        return None

    try:
        default_names = [t.name for t in FilesystemMiddleware(backend=backend).tools]
    except Exception as e:
        logger.warning("Could not enumerate filesystem tools: %s", e)
        return None

    _warn_unknown_excluded_tools(blocked, default_names)

    keep = [name for name in default_names if name not in blocked]
    if len(keep) == len(default_names):
        return None  # nothing blocked is a filesystem tool

    removed = sorted(set(default_names) - set(keep))
    logger.info("Filesystem tools not registered (excluded_tools): %s", removed)
    return FilesystemMiddleware(backend=backend, tools=keep)


def build_tool_access_middleware(blocked: frozenset[str]) -> Any | None:
    """Build middleware enforcing ``middleware.excluded_tools`` from agent.yaml.

    Three layers, because hiding a tool is not the same as denying it:

    1. the tool is stripped from the model request, so the model never sees it;
    2. a call that arrives anyway (hallucinated name, replayed transcript,
       subagent with its own schema) is refused at the tool node;
    3. the system prompt names the blocked tools so the agent can tell the
       user plainly instead of inventing a reason.

    Args:
        blocked: Tool names denied for this agent.

    Returns:
        Middleware instance, or None if nothing is blocked or the langchain
        middleware API is unavailable.
    """
    if not blocked:
        return None

    try:
        from langchain.agents.middleware.types import (
            AgentMiddleware,
            ModelRequest,
            ModelResponse,
        )
        from langchain_core.messages import ToolMessage
    except ImportError:
        logger.warning("langchain middleware API unavailable — excluded_tools ignored")
        return None

    denial = (
        "Tool '%s' is disabled by configuration and was not executed. "
        "Tell the user this action is not permitted, and do not retry it."
    )

    class ToolAccessMiddleware(AgentMiddleware):
        def _apply(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
            from deepagents.middleware.subagents import append_to_system_message

            visible = [t for t in request.tools if _tool_name(t) not in blocked]
            names = ", ".join(sorted(blocked))
            notice = (
                "<disabled-tools>\n"
                f"These tools are disabled and unavailable: {names}.\n"
                "If the user asks for one of these actions, say plainly that it "
                "is not permitted and suggest an alternative. Never claim to "
                "have performed it.\n"
                "</disabled-tools>"
            )
            return request.override(
                tools=visible,
                system_message=append_to_system_message(request.system_message, notice),
            )

        def wrap_model_call(
            self, request: ModelRequest[Any], handler: Any
        ) -> ModelResponse[Any]:
            return handler(self._apply(request))

        async def awrap_model_call(
            self, request: ModelRequest[Any], handler: Any
        ) -> ModelResponse[Any]:
            return await handler(self._apply(request))

        def wrap_tool_call(self, request: Any, handler: Any) -> Any:
            name = request.tool_call.get("name")
            if name in blocked:
                logger.warning("Blocked call to disabled tool '%s'", name)
                return ToolMessage(
                    content=denial % name,
                    tool_call_id=request.tool_call.get("id", ""),
                    name=name,
                    status="error",
                )
            return handler(request)

        async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
            name = request.tool_call.get("name")
            if name in blocked:
                logger.warning("Blocked call to disabled tool '%s'", name)
                return ToolMessage(
                    content=denial % name,
                    tool_call_id=request.tool_call.get("id", ""),
                    name=name,
                    status="error",
                )
            return await handler(request)

    logger.info("Tool access: %d tool(s) disabled: %s", len(blocked), sorted(blocked))
    return ToolAccessMiddleware()


def _tool_name(tool: Any) -> str | None:
    """Extract a tool name from a BaseTool or a dict-shaped tool spec."""
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def build_current_datetime_middleware() -> Any | None:
    """Build middleware that injects the current date and time (UTC) into the system prompt."""
    try:
        from langchain.agents.middleware.types import (
            AgentMiddleware,
            ModelRequest,
            ModelResponse,
        )
    except ImportError:
        return None

    class CurrentDatetimeMiddleware(AgentMiddleware):
        def _inject_current_datetime(
            self, request: ModelRequest[Any]
        ) -> ModelRequest[Any]:
            from deepagents.middleware.subagents import append_to_system_message

            current_datetime = get_current_datetime()
            current_datetime_block = (
                "<current-datetime>\n" + current_datetime + "\n</current-datetime>"
            )

            return request.override(
                system_message=append_to_system_message(
                    request.system_message, current_datetime_block
                ),
            )

        def wrap_model_call(
            self, request: ModelRequest[Any], handler: Any
        ) -> ModelResponse[Any]:
            return handler(self._inject_current_datetime(request))

        async def awrap_model_call(
            self, request: ModelRequest[Any], handler: Any
        ) -> ModelResponse[Any]:
            return await handler(self._inject_current_datetime(request))

    return CurrentDatetimeMiddleware()


def _build_gemini_safety_log_middleware() -> Any | None:
    """Build middleware that logs and replaces Gemini safety-blocked empty responses."""
    try:
        from langchain.agents.middleware import AgentMiddleware
    except ImportError:
        return None

    _SAFETY_FINISH_REASONS = frozenset(
        {
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "IMAGE_SAFETY",
            "SPII",
            "content_filter",
            "refusal",
        }
    )
    _PROMPT_BLOCK_REASONS = frozenset(
        {
            "SAFETY",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "IMAGE_SAFETY",
            "JAILBREAK",
            "MODEL_ARMOR",
            "OTHER",
        }
    )
    _REFUSAL = (
        "I'm unable to respond to that request as it was flagged by the "
        "content safety filter. Please rephrase your message."
    )

    class GeminiSafetyLogMiddleware(AgentMiddleware):
        """Detect Gemini safety filter blocks and log them.

        Handles two distinct blocking paths:
          1. Candidate-level: Gemini returns an empty AIMessage with
             finish_reason=SAFETY (response generated but blocked).
          2. Prompt-level: Gemini returns an empty AIMessage with only
             response_metadata["prompt_feedback"]["block_reason"] set
             (prompt itself rejected, no candidate produced).
        """

        def after_model(self, state: Any, runtime: Any) -> Any:
            return self._check_and_replace(state)

        async def aafter_model(self, state: Any, runtime: Any) -> Any:
            return self._check_and_replace(state)

        def _check_and_replace(self, state: Any) -> Any:
            from langchain_core.messages import AIMessage

            msgs = state.get("messages", []) if isinstance(state, dict) else []
            if not msgs:
                return None
            last = msgs[-1]
            if not isinstance(last, AIMessage):
                return None
            if last.content or last.tool_calls:
                return None
            meta = getattr(last, "response_metadata", {}) or {}

            reason = meta.get("finish_reason", meta.get("stop_reason", ""))
            if reason in _SAFETY_FINISH_REASONS:
                logger.warning(
                    "Gemini safety filter blocked response: "
                    "finish_reason=%s, model=%s, safety_ratings=%s",
                    reason,
                    meta.get("model_name", "unknown"),
                    meta.get("safety_ratings", []),
                )
                msgs[-1] = AIMessage(content=_REFUSAL, id=last.id)
                return {"messages": msgs}

            prompt_feedback = meta.get("prompt_feedback") or {}
            block_reason = prompt_feedback.get("block_reason", "")
            if isinstance(block_reason, int):
                block_reason = ""
            if block_reason in _PROMPT_BLOCK_REASONS:
                logger.warning(
                    "Gemini prompt-level block: block_reason=%s, safety_ratings=%s",
                    block_reason,
                    prompt_feedback.get("safety_ratings", []),
                )
                msgs[-1] = AIMessage(content=_REFUSAL, id=last.id)
                return {"messages": msgs}

            return None

    return GeminiSafetyLogMiddleware()


def _build_image_sanitize_middleware() -> Any | None:
    """Build middleware that strips markdown image tags from user input and model responses."""
    try:
        from langchain.agents.middleware import AgentMiddleware
    except ImportError:
        return None

    import re

    _IMG_INLINE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
    _IMG_REF = re.compile(r"!\[([^\]]*)\]\[([^\]]+)\]")
    _REF_DEF = re.compile(r"^\s{0,3}\[([^\]]+)\]:\s+\S+[^\n]*$", re.MULTILINE)

    def _strip_images(text: str) -> str:
        used_labels: set[str] = set()
        result = text
        while True:
            for match in _IMG_REF.finditer(result):
                used_labels.add(match.group(2).lower())
            updated = _IMG_REF.sub(r"\1", _IMG_INLINE.sub(r"\1", result))
            if updated == result:
                break
            result = updated
        if used_labels:
            result = _REF_DEF.sub(
                lambda m: "" if m.group(1).lower() in used_labels else m.group(0),
                result,
            )
        return result

    def _sanitize_content(content: Any) -> tuple[Any, bool]:
        """Sanitize string or list-based message content. Returns (new_content, changed)."""
        if isinstance(content, str):
            sanitized = _strip_images(content)
            return sanitized, sanitized != content
        if isinstance(content, list):
            changed = False
            new_blocks: list[Any] = []
            for block in content:
                if isinstance(block, str):
                    sanitized = _strip_images(block)
                    if sanitized != block:
                        changed = True
                    new_blocks.append(sanitized)
                elif (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                ):
                    sanitized = _strip_images(block["text"])
                    if sanitized != block["text"]:
                        changed = True
                        new_blocks.append({**block, "text": sanitized})
                    else:
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)
            return new_blocks, changed
        return content, False

    class ImageSanitizeMiddleware(AgentMiddleware):
        """Strip markdown image tags from both user input and AI responses."""

        def before_model(self, state: Any, runtime: Any) -> Any:
            return self._sanitize_input(state)

        async def abefore_model(self, state: Any, runtime: Any) -> Any:
            return self._sanitize_input(state)

        def after_model(self, state: Any, runtime: Any) -> Any:
            return self._sanitize_output(state)

        async def aafter_model(self, state: Any, runtime: Any) -> Any:
            return self._sanitize_output(state)

        def _sanitize_input(self, state: Any) -> Any:
            from langchain_core.messages import HumanMessage, ToolMessage

            msgs = state.get("messages", []) if isinstance(state, dict) else []
            if not msgs:
                return None
            modified = False
            for i in range(len(msgs) - 1, -1, -1):
                msg = msgs[i]
                if isinstance(msg, (HumanMessage, ToolMessage)) and msg.content:
                    sanitized, changed = _sanitize_content(msg.content)
                    if changed:
                        label = (
                            "user input"
                            if isinstance(msg, HumanMessage)
                            else "tool result"
                        )
                        logger.warning("Stripped markdown image tag(s) from %s", label)
                        msgs[i] = msg.model_copy(update={"content": sanitized})
                        modified = True
            if not modified:
                return None
            return {"messages": msgs}

        def _sanitize_output(self, state: Any) -> Any:
            from langchain_core.messages import AIMessage

            msgs = state.get("messages", []) if isinstance(state, dict) else []
            if not msgs:
                return None
            last = msgs[-1]
            if not isinstance(last, AIMessage):
                return None
            if not last.content:
                return None
            sanitized, changed = _sanitize_content(last.content)
            if not changed:
                return None
            logger.warning("Stripped markdown image tag(s) from model response")
            msgs[-1] = last.model_copy(update={"content": sanitized})
            return {"messages": msgs}

    return ImageSanitizeMiddleware()


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

    opa_mw = build_opa_middleware()
    if opa_mw is not None:
        middlewares.append(opa_mw)

    safety_mw = _build_gemini_safety_log_middleware()
    if safety_mw is not None:
        middlewares.append(safety_mw)

    identity_mw = build_user_identity_middleware()
    if identity_mw is not None:
        middlewares.append(identity_mw)

    datetime_mw = build_current_datetime_middleware()
    if datetime_mw is not None:
        middlewares.append(datetime_mw)

    image_sanitize_mw = _build_image_sanitize_middleware()
    if image_sanitize_mw is not None:
        middlewares.append(image_sanitize_mw)

    # Tool exclusion is a safety policy, not an optional feature, so it is
    # applied before the MIDDLEWARE_ENABLED early return. Otherwise disabling
    # middleware would re-register `delete` for the orchestrator while
    # subagents — which have no such switch — kept blocking it.
    _append_tool_exclusions(middlewares, resolved, backend)

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


def _append_tool_exclusions(
    target: list[Any],
    resolved: ResolvedMiddlewareConfig,
    backend: Any | None,
) -> None:
    """Append the middleware enforcing ``middleware.excluded_tools``.

    Args:
        target: Middleware list to append to.
        resolved: Resolved config carrying ``blocked_tools``.
        backend: Backend the replacement FilesystemMiddleware binds to.
    """
    # Replaces deepagents' own FilesystemMiddleware so blocked filesystem
    # tools are never registered at all.
    _append_if_built(
        target, build_restricted_filesystem_middleware(backend, resolved.blocked_tools)
    )

    # Backstop for blocked tools that are not filesystem tools (MCP, task),
    # and the source of the user-facing "not permitted" wording.
    _append_if_built(target, build_tool_access_middleware(resolved.blocked_tools))


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
    from deep_agent.src.memory.instructions import resolve_memory_namespaces

    return resolve_memory_namespaces(resolved.memory_namespaces)


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
