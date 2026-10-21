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
import time
from collections.abc import Iterable
from contextvars import ContextVar
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
    """Get the current date and time (UTC), rounded for cache stability.

    Rounded down to ``DATETIME_PROMPT_GRANULARITY_SECONDS``, which is
    the difference between a prompt prefix that can be cached and one
    that cannot.

    This block is appended to the system message on *every model
    call*. The system message is the start of the prefix a provider
    caches, and the tool definitions sit behind it — so a value that
    changed every second meant the prefix was unique on every call,
    every turn, for every user. Nothing downstream of it could ever
    be cached: not the system prompt, not the hundred-odd tool
    schemas that make up most of the input tokens. Seconds of
    precision no one asked for were paying for the whole prefix,
    every time.

    Rounding to the hour makes the prefix identical across all users
    and all turns within that hour. An agent that needs the time to
    the second can call a tool for it; the cost of putting it here is
    the entire cache.

    Returns:
        A formatted UTC timestamp, e.g. "September 25, 2026, 12:00:00 UTC".
    """
    from deep_agent.src.settings import settings

    now = datetime.now(timezone.utc)
    step = max(1, settings.DATETIME_PROMPT_GRANULARITY_SECONDS)
    if step > 1:
        epoch = int(now.timestamp()) // step * step
        now = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return now.strftime("%B %d, %Y, %H:%M:%S UTC")


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


def build_prompt_size_middleware() -> Any | None:
    """Log what each model call actually sends, split by component.

    Added because the input cost could not be explained. A turn that
    answered "ok" in one word recorded 119,769 input tokens, and the
    obvious culprit — tool schemas — turned out not to be it: the
    same figure appeared with 11 tools bound and with 322. The agent
    config on disk is about 9k tokens in total, so the remainder was
    coming from somewhere nobody was looking.

    Guessing at that from the outside wasted a round of analysis.
    This measures it instead, at the only place that sees the
    assembled request.
    """
    if not settings.LOG_PROMPT_SIZE:
        return None

    try:
        from langchain.agents.middleware.types import (
            AgentMiddleware,
            ModelRequest,
            ModelResponse,
        )
    except ImportError:
        return None

    class PromptSizeMiddleware(AgentMiddleware):
        @staticmethod
        def _report(request: ModelRequest[Any]) -> None:
            try:
                system = request.system_message or ""
                system_chars = len(str(system))

                messages = getattr(request, "messages", None) or []
                message_chars = sum(
                    len(str(getattr(m, "content", m))) for m in messages
                )

                tools = getattr(request, "tools", None) or []
                tool_chars = 0
                for tool in tools:
                    tool_chars += len(str(getattr(tool, "description", "") or ""))
                    schema = getattr(tool, "args_schema", None)
                    if schema is not None:
                        tool_chars += len(str(schema))

                total = system_chars + message_chars + tool_chars
                logger.info(
                    "prompt_size",
                    # Roughly four characters to a token. Good enough
                    # to apportion blame between three components;
                    # not a billing figure.
                    approx_tokens=total // 4,
                    system_chars=system_chars,
                    message_chars=message_chars,
                    tool_chars=tool_chars,
                    tools=len(tools),
                    messages=len(messages),
                )
            except Exception:
                # Diagnostics must never be the reason a turn fails.
                logger.debug("Could not measure prompt size", exc_info=True)

        def wrap_model_call(
            self, request: ModelRequest[Any], handler: Any
        ) -> ModelResponse[Any]:
            self._report(request)
            started = time.monotonic()
            try:
                return handler(request)
            finally:
                _log_model_call(time.monotonic() - started)

        async def awrap_model_call(
            self, request: ModelRequest[Any], handler: Any
        ) -> ModelResponse[Any]:
            self._report(request)
            started = time.monotonic()
            try:
                return await handler(request)
            finally:
                _log_model_call(time.monotonic() - started)

    return PromptSizeMiddleware()


def _log_model_call(seconds: float) -> None:
    """Time spent inside the model call itself.

    This middleware sits outermost, so the span covers the provider
    round trip plus every middleware nested below it. Compared
    against the run's own duration it answers the question that
    kept coming back: when a turn takes three seconds, is that the
    model or is it us? Guessing at that cost two rounds of analysis
    — first blaming prefill, then blaming a poll loop that turned
    out to be only part of it.
    """
    logger.info("model_call", seconds=round(seconds, 3))


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


def _build_model_router_middleware() -> Any | None:
    """Per-request model tier selection, when router.yaml enables it."""
    try:
        from deep_agent.src.routing import build_model_router_middleware

        return build_model_router_middleware()
    except Exception as exc:
        # Routing is an optimisation; it must never stop the agent
        # starting.
        logger.error("Model router unavailable: %s", exc, exc_info=True)
        return None


def build_middleware_list(
    resolved: ResolvedMiddlewareConfig,
    *,
    model: Any | None = None,
    model_name: str = "",
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

    # Registered after policy, before anything that inspects the response:
    # routing only swaps which model serves the call.
    router_mw = _build_model_router_middleware()
    if router_mw is not None:
        middlewares.append(router_mw)

    safety_mw = _build_gemini_safety_log_middleware()
    if safety_mw is not None:
        middlewares.append(safety_mw)

    identity_mw = build_user_identity_middleware()
    if identity_mw is not None:
        middlewares.append(identity_mw)

    datetime_mw = build_current_datetime_middleware()
    if datetime_mw is not None:
        middlewares.append(datetime_mw)

    prompt_size_mw = build_prompt_size_middleware()
    if prompt_size_mw is not None:
        middlewares.append(prompt_size_mw)

    from deep_agent.aegra.mcp_tool_auth import build_mcp_auth_notice_middleware

    auth_notice_mw = build_mcp_auth_notice_middleware()
    if auth_notice_mw is not None:
        middlewares.append(auth_notice_mw)

    image_sanitize_mw = _build_image_sanitize_middleware()
    if image_sanitize_mw is not None:
        middlewares.append(image_sanitize_mw)

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

    _append_guardrails(middlewares, resolved, model_name)

    for dotted_path in resolved.extra_middleware:
        _append_if_built(middlewares, _import_middleware(dotted_path))

    if middlewares:
        logger.info("Built %d extra middleware instance(s)", len(middlewares))
    return middlewares


def _append_if_built(target: list[Any], mw: Any | None) -> None:
    """Append middleware to list if it was built successfully."""
    if mw is not None:
        target.append(mw)


def _append_guardrails(
    target: list[Any],
    resolved: ResolvedMiddlewareConfig,
    model_name: str = "",
) -> None:
    """Build and append all production guardrail middleware."""
    if resolved.model_call_limit.enabled:
        _append_if_built(
            target, _build_model_call_limit(resolved.model_call_limit.run_limit)
        )

    if resolved.tool_call_limit.enabled:
        _append_if_built(
            target, _build_tool_call_limit(resolved.tool_call_limit.run_limit)
        )
        for limiter in _build_per_tool_limits(resolved.tool_call_limit.per_tool):
            _append_if_built(target, limiter)

    # The orchestrator's copy; subagents build their own list and
    # apply the same exclusions there, against their own model.
    for exclusion in _build_tool_exclusions(_profile_excluded_tools(model_name)):
        _append_if_built(target, exclusion)

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


def _build_per_tool_limits(per_tool: dict[str, int]) -> list[Any]:
    """Cap individual tools that loop.

    The run limit is 200 and never fires. What costs a user real
    time is one tool called over and over inside a single answer:
    one query tool ran six times in one turn at ~15s each,
    the agent trying variation after variation of a query that was
    not going to answer the question.

    `exit_behavior='continue'` blocks the exhausted tool and leaves
    the agent running, so it has to answer with what it already has
    instead of being killed mid-turn. Telling the model in prose to
    "run it once" did not survive contact with a query returning no
    rows.
    """
    built: list[Any] = []
    # A dict, or nothing. Config can arrive as a stand-in object and
    # a limiter built from one caps a tool nobody named.
    if not isinstance(per_tool, dict) or not per_tool:
        return built
    try:
        from langchain.agents.middleware import ToolCallLimitMiddleware
    except ImportError:
        logger.debug("ToolCallLimitMiddleware not available")
        return built

    for tool_name, limit in per_tool.items():
        if not isinstance(limit, int) or limit <= 0:
            continue
        built.append(
            ToolCallLimitMiddleware(
                tool_name=tool_name,
                run_limit=limit,
                exit_behavior="continue",
            )
        )
        logger.info("Tool call limit: %s capped at %d per run", tool_name, limit)
    return built


# Tools the approval policy refuses for this build.
#
# A ContextVar rather than an argument because the exclusion list is
# assembled in two places — the orchestrator's middleware and each
# subagent's — and both have to agree. Removing a denied tool from
# only the orchestrator's `tools` left it reachable from every
# subagent, which are built from the unfiltered MCP list.
_policy_denied_tools: ContextVar[tuple[str, ...]] = ContextVar(
    "policy_denied_tools", default=()
)


def set_policy_denied_tools(names: Iterable[str]) -> None:
    """Record the tools the approval policy denies for this graph build."""
    _policy_denied_tools.set(tuple(sorted(set(names))))


def _profile_excluded_tools(model_name: str = "") -> list[str]:
    """Every tool to remove, whatever the reason.

    Three sources, deliberately separate. A harness profile's
    ``excluded_tools`` means the model cannot use it, and the Models
    page shows those as limitations of that model.
    ``middleware.tool_exclusion`` means we removed it on purpose —
    its job moved into a skill — which is not a fact about the model
    and must not be shown as one. Enforcement does not care which;
    only the reader does.

    ``model_name`` picks the harness profile, which is keyed by
    model. Merging every profile applied one model's limitation to
    agents running on models that do not share it, which is the
    opposite of what a per-model profile is for.
    """
    names: set[str] = set()
    try:
        from deep_agent.src.agent.config import agent_config

        profiles = agent_config.get_providers_config().harness_profiles or {}
        profile = profiles.get(model_name) if model_name else None
        if profile is not None:
            for tool in getattr(profile, "excluded_tools", None) or []:
                names.add(str(tool))
    except Exception:
        logger.debug("Could not read harness excluded tools", exc_info=True)

    try:
        from deep_agent.src.agent.config import agent_config

        exclusion = agent_config.resolve_agent_middleware("", None).tool_exclusion
        if getattr(exclusion, "enabled", True):
            for tool in getattr(exclusion, "tools", None) or []:
                names.add(str(tool))
    except Exception:
        logger.debug("Could not read configured tool exclusions", exc_info=True)

    # And whatever the approval policy denies. `deny` means the tool
    # may not run at all, so leaving it out of `interrupt_on` is not
    # enough: without this it simply ran, unprompted, and under
    # `effects: dry_run` that is every acting tool.
    names.update(_policy_denied_tools.get())

    return sorted(names)


def _build_tool_exclusions(excluded: list[str]) -> list[Any]:
    """Remove named tools from an agent's reach."""
    if not excluded:
        return []
    try:
        from deepagents.middleware.tool_exclusion import ToolExclusionMiddleware
    except ImportError:
        try:
            from deepagents.graph import (
                _ToolExclusionMiddleware as ToolExclusionMiddleware,
            )
        except Exception:
            logger.warning("Tool exclusion middleware unavailable")
            return []
    logger.info("Excluding tools: %s", ", ".join(excluded))
    return [ToolExclusionMiddleware(excluded=excluded)]


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
            # All other errors: preserve the standard format for the
            # user, but log the traceback. Returning only the string
            # meant failures like "ValueError: string argument should
            # contain only ASCII characters" reached the transcript
            # with nothing in the logs to locate them by — no frame, no
            # field, no value.
            exc_type = type(exc).__name__
            logger.exception("Model call failed: %s: %s", exc_type, exc)
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
