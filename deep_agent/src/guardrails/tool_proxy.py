"""GuardianToolProxy — BaseTool subclass that safety-screens tool args and results.

Extends BaseTool so LangGraph's ToolNode isinstance(tool, BaseTool) check passes.
Overrides ainvoke to:
  1. Pre-check args before the inner tool runs (blocks unsafe LLM-generated inputs).
  2. Catch inner-tool exceptions so a malformed arg doesn't cancel the parallel batch.
  3. Post-check the result after the inner tool runs, replacing unsafe output.

Parallel tool behaviour: each proxy is independent. asyncio.gather sees normal
returns (no exceptions) in all three paths, so other parallel tools are unaffected.

Persistence: placeholders become real ToolMessages in LangGraph state and are
checkpointed, so conversation history survives reload/resume.
"""

from __future__ import annotations

from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, PrivateAttr

from deep_agent.src.guardrails import TOOL_SAFETY_REFUSAL
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

BLOCKED_RESULT = (
    "[SAFETY_BLOCKED] This tool's result was blocked by the content safety policy. "
    "Reply to the user with this exact sentence and nothing else: "
    f"'{TOOL_SAFETY_REFUSAL}' "
    "Do not describe this as an error. Do not suggest retrying. Do not call this tool any further tools."
)

BLOCKED_INPUT = (
    "[SAFETY_BLOCKED] The arguments for this tool were blocked by the content safety policy. "
    f"Reply to the user with this exact sentence and nothing else: "
    f"'{TOOL_SAFETY_REFUSAL}' "
    "Do not describe this as an error. Do not suggest retrying. Do not call any further tools."
)


def _make_blocked_result(original_result: Any) -> Any:
    """Return BLOCKED_RESULT packaged to match LangGraph's _normalize_tool_response rules.

    LangGraph ToolNode only accepts ToolMessage, Command, or list[Command | ToolMessage]
    from tool.ainvoke(). BaseTool.arun → _format_output wraps the raw _arun return in a
    ToolMessage (using tool_call_id from the ToolCall input). deepagents' task tool returns
    a Command containing a ToolMessage. We must mirror those types here.
    """
    from langchain_core.messages import ToolMessage

    if isinstance(original_result, ToolMessage):
        return ToolMessage(
            content=BLOCKED_RESULT,
            name=original_result.name,
            tool_call_id=original_result.tool_call_id,
            status="success",
        )

    # deepagents task tool returns a Command whose update["messages"] contains a ToolMessage
    try:
        from langgraph.types import Command

        if isinstance(original_result, Command):
            update = getattr(original_result, "update", {}) or {}
            if isinstance(update, dict) and "messages" in update:
                new_msgs = []
                for m in update["messages"]:
                    if isinstance(m, ToolMessage):
                        new_msgs.append(
                            ToolMessage(
                                content=BLOCKED_RESULT,
                                name=m.name,
                                tool_call_id=m.tool_call_id,
                                status="success",
                            )
                        )
                    else:
                        new_msgs.append(m)
                return Command(update={**update, "messages": new_msgs})
            return original_result
    except ImportError:
        pass

    # Fallback: return as-is (unknown type — let LangGraph surface the error)
    logger.warning(
        "_make_blocked_result: unrecognised result type %s",
        type(original_result).__name__,
    )
    return BLOCKED_RESULT


def _signal_safety_block(config: Any) -> None:
    """Set the blocked flag in the shared safety context injected by SafetyAwareRunnable."""
    if not isinstance(config, dict):
        return
    ctx = config.get("_safety_ctx")
    if isinstance(ctx, dict):
        ctx["blocked"] = True


def _get_tool_call_id(input: Any) -> str:
    """Extract tool_call_id from a LangGraph ToolCall dict."""
    if isinstance(input, dict):
        return str(input.get("id", ""))
    return ""


def _make_blocked_input_result(tool_name: str, input: Any) -> Any:
    """Return BLOCKED_INPUT as a ToolMessage when args are flagged before execution."""
    from langchain_core.messages import ToolMessage

    return ToolMessage(
        content=BLOCKED_INPUT,
        name=tool_name,
        tool_call_id=_get_tool_call_id(input),
        status="success",
    )


def _make_error_result(tool_name: str, input: Any, exc: Exception) -> Any:
    """Return a ToolMessage when the inner tool raises, preserving parallel isolation."""
    from langchain_core.messages import ToolMessage

    return ToolMessage(
        content=f"[TOOL_ERROR] Tool execution failed: {exc}",
        name=tool_name,
        tool_call_id=_get_tool_call_id(input),
        status="error",
    )


class GuardianToolProxy(BaseTool):
    """Transparent BaseTool wrapper that replaces unsafe results with BLOCKED_RESULT.

    All attributes (name, description, args_schema) are copied from the inner tool
    so LangGraph and deepagents see the same interface.  Only ainvoke is overridden
    to add the Guardian post-check; _run delegates to inner.invoke for the sync path.
    """

    name: str = ""
    description: str = ""
    _inner: Any = PrivateAttr()

    def __init__(self, inner_tool: Any) -> None:
        """Copy name/description/args_schema from inner_tool and store the reference."""
        schema: Optional[Type[BaseModel]] = getattr(inner_tool, "args_schema", None)
        super().__init__(
            name=getattr(inner_tool, "name", ""),
            description=getattr(inner_tool, "description", ""),
            args_schema=schema,
        )
        self._inner = inner_tool

    # ainvoke is the hot path — LangGraph's ToolNode calls this.
    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Pre-check args, execute inner tool, and post-check the result for safety."""
        from langchain_core.messages import ToolMessage as _TM

        from deep_agent.src.guardrails import get_guardrails_config
        from deep_agent.src.guardrails.client import check_safety
        from deep_agent.src.settings import settings

        # Pass through immediately if guardrails have been runtime-disabled.
        if get_guardrails_config() is None:
            return await self._inner.ainvoke(input, config, **kwargs)

        # Phase 1: pre-check args before the inner tool executes.
        if settings.GUARDIAN_API_BASE:
            arg_text = str(input)
            is_safe, verdict = await check_safety(arg_text, context="tool_input")
            if not is_safe:
                logger.warning(
                    "guardian_blocked_tool_input",
                    tool=self.name,
                    verdict=verdict,
                )
                _signal_safety_block(config)
                return _make_blocked_input_result(self.name, input)

        # Phase 2: execute inner tool; catch exceptions to preserve parallel isolation.
        try:
            result = await self._inner.ainvoke(input, config, **kwargs)
        except Exception as exc:
            logger.warning(
                "tool_invocation_failed",
                tool=self.name,
                error=str(exc),
            )
            return _make_error_result(self.name, input, exc)

        # Phase 3: post-check the result.
        if isinstance(result, _TM):
            content = str(result.content)
        else:
            content = str(result) if result is not None else ""
        if not content or not settings.GUARDIAN_API_BASE:
            return result

        is_safe, verdict = await check_safety(content, context="tool_result")
        if not is_safe:
            logger.warning(
                "guardian_blocked_tool_result", tool=self.name, verdict=verdict
            )
            _signal_safety_block(config)
            return _make_blocked_result(result)

        from deep_agent.src.guardrails.client import check_injection

        is_safe, verdict = await check_injection(content, context="tool_result")
        if not is_safe:
            logger.warning(
                "guardian_injection_blocked_tool_result",
                tool=self.name,
                verdict=verdict,
            )
            _signal_safety_block(config)
            return _make_blocked_result(result)

        return result

    # _run satisfies BaseTool's abstract requirement; used only in sync contexts.
    # Guardian screening is NOT applied here — callers must use ainvoke.
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.invoke(*args, **kwargs)


def wrap_tools(tools: list[Any]) -> list[Any]:
    """Wrap a list of tools with GuardianToolProxy when Guardian is enabled."""
    from deep_agent.src.guardrails import get_guardrails_config
    from deep_agent.src.settings import settings

    if not settings.GUARDIAN_API_BASE or not tools:
        return tools
    cfg = get_guardrails_config()
    if cfg is None or not cfg.enabled:
        return tools
    return [GuardianToolProxy(t) for t in tools]
