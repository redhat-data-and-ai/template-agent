"""Safety-aware graph and runnable wrappers for Granite Guardian integration.

Shared by the orchestrator graph (graph.py) and subagent construction
(subagents.py) so ContentSafetyError is caught and converted to a clean
refusal message at every execution boundary, including inside subagents
and their skills.
"""

from __future__ import annotations

from typing import Any

from deep_agent.src.guardrails import (
    TOOL_SAFETY_REFUSAL as _TOOL_SAFETY_REFUSAL,
)
from deep_agent.src.guardrails import (
    ContentSafetyError,
    InputContentSafetyError,
    ToolContentSafetyError,
)
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_INPUT_SAFETY_REFUSAL = "I can't help with that request due to content safety policy."


def safety_refusal(exc: BaseException) -> str | None:
    """Walk the exception chain and return the appropriate refusal message.

    ModelRetryMiddleware raises a fresh exception with the original class name
    embedded in the message string but NOT in __cause__/__context__ (it collects
    exceptions across retries and raises after the loop, so the raise is outside
    any except block).  We therefore check both the exception type and the message
    text at each step.
    Returns None if no safety-related error is found anywhere in the chain.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, ToolContentSafetyError):
            return _TOOL_SAFETY_REFUSAL
        if isinstance(current, InputContentSafetyError):
            return _INPUT_SAFETY_REFUSAL
        if isinstance(current, ContentSafetyError):
            return _INPUT_SAFETY_REFUSAL
        # ModelRetryMiddleware raises a wrapper whose message contains the
        # original class name — check the string representation as a fallback.
        msg = str(current)
        if "ToolContentSafetyError" in msg:
            return _TOOL_SAFETY_REFUSAL
        if "InputContentSafetyError" in msg or "ContentSafetyError" in msg:
            return _INPUT_SAFETY_REFUSAL
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _build_merged_config(config: Any) -> tuple[dict, dict]:
    """Inject a shared _safety_ctx into config so GuardianToolProxy can signal blocks."""
    safety_ctx: dict = {"blocked": False}
    base = config or {}
    merged = {
        **base,
        "_safety_ctx": safety_ctx,
        "metadata": {**(base.get("metadata") or {}), "_safety_ctx": safety_ctx},
    }
    return merged, safety_ctx


class SafetyAwareRunnable:
    """Proxy over any async runnable that converts ContentSafetyError to a refusal message.

    Used to wrap both the orchestrator's compiled graph (_SafetyAwareGraph alias)
    and CompiledSubAgent runnables so that safety errors raised anywhere inside
    the runnable — including in skills — produce a consistent user-facing message
    instead of crashing or being stringified by deepagents.

    Tool-result safety is handled upstream by GuardianToolProxy, which replaces
    unsafe results with a safe placeholder before they enter LangGraph state.
    This runnable only needs to handle input safety errors (from on_chat_model_start
    via ModelRetryMiddleware).

    outermost=True  (orchestrator graph): catches all safety exceptions.
    outermost=False (inner subagent runnables): re-raises so the outermost catches it.
    """

    def __init__(self, runnable: Any, *, outermost: bool = False) -> None:
        """Wrap runnable, flagging whether this is the outermost safety boundary."""
        self._runnable = runnable
        self._outermost = outermost

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped runnable."""
        return getattr(self._runnable, name)

    def copy(self, **kwargs: Any) -> "SafetyAwareRunnable":
        """Return a wrapped copy so Aegra's checkpointer injection stays inside the proxy."""
        return SafetyAwareRunnable(
            self._runnable.copy(**kwargs), outermost=self._outermost
        )

    def with_config(self, config: Any = None, **kwargs: Any) -> "SafetyAwareRunnable":
        """Re-wrap after with_config so SafetyAwareRunnable is not stripped by __getattr__."""
        if config is not None:
            inner = self._runnable.with_config(config, **kwargs)
        else:
            inner = self._runnable.with_config(**kwargs)
        return SafetyAwareRunnable(inner, outermost=self._outermost)

    # ── Core async interface ──────────────────────────────────────────

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Invoke the runnable, converting safety errors to a refusal message at the boundary."""
        logger.debug(
            "safety_aware_runnable ainvoke called outermost=%s", self._outermost
        )
        try:
            merged_config, safety_ctx = _build_merged_config(config)
            result = await self._runnable.ainvoke(input, merged_config, **kwargs)
            # Override LLM output with consistent refusal if any tool was safety-blocked.
            # Run at every level (not just outermost) so that inner SafetyAwareRunnables
            # (e.g. analyst subagent, outermost=False) also override their final AIMessage.
            # This puts _TOOL_SAFETY_REFUSAL into the task tool's return value, which the
            # orchestrator's on_tool_end sentinel check can then detect.
            from langchain_core.messages import AIMessage, ToolMessage

            msgs = list(result.get("messages", []) if isinstance(result, dict) else [])
            tool_blocked = safety_ctx["blocked"] or any(
                isinstance(m, ToolMessage) and _TOOL_SAFETY_REFUSAL in str(m.content)
                for m in msgs
            )
            if tool_blocked:
                for i in range(len(msgs) - 1, -1, -1):
                    if isinstance(msgs[i], AIMessage):
                        msgs[i] = AIMessage(content=_TOOL_SAFETY_REFUSAL)
                        break
                result = {
                    **(result if isinstance(result, dict) else {}),
                    "messages": msgs,
                }
            return result
        except Exception as exc:
            if not self._outermost:
                raise
            refusal = safety_refusal(exc)
            if refusal is None:
                raise
            from langchain_core.messages import AIMessage

            return {"messages": [AIMessage(content=refusal)]}

    async def astream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Stream chunks, yielding a refusal message if a safety error is raised."""
        try:
            async for chunk in self._runnable.astream(input, config, **kwargs):
                yield chunk
        except Exception as exc:
            if not self._outermost:
                raise
            refusal = safety_refusal(exc)
            if refusal is None:
                raise
            from langchain_core.messages import AIMessage

            yield ("messages", (AIMessage(content=refusal), {}))

    async def astream_events(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Any:
        """Stream events, suppressing buffered AI output when a safety block is detected."""
        logger.debug(
            "safety_aware_runnable astream_events called outermost=%s", self._outermost
        )
        try:
            merged_config, safety_ctx = _build_merged_config(config)
            # Buffer AI output chunks so we can replace them with the refusal if blocked.
            # Non-AI events (tool calls, tool results, metadata) stream through immediately.
            ai_chunks: list[Any] = []
            tool_blocked_via_sentinel = False
            active_tool_calls = 0  # tracks in-flight tools at this graph level
            async for event in self._runnable.astream_events(
                input, merged_config, **kwargs
            ):
                event_type = event.get("event", "")

                if self._outermost and event_type == "on_tool_start":
                    active_tool_calls += 1

                if self._outermost and event_type == "on_tool_end":
                    active_tool_calls = max(0, active_tool_calls - 1)
                    output = event.get("data", {}).get("output", "")
                    if _TOOL_SAFETY_REFUSAL in str(output):
                        tool_blocked_via_sentinel = True
                    yield event
                    # Break only when every tool in this batch has finished AND one was
                    # blocked. Other parallel tools run to completion first; the break
                    # fires between the last on_tool_end and the orchestrator's next LLM
                    # call, so no retry is ever dispatched.
                    if tool_blocked_via_sentinel and active_tool_calls == 0:
                        break
                    continue

                if self._outermost and event_type == "on_chat_model_stream":
                    ai_chunks.append(event)
                else:
                    yield event

            # Emit either the consistent refusal or the buffered LLM chunks.
            if self._outermost and (safety_ctx["blocked"] or tool_blocked_via_sentinel):
                from langchain_core.messages import AIMessage

                yield {
                    "event": "on_chat_model_stream",
                    "name": "guardian_refusal",
                    "data": {"chunk": AIMessage(content=_TOOL_SAFETY_REFUSAL)},
                }
            else:
                # Pass buffered AI chunks through unchanged.
                for chunk in ai_chunks:
                    yield chunk
        except Exception as exc:
            if not self._outermost:
                raise
            refusal = safety_refusal(exc)
            if refusal is None:
                raise
            from langchain_core.messages import AIMessage

            yield {
                "event": "on_chat_model_stream",
                "name": "guardian_refusal",
                "data": {"chunk": AIMessage(content=refusal)},
            }
