"""Safety-aware graph and runnable wrappers for Granite Guardian integration.

Shared by the orchestrator graph (graph.py) and subagent construction
(subagents.py) so ContentSafetyError is caught and converted to a clean
refusal message at every execution boundary, including inside subagents
and their skills.
"""

from __future__ import annotations

from typing import Any

from deep_agent.src.agent.repetition import detect_repetition_loop
from deep_agent.src.guardrails import (
    TOOL_SAFETY_REFUSAL as _TOOL_SAFETY_REFUSAL,
)
from deep_agent.src.guardrails import (
    ContentSafetyError,
    InputContentSafetyError,
    ToolContentSafetyError,
)
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_INPUT_SAFETY_REFUSAL = "I can't help with that request due to content safety policy."


def _message_text(content: Any) -> str:
    """Extract plain text from an AIMessage content field.

    Handles both str and Gemini's list-of-parts format
    (e.g. ``[{"type": "text", "text": "..."}]``).
    """
    if isinstance(content, list):
        return "".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return str(content) if content else ""


def safety_refusal(exc: BaseException) -> str | None:
    """Walk the exception chain and return the appropriate refusal message.

    Checks both exception type and string representation at each step since
    ModelRetryMiddleware wraps originals in a fresh exception with the class
    name only in the message string (not in __cause__/__context__).
    Returns None if no safety-related error is found.
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
        # ModelRetryMiddleware embeds the original class name in the wrapper
        # message — check string representation as a fallback.
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
    """Proxy that converts ContentSafetyError to a refusal message.

    Wraps both the orchestrator graph and CompiledSubAgent runnables so
    safety errors raised anywhere — including in skills — produce a
    consistent user-facing message instead of crashing.

    Tool-result safety is handled upstream by GuardianToolProxy.
    This runnable handles input safety errors (via ModelRetryMiddleware).

    outermost=True  (orchestrator): catches all safety exceptions.
    outermost=False (inner subagent): re-raises so the outermost catches it.
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
            # Override with consistent refusal if any tool was safety-blocked.
            # Runs at every level so inner SafetyAwareRunnables also override.
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
            elif settings.REPETITION_LOOP_DETECTION_ENABLED:
                # Truncate degenerate repetition loops (same refusal sentence
                # dozens of times) before re-entering conversation state.
                # Runs at every level — same rationale as the tool-block above.
                for i in range(len(msgs) - 1, -1, -1):
                    if isinstance(msgs[i], AIMessage):
                        text = _message_text(msgs[i].content)
                        is_loop, truncated = detect_repetition_loop(text)
                        if is_loop:
                            logger.warning(
                                "Repetition loop detected in final message; "
                                "truncating (original_chars=%d, truncated_chars=%d)",
                                len(text),
                                len(truncated),
                            )
                            # model_copy preserves tool_calls, response_metadata,
                            # usage_metadata, structured content, and the message's
                            # existing id — only `content` is replaced.
                            msgs[i] = msgs[i].model_copy(update={"content": truncated})
                            result = {
                                **(result if isinstance(result, dict) else {}),
                                "messages": msgs,
                            }
                        break
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
        """Stream events, suppressing AI output when a safety block or repetition loop is detected."""
        logger.debug(
            "safety_aware_runnable astream_events called outermost=%s", self._outermost
        )
        try:
            merged_config, safety_ctx = _build_merged_config(config)
            # Buffer AI chunks; non-AI events stream through immediately.
            ai_chunks: list[Any] = []
            tool_blocked_via_sentinel = False
            active_tool_calls = 0  # tracks in-flight tools at this graph level
            # Accumulated text of the current AI response, used to detect
            # repetition loops early and break before consuming unbounded
            # tokens. Scoped per model invocation (keyed by run_id) so
            # unrelated LLM calls can't dilute detection or combine into
            # a false positive.
            repetition_text = ""
            repetition_run_id: Any = None
            repetition_loop_hit = False
            repetition_truncated_text = ""
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
                    # Break after last tool in batch finishes when blocked.
                    if tool_blocked_via_sentinel and active_tool_calls == 0:
                        break
                    continue

                if self._outermost and event_type == "on_chat_model_stream":
                    ai_chunks.append(event)
                    if settings.REPETITION_LOOP_DETECTION_ENABLED:
                        run_id = event.get("run_id")
                        if run_id != repetition_run_id:
                            repetition_run_id = run_id
                            repetition_text = ""
                        chunk_obj = event.get("data", {}).get("chunk")
                        repetition_text += _message_text(
                            getattr(chunk_obj, "content", "")
                        )
                        is_loop, truncated = detect_repetition_loop(repetition_text)
                        if is_loop:
                            repetition_loop_hit = True
                            repetition_truncated_text = truncated
                            logger.warning(
                                "Repetition loop detected mid-stream; breaking "
                                "generation early (buffered_chars=%d, "
                                "truncated_chars=%d)",
                                len(repetition_text),
                                len(truncated),
                            )
                            break
                else:
                    yield event

            # Emit refusal, truncated loop content, or buffered chunks (priority order).
            if self._outermost and (safety_ctx["blocked"] or tool_blocked_via_sentinel):
                from langchain_core.messages import AIMessage

                yield {
                    "event": "on_chat_model_stream",
                    "name": "guardian_refusal",
                    "data": {"chunk": AIMessage(content=_TOOL_SAFETY_REFUSAL)},
                }
            elif self._outermost and repetition_loop_hit:
                from langchain_core.messages import AIMessage

                yield {
                    "event": "on_chat_model_stream",
                    "name": "repetition_loop_truncated",
                    "data": {"chunk": AIMessage(content=repetition_truncated_text)},
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
