"""Unit tests for deep_agent.aegra.safety."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from deep_agent.aegra.safety import (
    SafetyAwareRunnable,
    _build_merged_config,
    safety_refusal,
)
from deep_agent.src.guardrails import (
    ContentSafetyError,
    InputContentSafetyError,
    ToolContentSafetyError,
)
from deep_agent.src.guardrails import TOOL_SAFETY_REFUSAL as _TOOL_SAFETY_REFUSAL

_INPUT_SAFETY_REFUSAL = "I can't help with that request due to content safety policy."


# ---------------------------------------------------------------------------
# safety_refusal
# ---------------------------------------------------------------------------


class TestSafetyRefusal:
    def test_tool_content_safety_error_returns_tool_refusal(self):
        exc = ToolContentSafetyError("blocked")
        assert safety_refusal(exc) == _TOOL_SAFETY_REFUSAL

    def test_input_content_safety_error_returns_input_refusal(self):
        exc = InputContentSafetyError("blocked")
        assert safety_refusal(exc) == _INPUT_SAFETY_REFUSAL

    def test_content_safety_error_returns_input_refusal(self):
        exc = ContentSafetyError("blocked")
        assert safety_refusal(exc) == _INPUT_SAFETY_REFUSAL

    def test_string_contains_tool_content_safety_error(self):
        exc = RuntimeError("ToolContentSafetyError: some message")
        assert safety_refusal(exc) == _TOOL_SAFETY_REFUSAL

    def test_string_contains_input_content_safety_error(self):
        exc = RuntimeError("InputContentSafetyError: blocked input")
        assert safety_refusal(exc) == _INPUT_SAFETY_REFUSAL

    def test_string_contains_content_safety_error(self):
        exc = RuntimeError("ContentSafetyError occurred")
        assert safety_refusal(exc) == _INPUT_SAFETY_REFUSAL

    def test_non_safety_exception_returns_none(self):
        exc = ValueError("random error")
        assert safety_refusal(exc) is None

    def test_chained_cause_is_safety_error(self):
        cause = ToolContentSafetyError("cause")
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = cause
        assert safety_refusal(wrapper) == _TOOL_SAFETY_REFUSAL

    def test_chained_context_is_safety_error(self):
        ctx = InputContentSafetyError("context")
        wrapper = RuntimeError("wrapper")
        wrapper.__context__ = ctx
        assert safety_refusal(wrapper) == _INPUT_SAFETY_REFUSAL

    def test_cycle_prevention_returns_none(self):
        exc = RuntimeError("no safety")
        exc.__cause__ = exc  # self-referential cycle
        assert safety_refusal(exc) is None


# ---------------------------------------------------------------------------
# _build_merged_config
# ---------------------------------------------------------------------------


class TestBuildMergedConfig:
    def test_none_config_creates_empty_base(self):
        merged, ctx = _build_merged_config(None)
        assert "_safety_ctx" in merged
        assert merged["_safety_ctx"] is ctx
        assert ctx == {"blocked": False}

    def test_existing_config_is_merged(self):
        merged, ctx = _build_merged_config({"run_name": "test"})
        assert merged["run_name"] == "test"
        assert merged["_safety_ctx"] is ctx

    def test_existing_metadata_is_preserved(self):
        merged, ctx = _build_merged_config({"metadata": {"user": "alice"}})
        assert merged["metadata"]["user"] == "alice"
        assert merged["metadata"]["_safety_ctx"] is ctx

    def test_safety_ctx_shared_between_config_and_metadata(self):
        merged, ctx = _build_merged_config({})
        assert merged["_safety_ctx"] is merged["metadata"]["_safety_ctx"]

    def test_safety_ctx_starts_unblocked(self):
        _, ctx = _build_merged_config(None)
        assert ctx["blocked"] is False


# ---------------------------------------------------------------------------
# SafetyAwareRunnable — sync interface
# ---------------------------------------------------------------------------


class TestSafetyAwareRunnableInit:
    def test_stores_runnable_and_outermost(self):
        inner = MagicMock()
        sar = SafetyAwareRunnable(inner, outermost=True)
        assert sar._runnable is inner
        assert sar._outermost is True

    def test_default_outermost_is_false(self):
        sar = SafetyAwareRunnable(MagicMock())
        assert sar._outermost is False

    def test_getattr_delegates_to_inner(self):
        inner = MagicMock()
        inner.some_attr = "value"
        sar = SafetyAwareRunnable(inner)
        assert sar.some_attr == "value"

    def test_copy_wraps_inner_copy(self):
        inner = MagicMock()
        inner.copy.return_value = MagicMock()
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = sar.copy(update={})
        assert isinstance(result, SafetyAwareRunnable)
        assert result._outermost is True
        inner.copy.assert_called_once_with(update={})

    def test_with_config_none_uses_kwargs_only(self):
        inner = MagicMock()
        inner.with_config.return_value = MagicMock()
        sar = SafetyAwareRunnable(inner)
        result = sar.with_config(None, tags=["x"])
        inner.with_config.assert_called_once_with(tags=["x"])
        assert isinstance(result, SafetyAwareRunnable)

    def test_with_config_non_none_passes_config(self):
        inner = MagicMock()
        inner.with_config.return_value = MagicMock()
        sar = SafetyAwareRunnable(inner)
        cfg = {"run_name": "r"}
        result = sar.with_config(cfg, tags=["x"])
        inner.with_config.assert_called_once_with(cfg, tags=["x"])
        assert isinstance(result, SafetyAwareRunnable)


# ---------------------------------------------------------------------------
# SafetyAwareRunnable.ainvoke
# ---------------------------------------------------------------------------


class TestSafetyAwareRunnableAinvoke:
    @pytest.mark.asyncio
    async def test_safe_result_returned_unchanged(self):
        ai = AIMessage(content="hello")
        result = {"messages": [ai]}
        inner = MagicMock()
        inner.ainvoke = AsyncMock(return_value=result)
        sar = SafetyAwareRunnable(inner)
        out = await sar.ainvoke({"input": "hi"})
        assert out["messages"][0].content == "hello"

    @pytest.mark.asyncio
    async def test_tool_blocked_via_safety_ctx_rewrites_last_ai_message(self):
        ai = AIMessage(content="original response")
        tm = ToolMessage(content="tool output", name="t", tool_call_id="c1")

        async def fake_ainvoke(input, config, **kwargs):
            # Simulate GuardianToolProxy setting blocked=True in safety_ctx
            config["_safety_ctx"]["blocked"] = True
            return {"messages": [tm, ai]}

        inner = MagicMock()
        inner.ainvoke = fake_ainvoke
        sar = SafetyAwareRunnable(inner)
        out = await sar.ainvoke({})
        last_ai = next(m for m in reversed(out["messages"]) if isinstance(m, AIMessage))
        assert last_ai.content == _TOOL_SAFETY_REFUSAL

    @pytest.mark.asyncio
    async def test_tool_blocked_via_sentinel_in_tool_message(self):
        ai = AIMessage(content="should be replaced")
        tm = ToolMessage(
            content=f"...{_TOOL_SAFETY_REFUSAL}...", name="t", tool_call_id="c1"
        )
        inner = MagicMock()
        inner.ainvoke = AsyncMock(return_value={"messages": [tm, ai]})
        sar = SafetyAwareRunnable(inner)
        out = await sar.ainvoke({})
        last_ai = next(m for m in reversed(out["messages"]) if isinstance(m, AIMessage))
        assert last_ai.content == _TOOL_SAFETY_REFUSAL

    @pytest.mark.asyncio
    async def test_non_outermost_reraises_exception(self):
        inner = MagicMock()
        inner.ainvoke = AsyncMock(side_effect=InputContentSafetyError("blocked"))
        sar = SafetyAwareRunnable(inner, outermost=False)
        with pytest.raises(InputContentSafetyError):
            await sar.ainvoke({})

    @pytest.mark.asyncio
    async def test_outermost_safety_exception_returns_refusal(self):
        inner = MagicMock()
        inner.ainvoke = AsyncMock(side_effect=InputContentSafetyError("blocked"))
        sar = SafetyAwareRunnable(inner, outermost=True)
        out = await sar.ainvoke({})
        assert isinstance(out["messages"][0], AIMessage)
        assert out["messages"][0].content == _INPUT_SAFETY_REFUSAL

    @pytest.mark.asyncio
    async def test_outermost_non_safety_exception_reraises(self):
        inner = MagicMock()
        inner.ainvoke = AsyncMock(side_effect=RuntimeError("unexpected"))
        sar = SafetyAwareRunnable(inner, outermost=True)
        with pytest.raises(RuntimeError, match="unexpected"):
            await sar.ainvoke({})

    @pytest.mark.asyncio
    async def test_outermost_tool_safety_exception_returns_tool_refusal(self):
        inner = MagicMock()
        inner.ainvoke = AsyncMock(side_effect=ToolContentSafetyError("tool blocked"))
        sar = SafetyAwareRunnable(inner, outermost=True)
        out = await sar.ainvoke({})
        assert out["messages"][0].content == _TOOL_SAFETY_REFUSAL

    @pytest.mark.asyncio
    async def test_non_dict_result_is_returned_without_rewrite(self):
        inner = MagicMock()
        inner.ainvoke = AsyncMock(return_value="plain string result")
        sar = SafetyAwareRunnable(inner)
        out = await sar.ainvoke({})
        assert out == "plain string result"


# ---------------------------------------------------------------------------
# SafetyAwareRunnable.astream
# ---------------------------------------------------------------------------


async def _collect(agen):
    items = []
    async for item in agen:
        items.append(item)
    return items


class TestSafetyAwareRunnableAstream:
    @pytest.mark.asyncio
    async def test_yields_chunks_normally(self):
        chunks = [{"event": "chunk", "data": i} for i in range(3)]

        async def gen(*a, **kw):
            for c in chunks:
                yield c

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner)
        result = await _collect(sar.astream({}))
        assert result == chunks

    @pytest.mark.asyncio
    async def test_non_outermost_reraises_on_stream_exception(self):
        async def gen(*a, **kw):
            yield {"data": 1}
            raise InputContentSafetyError("blocked")

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner, outermost=False)
        with pytest.raises(InputContentSafetyError):
            await _collect(sar.astream({}))

    @pytest.mark.asyncio
    async def test_outermost_yields_refusal_on_safety_exception(self):
        async def gen(*a, **kw):
            yield {"data": 1}
            raise InputContentSafetyError("blocked")

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream({}))
        assert len(result) == 2
        event_type, (ai_msg, _) = result[-1]
        assert event_type == "messages"
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.content == _INPUT_SAFETY_REFUSAL

    @pytest.mark.asyncio
    async def test_outermost_reraises_non_safety_stream_exception(self):
        async def gen(*a, **kw):
            yield {"data": 1}
            raise RuntimeError("crash")

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        with pytest.raises(RuntimeError, match="crash"):
            await _collect(sar.astream({}))


# ---------------------------------------------------------------------------
# SafetyAwareRunnable.astream_events
# ---------------------------------------------------------------------------


class TestSafetyAwareRunnableAstreamEvents:
    @pytest.mark.asyncio
    async def test_non_ai_events_pass_through_immediately(self):
        events = [
            {"event": "on_tool_start", "data": {}},
            {"event": "on_tool_end", "data": {"output": "ok"}},
            {"event": "on_chain_end", "data": {}},
        ]

        async def gen(*a, **kw):
            for e in events:
                yield e

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))
        # tool_start and chain_end passed through; tool_end also yielded
        assert any(e.get("event") == "on_tool_start" for e in result)
        assert any(e.get("event") == "on_chain_end" for e in result)

    @pytest.mark.asyncio
    async def test_ai_chunks_buffered_and_flushed_when_safe(self):
        chunk_event = {"event": "on_chat_model_stream", "data": {"chunk": "hi"}}
        other_event = {"event": "on_chain_end", "data": {}}

        async def gen(*a, **kw):
            yield chunk_event
            yield other_event

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))
        # chunk should be flushed at end (safe path)
        assert chunk_event in result

    @pytest.mark.asyncio
    async def test_blocked_tool_via_sentinel_emits_refusal_event(self):
        async def gen(*a, **kw):
            config = a[1] if len(a) > 1 else kw.get("config", {})
            yield {"event": "on_tool_start", "data": {}}
            yield {
                "event": "on_tool_end",
                "data": {"output": f"prefix {_TOOL_SAFETY_REFUSAL} suffix"},
            }
            # This should not be yielded — loop breaks after tool batch completes
            yield {"event": "on_chat_model_stream", "data": {"chunk": "dropped"}}

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))
        refusal_events = [e for e in result if e.get("name") == "guardian_refusal"]
        assert len(refusal_events) == 1
        assert isinstance(refusal_events[0]["data"]["chunk"], AIMessage)

    @pytest.mark.asyncio
    async def test_non_outermost_does_not_track_tool_calls(self):
        chunk = {"event": "on_chat_model_stream", "data": {"chunk": "x"}}

        async def gen(*a, **kw):
            yield chunk

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=False)
        result = await _collect(sar.astream_events({}))
        assert chunk in result

    @pytest.mark.asyncio
    async def test_outermost_safety_exception_yields_refusal_event(self):
        async def gen(*a, **kw):
            raise InputContentSafetyError("blocked")
            yield  # noqa: unreachable — makes this an async generator

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))
        assert len(result) == 1
        assert result[0]["name"] == "guardian_refusal"

    @pytest.mark.asyncio
    async def test_outermost_non_safety_exception_reraises(self):
        async def gen(*a, **kw):
            raise RuntimeError("crash")
            yield  # noqa: unreachable

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        with pytest.raises(RuntimeError, match="crash"):
            await _collect(sar.astream_events({}))

    @pytest.mark.asyncio
    async def test_blocked_via_safety_ctx_emits_refusal_instead_of_ai_chunks(self):
        chunk_event = {"event": "on_chat_model_stream", "data": {"chunk": "response"}}

        async def gen(*a, **kw):
            config = a[1]
            config["_safety_ctx"]["blocked"] = True
            yield chunk_event

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))
        refusal_events = [e for e in result if e.get("name") == "guardian_refusal"]
        assert len(refusal_events) == 1
        assert chunk_event not in result
