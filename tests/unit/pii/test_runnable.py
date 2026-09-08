"""Unit tests for PII-aware runnable wrapper — token restoration across streams."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from langchain_core.messages import AIMessageChunk

from deep_agent.src.pii.runnable import (
    PIIAwareRunnable,
    _preview,
    _restore_args,
    _restore_content,
    _restore_message_stream_data,
    _restore_text,
)


# ── Helpers ──────────────────────────────────────────────────────────────


async def _async_iter(items):
    """Yield items from a list as an async iterator."""
    for item in items:
        yield item


def _make_chunk(content: Any = "", tool_calls: list | None = None) -> AIMessageChunk:
    """Build an AIMessageChunk with optional tool_calls."""
    if tool_calls:
        tool_call_chunks = [
            {
                "name": tc.get("name"),
                "args": json.dumps(tc.get("args", {})),
                "id": tc.get("id"),
                "index": i,
            }
            for i, tc in enumerate(tool_calls)
        ]
        return AIMessageChunk(content=content, tool_call_chunks=tool_call_chunks)
    return AIMessageChunk(content=content)


# ═══════════════════════════════════════════════════════════════════════
# Free-function tests
# ═══════════════════════════════════════════════════════════════════════


class TestPreview:
    """Tests for _preview()."""

    def test_truncates_long_string(self):
        long = "a" * 200
        assert _preview(long, max_len=10) == "a" * 10

    def test_short_string_unchanged(self):
        assert _preview("hello") == "hello"

    def test_non_string_uses_repr(self):
        assert _preview(42) == "42"
        assert _preview([1, 2]) == "[1, 2]"

    def test_non_string_truncated(self):
        result = _preview(list(range(200)), max_len=20)
        assert len(result) == 20

    def test_default_max_len_is_120(self):
        long = "x" * 200
        assert len(_preview(long)) == 120


class TestRestoreText:
    """Tests for _restore_text()."""

    def test_replaces_tokens(self):
        container = {"[EMAIL_1]": "alice@example.com", "[PHONE_1]": "555-0100"}
        text = "Contact [EMAIL_1] or [PHONE_1]"
        assert _restore_text(text, container) == "Contact alice@example.com or 555-0100"

    def test_multiple_occurrences(self):
        container = {"[NAME_1]": "Alice"}
        assert _restore_text("[NAME_1] met [NAME_1]", container) == "Alice met Alice"

    def test_non_string_returns_as_is(self):
        assert _restore_text(42, {"[X]": "y"}) == 42
        assert _restore_text(None, {"[X]": "y"}) is None

    def test_empty_container_returns_original(self):
        assert _restore_text("hello [EMAIL_1]", {}) == "hello [EMAIL_1]"

    def test_no_matching_tokens(self):
        assert _restore_text("no tokens here", {"[EMAIL_1]": "x"}) == "no tokens here"


class TestRestoreContent:
    """Tests for _restore_content()."""

    def test_restores_string_content(self):
        container = {"[EMAIL_1]": "a@b.com"}
        assert _restore_content("hi [EMAIL_1]", container) == "hi a@b.com"

    def test_restores_list_of_text_blocks(self):
        container = {"[NAME_1]": "Alice"}
        blocks = [
            {"type": "text", "text": "Hello [NAME_1]"},
            {"type": "text", "text": "[NAME_1] here"},
        ]
        result = _restore_content(blocks, container)
        assert result == [
            {"type": "text", "text": "Hello Alice"},
            {"type": "text", "text": "Alice here"},
        ]

    def test_non_text_blocks_pass_through(self):
        container = {"[X]": "y"}
        blocks = [
            {"type": "image", "url": "[X]"},
            {"type": "text", "text": "[X]"},
        ]
        result = _restore_content(blocks, container)
        assert result[0] == {"type": "image", "url": "[X]"}
        assert result[1] == {"type": "text", "text": "y"}

    def test_non_string_non_list_passes_through(self):
        assert _restore_content(42, {"[X]": "y"}) == 42
        assert _restore_content(None, {"[X]": "y"}) is None

    def test_empty_text_in_block(self):
        container = {"[X]": "y"}
        blocks = [{"type": "text"}]
        result = _restore_content(blocks, container)
        assert result == [{"type": "text", "text": ""}]


class TestRestoreArgs:
    """Tests for _restore_args()."""

    def test_restores_dict_values_recursively(self):
        container = {"[EMAIL_1]": "a@b.com"}
        args = {"to": "[EMAIL_1]", "nested": {"cc": "[EMAIL_1]"}}
        result = _restore_args(args, container)
        assert result == {"to": "a@b.com", "nested": {"cc": "a@b.com"}}

    def test_restores_list_items_recursively(self):
        container = {"[X]": "y"}
        assert _restore_args(["[X]", "[X]"], container) == ["y", "y"]

    def test_restores_string_values(self):
        container = {"[X]": "y"}
        assert _restore_args("[X]", container) == "y"

    def test_non_matching_types_pass_through(self):
        assert _restore_args(42, {"[X]": "y"}) == 42
        assert _restore_args(3.14, {"[X]": "y"}) == 3.14
        assert _restore_args(True, {}) is True

    def test_deeply_nested(self):
        container = {"[T]": "v"}
        args = {"a": [{"b": "[T]"}]}
        assert _restore_args(args, container) == {"a": [{"b": "v"}]}


class TestRestoreMessageStreamData:
    """Tests for _restore_message_stream_data()."""

    def test_restores_tuple(self):
        container = {"[EMAIL_1]": "a@b.com"}
        chunk = _make_chunk(content="hi [EMAIL_1]")
        data = (chunk, {"some": "meta"})
        result = _restore_message_stream_data(data, container)
        assert isinstance(result, tuple)
        assert result[0].content == "hi a@b.com"
        assert result[1] == {"some": "meta"}

    def test_restores_list(self):
        container = {"[EMAIL_1]": "a@b.com"}
        chunk = _make_chunk(content="hi [EMAIL_1]")
        data = [chunk, {"some": "meta"}]
        result = _restore_message_stream_data(data, container)
        assert isinstance(result, list)
        assert result[0].content == "hi a@b.com"

    def test_empty_container_returns_original(self):
        chunk = _make_chunk(content="hi [EMAIL_1]")
        data = (chunk, {})
        result = _restore_message_stream_data(data, {})
        assert result is data

    def test_chunk_without_content(self):
        container = {"[X]": "y"}
        data = ("not_a_chunk", {"meta": 1})
        result = _restore_message_stream_data(data, container)
        assert result[0] == "not_a_chunk"

    def test_chunk_with_tool_calls(self):
        container = {"[ARG]": "real_value"}
        tool_calls = [{"name": "my_tool", "args": {"key": "[ARG]"}, "id": "tc1"}]
        chunk = _make_chunk(content="", tool_calls=tool_calls)
        data = (chunk, {})
        result = _restore_message_stream_data(data, container)
        restored_chunk = result[0]
        assert restored_chunk.tool_calls[0]["args"]["key"] == "real_value"

    def test_fallback_for_non_pair_data(self):
        container = {"[X]": "y"}
        data = "[X] plain string"
        result = _restore_message_stream_data(data, container)
        assert result == "y plain string"

    def test_preserves_unchanged_content(self):
        container = {"[MISS]": "val"}
        chunk = _make_chunk(content="nothing to replace")
        data = (chunk, {})
        result = _restore_message_stream_data(data, container)
        assert result[0].content == "nothing to replace"


# ═══════════════════════════════════════════════════════════════════════
# PIIAwareRunnable — wrappers & delegation
# ═══════════════════════════════════════════════════════════════════════


class TestPIIAwareRunnable:
    """Tests for __init__, __getattr__, copy, with_config."""

    def test_getattr_delegates(self):
        inner = MagicMock()
        inner.some_attr = "hello"
        wrapper = PIIAwareRunnable(inner)
        assert wrapper.some_attr == "hello"

    def test_copy_returns_pii_aware_runnable(self):
        inner = MagicMock()
        inner.copy.return_value = MagicMock(name="copied_inner")
        wrapper = PIIAwareRunnable(inner)
        result = wrapper.copy(update={"x": 1})
        assert isinstance(result, PIIAwareRunnable)
        inner.copy.assert_called_once_with(update={"x": 1})

    def test_with_config_returns_pii_aware_runnable(self):
        inner = MagicMock()
        inner.with_config.return_value = MagicMock(name="configured_inner")
        wrapper = PIIAwareRunnable(inner)
        result = wrapper.with_config({"key": "val"}, tags=["a"])
        assert isinstance(result, PIIAwareRunnable)
        inner.with_config.assert_called_once_with({"key": "val"}, tags=["a"])

    def test_with_config_no_config_arg(self):
        inner = MagicMock()
        inner.with_config.return_value = MagicMock()
        wrapper = PIIAwareRunnable(inner)
        result = wrapper.with_config(tags=["b"])
        assert isinstance(result, PIIAwareRunnable)
        inner.with_config.assert_called_once_with(tags=["b"])

    def test_with_config_none_config(self):
        inner = MagicMock()
        inner.with_config.return_value = MagicMock()
        wrapper = PIIAwareRunnable(inner)
        result = wrapper.with_config(None, tags=["b"])
        assert isinstance(result, PIIAwareRunnable)
        inner.with_config.assert_called_once_with(tags=["b"])


# ═══════════════════════════════════════════════════════════════════════
# _setup_container / _clear_container
# ═══════════════════════════════════════════════════════════════════════


class TestSetupContainer:
    """Tests for _setup_container()."""

    def test_creates_new_container(self):
        mock_scrubber = MagicMock()
        mock_scrubber._get_shared_container.return_value = None
        with patch("deep_agent.src.pii.get_scrubber", return_value=mock_scrubber):
            wrapper = PIIAwareRunnable(MagicMock())
            container, owned = wrapper._setup_container()
            assert owned is True
            assert container == {}
            mock_scrubber.set_shared_container.assert_called_once_with(container)

    def test_reuses_existing_container(self):
        existing = {"[EMAIL_1]": "a@b.com"}
        mock_scrubber = MagicMock()
        mock_scrubber._get_shared_container.return_value = existing
        with patch("deep_agent.src.pii.get_scrubber", return_value=mock_scrubber):
            wrapper = PIIAwareRunnable(MagicMock())
            container, owned = wrapper._setup_container()
            assert owned is False
            assert container is existing
            mock_scrubber.set_shared_container.assert_not_called()

    def test_returns_empty_when_scrubber_not_available(self):
        with patch("deep_agent.src.pii.get_scrubber", return_value=None):
            wrapper = PIIAwareRunnable(MagicMock())
            container, owned = wrapper._setup_container()
            assert container == {}
            assert owned is False

    def test_returns_empty_on_import_error(self):
        with patch(
            "deep_agent.src.pii.get_scrubber",
            side_effect=ImportError("no module"),
        ):
            wrapper = PIIAwareRunnable(MagicMock())
            container, owned = wrapper._setup_container()
            assert container == {}
            assert owned is False


class TestClearContainer:
    """Tests for _clear_container()."""

    def test_clears_when_owned(self):
        mock_scrubber = MagicMock()
        mock_scrubber._instance_container = {"[X]": "y"}
        with patch("deep_agent.src.pii.get_scrubber", return_value=mock_scrubber):
            wrapper = PIIAwareRunnable(MagicMock())
            wrapper._clear_container(owned=True)
            assert mock_scrubber._instance_container is None

    def test_does_nothing_when_not_owned(self):
        mock_scrubber = MagicMock()
        mock_scrubber._instance_container = {"[X]": "y"}
        with patch("deep_agent.src.pii.get_scrubber", return_value=mock_scrubber):
            wrapper = PIIAwareRunnable(MagicMock())
            wrapper._clear_container(owned=False)
            assert mock_scrubber._instance_container == {"[X]": "y"}

    def test_no_error_when_scrubber_missing(self):
        with patch("deep_agent.src.pii.get_scrubber", return_value=None):
            wrapper = PIIAwareRunnable(MagicMock())
            wrapper._clear_container(owned=True)

    def test_no_error_when_instance_container_not_set(self):
        mock_scrubber = MagicMock(spec=[])
        with patch("deep_agent.src.pii.get_scrubber", return_value=mock_scrubber):
            wrapper = PIIAwareRunnable(MagicMock())
            wrapper._clear_container(owned=True)


# ═══════════════════════════════════════════════════════════════════════
# _assemble_chunk / _restore_chunk / _flush_run
# ═══════════════════════════════════════════════════════════════════════


class TestAssembleChunk:
    """Tests for _assemble_chunk()."""

    def test_sums_multiple_chunks(self):
        wrapper = PIIAwareRunnable(MagicMock())
        events = [
            {"data": {"chunk": AIMessageChunk(content="Hello ")}},
            {"data": {"chunk": AIMessageChunk(content="world")}},
        ]
        result = wrapper._assemble_chunk(events)
        assert result.content == "Hello world"

    def test_returns_none_for_empty_events(self):
        wrapper = PIIAwareRunnable(MagicMock())
        assert wrapper._assemble_chunk([]) is None

    def test_handles_events_with_missing_chunk(self):
        wrapper = PIIAwareRunnable(MagicMock())
        events = [
            {"data": {}},
            {"data": {"chunk": AIMessageChunk(content="only")}},
            {"data": {"other": "stuff"}},
        ]
        result = wrapper._assemble_chunk(events)
        assert result.content == "only"

    def test_single_event(self):
        wrapper = PIIAwareRunnable(MagicMock())
        events = [{"data": {"chunk": AIMessageChunk(content="solo")}}]
        result = wrapper._assemble_chunk(events)
        assert result.content == "solo"

    def test_all_events_missing_chunk(self):
        wrapper = PIIAwareRunnable(MagicMock())
        events = [{"data": {}}, {"data": {"other": 1}}]
        assert wrapper._assemble_chunk(events) is None


class TestRestoreChunk:
    """Tests for _restore_chunk()."""

    def test_restores_content(self):
        wrapper = PIIAwareRunnable(MagicMock())
        container = {"[EMAIL_1]": "a@b.com"}
        chunk = AIMessageChunk(content="hi [EMAIL_1]")
        result = wrapper._restore_chunk(chunk, container)
        assert result.content == "hi a@b.com"
        assert not result.tool_calls

    def test_restores_tool_calls(self):
        wrapper = PIIAwareRunnable(MagicMock())
        container = {"[ARG]": "real_val"}
        tc = [{"name": "t", "args": {"k": "[ARG]"}, "id": "tc1"}]
        chunk = _make_chunk(content="", tool_calls=tc)
        result = wrapper._restore_chunk(chunk, container)
        assert result.tool_call_chunks
        parsed = json.loads(result.tool_call_chunks[0]["args"])
        assert parsed["k"] == "real_val"

    def test_chunk_without_tool_calls(self):
        wrapper = PIIAwareRunnable(MagicMock())
        chunk = AIMessageChunk(content="plain")
        result = wrapper._restore_chunk(chunk, {})
        assert result.content == "plain"


class TestFlushRun:
    """Tests for _flush_run()."""

    @pytest.mark.asyncio
    async def test_empty_events_yields_nothing(self):
        wrapper = PIIAwareRunnable(MagicMock())
        collected = []
        async for event in wrapper._flush_run("run1", [], {"[X]": "y"}):
            collected.append(event)
        assert collected == []

    @pytest.mark.asyncio
    async def test_assembled_and_restored(self):
        wrapper = PIIAwareRunnable(MagicMock())
        container = {"[EMAIL_1]": "a@b.com"}
        events = [
            {
                "event": "on_chat_model_stream",
                "run_id": "r1",
                "data": {"chunk": AIMessageChunk(content="hi [EMAIL_1]")},
            },
        ]
        collected = []
        async for event in wrapper._flush_run("r1", events, container):
            collected.append(event)
        assert len(collected) == 1
        assert collected[0]["data"]["chunk"].content == "hi a@b.com"

    @pytest.mark.asyncio
    async def test_no_container_passes_through(self):
        wrapper = PIIAwareRunnable(MagicMock())
        events = [
            {
                "event": "on_chat_model_stream",
                "run_id": "r1",
                "data": {"chunk": AIMessageChunk(content="text")},
            },
        ]
        collected = []
        async for event in wrapper._flush_run("r1", events, {}):
            collected.append(event)
        assert len(collected) == 1
        assert collected[0] is events[0]

    @pytest.mark.asyncio
    async def test_none_assembled_passes_through(self):
        wrapper = PIIAwareRunnable(MagicMock())
        events = [{"data": {}}]
        collected = []
        async for event in wrapper._flush_run("r1", events, {"[X]": "y"}):
            collected.append(event)
        assert len(collected) == 1

    @pytest.mark.asyncio
    async def test_multiple_chunks_assembled(self):
        wrapper = PIIAwareRunnable(MagicMock())
        container = {"[NAME_1]": "Alice"}
        events = [
            {
                "event": "on_chat_model_stream",
                "run_id": "r1",
                "data": {"chunk": AIMessageChunk(content="Hello ")},
            },
            {
                "event": "on_chat_model_stream",
                "run_id": "r1",
                "data": {"chunk": AIMessageChunk(content="[NAME_1]!")},
            },
        ]
        collected = []
        async for event in wrapper._flush_run("r1", events, container):
            collected.append(event)
        assert len(collected) == 1
        assert collected[0]["data"]["chunk"].content == "Hello Alice!"


# ═══════════════════════════════════════════════════════════════════════
# _restore_tool_event_args / _restore_tool_event_output
# ═══════════════════════════════════════════════════════════════════════


class TestRestoreToolEventArgs:
    """Tests for _restore_tool_event_args()."""

    def test_restores_args_in_input(self):
        wrapper = PIIAwareRunnable(MagicMock())
        container = {"[EMAIL_1]": "a@b.com"}
        event = {
            "event": "on_tool_start",
            "data": {"input": {"args": {"to": "[EMAIL_1]"}}},
        }
        result = wrapper._restore_tool_event_args(event, container)
        assert result["data"]["input"]["args"]["to"] == "a@b.com"

    def test_empty_container_returns_unchanged(self):
        wrapper = PIIAwareRunnable(MagicMock())
        event = {"event": "on_tool_start", "data": {"input": {"args": {"k": "v"}}}}
        result = wrapper._restore_tool_event_args(event, {})
        assert result is event

    def test_input_without_args_key(self):
        wrapper = PIIAwareRunnable(MagicMock())
        container = {"[EMAIL_1]": "a@b.com"}
        event = {
            "event": "on_tool_start",
            "data": {"input": {"query": "[EMAIL_1]"}},
        }
        result = wrapper._restore_tool_event_args(event, container)
        assert result["data"]["input"]["query"] == "a@b.com"

    def test_non_dict_input(self):
        wrapper = PIIAwareRunnable(MagicMock())
        event = {"event": "on_tool_start", "data": {"input": "plain_string"}}
        result = wrapper._restore_tool_event_args(event, {"[X]": "y"})
        assert result is event

    def test_no_data_key(self):
        wrapper = PIIAwareRunnable(MagicMock())
        event = {"event": "on_tool_start"}
        result = wrapper._restore_tool_event_args(event, {"[X]": "y"})
        assert result is event


class TestRestoreToolEventOutput:
    """Tests for _restore_tool_event_output()."""

    def test_restores_output(self):
        wrapper = PIIAwareRunnable(MagicMock())
        container = {"[EMAIL_1]": "a@b.com"}
        event = {
            "event": "on_tool_end",
            "data": {"output": "Result: [EMAIL_1]"},
        }
        result = wrapper._restore_tool_event_output(event, container)
        assert result["data"]["output"] == "Result: a@b.com"

    def test_empty_container_returns_unchanged(self):
        wrapper = PIIAwareRunnable(MagicMock())
        event = {"event": "on_tool_end", "data": {"output": "val"}}
        result = wrapper._restore_tool_event_output(event, {})
        assert result is event

    def test_no_output_returns_unchanged(self):
        wrapper = PIIAwareRunnable(MagicMock())
        event = {"event": "on_tool_end", "data": {}}
        result = wrapper._restore_tool_event_output(event, {"[X]": "y"})
        assert result is event

    def test_dict_output_restored_recursively(self):
        wrapper = PIIAwareRunnable(MagicMock())
        container = {"[NAME_1]": "Alice"}
        event = {
            "event": "on_tool_end",
            "data": {"output": {"name": "[NAME_1]", "nested": {"val": "[NAME_1]"}}},
        }
        result = wrapper._restore_tool_event_output(event, container)
        assert result["data"]["output"]["name"] == "Alice"
        assert result["data"]["output"]["nested"]["val"] == "Alice"

    def test_none_output_returns_unchanged(self):
        wrapper = PIIAwareRunnable(MagicMock())
        event = {"event": "on_tool_end", "data": {"output": None}}
        result = wrapper._restore_tool_event_output(event, {"[X]": "y"})
        assert result is event


# ═══════════════════════════════════════════════════════════════════════
# Async interface — ainvoke / astream / astream_events
# ═══════════════════════════════════════════════════════════════════════


class TestAinvoke:
    """Tests for ainvoke()."""

    @pytest.mark.asyncio
    async def test_calls_inner_ainvoke(self):
        inner = MagicMock()
        inner.ainvoke = AsyncMock(return_value="result")
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=({}, True)):
            with patch.object(wrapper, "_clear_container") as mock_clear:
                result = await wrapper.ainvoke({"input": "val"}, {"cfg": True})
                inner.ainvoke.assert_called_once_with({"input": "val"}, {"cfg": True})
                assert result == "result"
                mock_clear.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_clears_container_on_exception(self):
        inner = MagicMock()
        inner.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=({}, True)):
            with patch.object(wrapper, "_clear_container") as mock_clear:
                with pytest.raises(RuntimeError, match="boom"):
                    await wrapper.ainvoke("input")
                mock_clear.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_passes_kwargs_through(self):
        inner = MagicMock()
        inner.ainvoke = AsyncMock(return_value="ok")
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=({}, False)):
            with patch.object(wrapper, "_clear_container"):
                await wrapper.ainvoke("in", None, stream_mode="messages")
                inner.ainvoke.assert_called_once_with(
                    "in", None, stream_mode="messages"
                )


class TestAstream:
    """Tests for astream()."""

    @pytest.mark.asyncio
    async def test_streams_plain_chunks(self):
        inner = MagicMock()
        inner.astream = MagicMock(return_value=_async_iter(["chunk1", "chunk2"]))
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=({}, True)):
            with patch.object(wrapper, "_clear_container"):
                collected = []
                async for chunk in wrapper.astream("input"):
                    collected.append(chunk)
                assert collected == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_restores_messages_mode_tuples(self):
        container = {"[EMAIL_1]": "a@b.com"}
        chunk = _make_chunk(content="hi [EMAIL_1]")
        items = [("messages", [chunk, {"meta": True}])]

        inner = MagicMock()
        inner.astream = MagicMock(return_value=_async_iter(items))
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=(container, True)):
            with patch.object(wrapper, "_clear_container"):
                collected = []
                async for item in wrapper.astream("input"):
                    collected.append(item)
                assert len(collected) == 1
                mode, data = collected[0]
                assert mode == "messages"
                assert data[0].content == "hi a@b.com"

    @pytest.mark.asyncio
    async def test_non_messages_mode_tuples_pass_through(self):
        items = [("updates", {"state": "val"})]
        inner = MagicMock()
        inner.astream = MagicMock(return_value=_async_iter(items))
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=({}, True)):
            with patch.object(wrapper, "_clear_container"):
                collected = []
                async for item in wrapper.astream("input"):
                    collected.append(item)
                assert collected == [("updates", {"state": "val"})]

    @pytest.mark.asyncio
    async def test_clears_container_after_stream(self):
        inner = MagicMock()
        inner.astream = MagicMock(return_value=_async_iter([]))
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=({}, True)):
            with patch.object(wrapper, "_clear_container") as mock_clear:
                async for _ in wrapper.astream("input"):
                    pass
                mock_clear.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_uses_thread_token_map_when_available(self):
        container = {}
        thread_maps = {"t1": {"[EMAIL_1]": "a@b.com"}}
        chunk = _make_chunk(content="hi [EMAIL_1]")
        items = [("messages", [chunk, {}])]

        inner = MagicMock()
        inner.astream = MagicMock(return_value=_async_iter(items))
        wrapper = PIIAwareRunnable(inner)
        config = {"configurable": {"thread_id": "t1"}}
        with patch.object(wrapper, "_setup_container", return_value=(container, True)):
            with patch.object(wrapper, "_clear_container"):
                with patch(
                    "deep_agent.src.pii.runnable._thread_token_maps",
                    thread_maps,
                    create=True,
                ):
                    with patch(
                        "deep_agent.src.pii.scrubber._thread_token_maps",
                        thread_maps,
                    ):
                        collected = []
                        async for item in wrapper.astream("input", config):
                            collected.append(item)
                        mode, data = collected[0]
                        assert data[0].content == "hi a@b.com"


class TestAstreamEvents:
    """Tests for astream_events()."""

    @pytest.mark.asyncio
    async def test_buffers_chat_model_stream_events(self):
        """Stream events are buffered until on_chat_model_end, then flushed together."""
        container = {"[EMAIL_1]": "a@b.com"}
        events = [
            {
                "event": "on_chat_model_stream",
                "run_id": "r1",
                "data": {"chunk": AIMessageChunk(content="hello [EMAIL_1]")},
            },
            {"event": "on_chat_model_end", "run_id": "r1", "data": {}},
        ]
        inner = MagicMock()
        inner.astream_events = MagicMock(return_value=_async_iter(events))
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=(container, True)):
            with patch.object(wrapper, "_clear_container"):
                collected = []
                async for event in wrapper.astream_events("input"):
                    collected.append(event)
                chunks = [e for e in collected if e.get("data", {}).get("chunk")]
                assert len(chunks) == 1
                assert chunks[0]["data"]["chunk"].content == "hello a@b.com"
                assert any(e.get("event") == "on_chat_model_end" for e in collected)

    @pytest.mark.asyncio
    async def test_flushes_on_chat_model_end(self):
        container = {"[EMAIL_1]": "a@b.com"}
        events = [
            {
                "event": "on_chat_model_stream",
                "run_id": "r1",
                "data": {"chunk": AIMessageChunk(content="hi [EMAIL_1]")},
            },
            {"event": "on_chat_model_end", "run_id": "r1", "data": {}},
        ]
        inner = MagicMock()
        inner.astream_events = MagicMock(return_value=_async_iter(events))
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=(container, True)):
            with patch.object(wrapper, "_clear_container"):
                collected = []
                async for event in wrapper.astream_events("input"):
                    collected.append(event)
                restored = [e for e in collected if e.get("data", {}).get("chunk")]
                assert len(restored) == 1
                assert restored[0]["data"]["chunk"].content == "hi a@b.com"

    @pytest.mark.asyncio
    async def test_restores_tool_start_events(self):
        container = {"[EMAIL_1]": "a@b.com"}
        events = [
            {
                "event": "on_tool_start",
                "run_id": "r1",
                "data": {"input": {"args": {"to": "[EMAIL_1]"}}},
            },
        ]
        inner = MagicMock()
        inner.astream_events = MagicMock(return_value=_async_iter(events))
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=(container, True)):
            with patch.object(wrapper, "_clear_container"):
                collected = []
                async for event in wrapper.astream_events("input"):
                    collected.append(event)
                assert collected[0]["data"]["input"]["args"]["to"] == "a@b.com"

    @pytest.mark.asyncio
    async def test_restores_tool_end_events(self):
        container = {"[EMAIL_1]": "a@b.com"}
        events = [
            {
                "event": "on_tool_end",
                "run_id": "r1",
                "data": {"output": "Result: [EMAIL_1]"},
            },
        ]
        inner = MagicMock()
        inner.astream_events = MagicMock(return_value=_async_iter(events))
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=(container, True)):
            with patch.object(wrapper, "_clear_container"):
                collected = []
                async for event in wrapper.astream_events("input"):
                    collected.append(event)
                assert collected[0]["data"]["output"] == "Result: a@b.com"

    @pytest.mark.asyncio
    async def test_passes_through_other_events(self):
        events = [
            {"event": "on_chain_start", "run_id": "r1", "data": {"input": "x"}},
            {"event": "on_chain_end", "run_id": "r1", "data": {"output": "y"}},
        ]
        inner = MagicMock()
        inner.astream_events = MagicMock(return_value=_async_iter(events))
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=({}, True)):
            with patch.object(wrapper, "_clear_container"):
                collected = []
                async for event in wrapper.astream_events("input"):
                    collected.append(event)
                assert len(collected) == 2
                assert collected[0]["event"] == "on_chain_start"
                assert collected[1]["event"] == "on_chain_end"

    @pytest.mark.asyncio
    async def test_flushes_remaining_buffers_at_end(self):
        container = {"[EMAIL_1]": "a@b.com"}
        events = [
            {
                "event": "on_chat_model_stream",
                "run_id": "r1",
                "data": {"chunk": AIMessageChunk(content="orphan [EMAIL_1]")},
            },
        ]
        inner = MagicMock()
        inner.astream_events = MagicMock(return_value=_async_iter(events))
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=(container, True)):
            with patch.object(wrapper, "_clear_container"):
                collected = []
                async for event in wrapper.astream_events("input"):
                    collected.append(event)
                assert len(collected) == 1
                assert collected[0]["data"]["chunk"].content == "orphan a@b.com"

    @pytest.mark.asyncio
    async def test_flushes_on_llm_end(self):
        """on_llm_end should trigger flush just like on_chat_model_end."""
        container = {"[NAME_1]": "Alice"}
        events = [
            {
                "event": "on_chat_model_stream",
                "run_id": "r1",
                "data": {"chunk": AIMessageChunk(content="hi [NAME_1]")},
            },
            {"event": "on_llm_end", "run_id": "r1", "data": {}},
        ]
        inner = MagicMock()
        inner.astream_events = MagicMock(return_value=_async_iter(events))
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=(container, True)):
            with patch.object(wrapper, "_clear_container"):
                collected = []
                async for event in wrapper.astream_events("input"):
                    collected.append(event)
                restored = [e for e in collected if e.get("data", {}).get("chunk")]
                assert restored[0]["data"]["chunk"].content == "hi Alice"

    @pytest.mark.asyncio
    async def test_multiple_runs_buffered_independently(self):
        container = {"[A]": "val_a", "[B]": "val_b"}
        events = [
            {
                "event": "on_chat_model_stream",
                "run_id": "r1",
                "data": {"chunk": AIMessageChunk(content="[A]")},
            },
            {
                "event": "on_chat_model_stream",
                "run_id": "r2",
                "data": {"chunk": AIMessageChunk(content="[B]")},
            },
            {"event": "on_chat_model_end", "run_id": "r1", "data": {}},
            {"event": "on_chat_model_end", "run_id": "r2", "data": {}},
        ]
        inner = MagicMock()
        inner.astream_events = MagicMock(return_value=_async_iter(events))
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=(container, True)):
            with patch.object(wrapper, "_clear_container"):
                collected = []
                async for event in wrapper.astream_events("input"):
                    collected.append(event)
                chunks = [
                    e["data"]["chunk"]
                    for e in collected
                    if e.get("data", {}).get("chunk")
                ]
                contents = sorted(c.content for c in chunks)
                assert contents == ["val_a", "val_b"]

    @pytest.mark.asyncio
    async def test_clears_container_after_stream_events(self):
        inner = MagicMock()
        inner.astream_events = MagicMock(return_value=_async_iter([]))
        wrapper = PIIAwareRunnable(inner)
        with patch.object(wrapper, "_setup_container", return_value=({}, True)):
            with patch.object(wrapper, "_clear_container") as mock_clear:
                async for _ in wrapper.astream_events("input"):
                    pass
                mock_clear.assert_called_once_with(True)
