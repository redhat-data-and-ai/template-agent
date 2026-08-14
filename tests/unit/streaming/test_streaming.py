"""Unit tests for streaming components."""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Overwrite

from deep_agent.src.streaming import (
    MessageDeduplicator,
    StreamContext,
    ToolCallTracker,
    remove_tool_calls,
)
from deep_agent.src.streaming.converter import (
    convert_message_to_api_format,
    should_skip_message,
)
from deep_agent.src.streaming.handlers import (
    TokenEventHandler,
    UpdateEventHandler,
)


@pytest.fixture
def stream_context():
    """Fixture providing a standard StreamContext for tests."""
    return StreamContext(
        run_id="test_run_1",
        trace_id="test_trace_1",
        thread_id="test_thread_1",
        session_id="test_session_1",
        user_id="test_user",
        stream_tokens=True,
    )


@pytest.fixture
def deduplicator():
    """Fixture providing a fresh MessageDeduplicator."""
    return MessageDeduplicator()


@pytest.fixture
def tracker():
    """Fixture providing a fresh ToolCallTracker."""
    return ToolCallTracker()


class TestMessageDeduplicator:
    """Tests for MessageDeduplicator component."""

    def test_mark_and_check_seen(self, deduplicator):
        """Test marking messages as seen and checking if seen."""
        msg = AIMessage(content="Hello", id="msg_1")

        assert not deduplicator.is_seen(msg)
        deduplicator.mark_seen(msg)
        assert deduplicator.is_seen(msg)

    def test_get_unseen_messages(self, deduplicator):
        """Test getting only unseen messages."""
        msg1 = AIMessage(content="Hello", id="msg_1")
        msg2 = AIMessage(content="World", id="msg_2")
        msg3 = AIMessage(content="Hello again", id="msg_1")  # Duplicate ID

        messages = [msg1, msg2, msg3]
        unseen = deduplicator.get_unseen_messages(messages)

        assert len(unseen) == 2  # msg1 and msg2 are new
        assert unseen[0].id == "msg_1"
        assert unseen[1].id == "msg_2"

        # Second call should return empty since all are seen
        unseen_again = deduplicator.get_unseen_messages(messages)
        assert len(unseen_again) == 0

    def test_tool_message_uses_tool_call_id(self, deduplicator):
        """Test that ToolMessage without id uses tool_call_id."""
        tool_msg = ToolMessage(content="result", tool_call_id="tc_1", name="test_tool")

        assert not deduplicator.is_seen(tool_msg)
        deduplicator.mark_seen(tool_msg)
        assert deduplicator.is_seen(tool_msg)

    def test_message_without_id_always_unseen(self, deduplicator):
        """Test that messages without stable IDs are never marked as seen."""
        msg_no_id = AIMessage(content="No ID")

        # Should always be unseen since no stable ID
        assert not deduplicator.is_seen(msg_no_id)
        deduplicator.mark_seen(msg_no_id)
        # Still not seen because no ID to track
        assert not deduplicator.is_seen(msg_no_id)

    def test_reset_clears_seen_messages(self, deduplicator):
        """Test that reset clears all seen message IDs."""
        msg = AIMessage(content="Hello", id="msg_1")

        deduplicator.mark_seen(msg)
        assert deduplicator.is_seen(msg)

        deduplicator.reset()
        assert not deduplicator.is_seen(msg)

    def test_populate_from_history(self, deduplicator):
        """Test pre-populating seen IDs from message history."""
        msg1 = AIMessage(content="Old message 1", id="msg_1")
        msg2 = AIMessage(content="Old message 2", id="msg_2")
        history = [msg1, msg2]

        deduplicator.populate_from_history(history)

        assert deduplicator.is_seen(msg1)
        assert deduplicator.is_seen(msg2)


class TestToolCallTracker:
    """Tests for ToolCallTracker component."""

    def test_track_tool_call_from_updates(self, tracker):
        """Test tracking tool call ID from updates stream mode."""
        event = {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "test_tool", "args": {}, "id": "tc_123"}],
                    )
                ]
            }
        }

        tracker.update_from_stream_event("updates", event)
        assert tracker.current_id == "tc_123"

    def test_track_tool_response_from_updates(self, tracker):
        """Test tracking tool response ID from updates stream mode."""
        event = {
            "agent": {
                "messages": [
                    ToolMessage(
                        content="result", tool_call_id="tc_456", name="test_tool"
                    )
                ]
            }
        }

        tracker.update_from_stream_event("updates", event)
        assert tracker.current_id == "tc_456"

    def test_track_from_message_stream(self, tracker):
        """Test tracking from messages stream mode."""
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "test_tool", "args": {}, "id": "tc_789"}],
        )
        event = (msg, {})

        tracker.update_from_stream_event("messages", event)
        assert tracker.current_id == "tc_789"

    def test_extract_tool_call_id(self):
        """Test extracting tool call ID directly from message."""
        from langchain_core.messages import AIMessageChunk

        from deep_agent.src.streaming.tracker import extract_tool_call_id

        msg = AIMessageChunk(
            content="",
            tool_calls=[{"name": "test_tool", "args": {}, "id": "tc_abc"}],
        )

        tool_id = extract_tool_call_id(msg)
        assert tool_id == "tc_abc"

    def test_reset_clears_current_id(self, tracker):
        """Test that reset clears the current tool call ID."""
        event = {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "test_tool", "args": {}, "id": "tc_123"}],
                    )
                ]
            }
        }

        tracker.update_from_stream_event("updates", event)
        assert tracker.current_id == "tc_123"

        tracker.reset()
        assert tracker.current_id is None


class TestConverter:
    """Tests for message conversion utilities."""

    def test_should_skip_empty_tool_message(self):
        """Test that empty tool messages are skipped."""
        empty_tool_msg = ToolMessage(content="", tool_call_id="tc_1", name="empty_tool")
        should_skip, reason = should_skip_message(empty_tool_msg)

        assert should_skip
        assert "empty result" in reason
        assert "empty_tool" in reason

    def test_should_skip_malformed_function_call(self):
        """Test that malformed function call messages are skipped."""
        malformed_msg = AIMessage(
            content="",
            response_metadata={"finish_reason": "MALFORMED_FUNCTION_CALL"},
        )
        should_skip, reason = should_skip_message(malformed_msg)

        assert should_skip
        assert "MALFORMED_FUNCTION_CALL" in reason

    def test_should_not_skip_normal_message(self):
        """Test that normal messages are not skipped."""
        normal_msg = AIMessage(content="Hello, how can I help?")
        should_skip, reason = should_skip_message(normal_msg)

        assert not should_skip
        assert reason is None

    def test_should_not_skip_ai_message_with_tool_calls(self):
        """Test that AI messages with tool calls are not skipped even if empty content."""
        msg_with_tool = AIMessage(
            content="",
            tool_calls=[{"name": "test_tool", "args": {}, "id": "tc_1"}],
        )
        should_skip, reason = should_skip_message(msg_with_tool)

        assert not should_skip

    def test_convert_message_to_api_format(self, stream_context):
        """Test conversion of chat message to simplified format."""

        class MockChatMessage:
            def __init__(self):
                self.type = "ai"
                self.content = "Hello, how can I help?"
                self.tool_calls = None
                self.tool_call_id = None
                self.response_metadata = {"model": "test-model"}

        chat_msg = MockChatMessage()
        result = convert_message_to_api_format(chat_msg, stream_context)

        assert result["type"] == "ai"
        assert result["content"] == "Hello, how can I help?"
        # run_id and trace_id come from stream context (authoritative)
        assert result["run_id"] == "test_run_1"
        assert result["trace_id"] == "test_trace_1"
        assert result["thread_id"] == "test_thread_1"
        assert result["session_id"] == "test_session_1"
        assert result["user_id"] == "test_user"
        assert result["response_metadata"] == {"model": "test-model"}

    def test_convert_includes_trace_id_from_context(self, stream_context):
        """Test that trace_id and run_id come from stream context (authoritative)."""

        class MockChatMessage:
            def __init__(self):
                self.type = "ai"
                self.content = "Test"
                self.tool_calls = None
                self.tool_call_id = None
                self.response_metadata = {}

        chat_msg = MockChatMessage()
        result = convert_message_to_api_format(chat_msg, stream_context)

        # Verify all context metadata is included (authoritative for the stream)
        assert result["run_id"] == stream_context.run_id
        assert result["trace_id"] == stream_context.trace_id
        assert result["thread_id"] == stream_context.thread_id
        assert result["session_id"] == stream_context.session_id
        assert result["user_id"] == stream_context.user_id

    def test_convert_with_tool_calls(self, stream_context):
        """Test conversion with tool calls, including subagent name rewriting."""

        class MockChatMessage:
            def __init__(self):
                self.type = "ai"
                self.content = ""
                self.tool_calls = [
                    {
                        "name": "task",
                        "args": {"subagent_type": "research_agent", "query": "test"},
                        "id": "tc_1",
                    }
                ]
                self.tool_call_id = None
                self.run_id = "test_run_1"
                self.trace_id = "test_trace_1"
                self.response_metadata = {}

        chat_msg = MockChatMessage()
        result = convert_message_to_api_format(chat_msg, stream_context)

        # Should rewrite "task" to actual subagent name
        assert result["tool_calls"][0]["name"] == "research_agent"
        assert result["tool_calls"][0]["args"]["subagent_type"] == "research_agent"

    def test_remove_tool_calls_string_content(self):
        """Test that remove_tool_calls returns string content unchanged."""
        content = "Hello, how can I help?"
        result = remove_tool_calls(content)

        assert result == "Hello, how can I help?"
        assert isinstance(result, str)

    def test_remove_tool_calls_filters_tool_use(self):
        """Test that remove_tool_calls filters out tool_use items from list content."""
        content = [
            {"type": "text", "text": "Let me search for that"},
            {"type": "tool_use", "name": "search", "id": "tc_1"},
            {"type": "text", "text": "..."},
        ]
        result = remove_tool_calls(content)

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Let me search for that"
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "..."

    def test_remove_tool_calls_preserves_string_items(self):
        """Test that remove_tool_calls preserves string items in list content."""
        content = [
            "Plain string",
            {"type": "text", "text": "Dict content"},
            {"type": "tool_use", "name": "search", "id": "tc_1"},
        ]
        result = remove_tool_calls(content)

        assert len(result) == 2
        assert result[0] == "Plain string"
        assert result[1]["type"] == "text"

    def test_remove_tool_calls_empty_list(self):
        """Test that remove_tool_calls handles empty list."""
        content = []
        result = remove_tool_calls(content)

        assert result == []
        assert isinstance(result, list)


class TestTokenEventHandler:
    """Tests for TokenEventHandler."""

    def test_handle_basic_token_streaming(self, tracker, stream_context):
        """Test basic token streaming functionality."""
        from langchain_core.messages import AIMessageChunk

        handler = TokenEventHandler(tracker)

        msg = AIMessageChunk(content="Hello")
        event = (msg, {})

        events = handler.handle(event, stream_context)

        assert len(events) == 1
        assert events[0]["type"] == "token"
        assert events[0]["content"] == "Hello"

    def test_respects_stream_tokens_flag(self, tracker, stream_context):
        """Test that handler respects ctx.stream_tokens flag."""
        from langchain_core.messages import AIMessageChunk

        handler = TokenEventHandler(tracker)

        # Create context with stream_tokens=False
        no_stream_ctx = StreamContext(
            run_id="r1",
            trace_id="tr1",
            thread_id="t1",
            session_id="s1",
            user_id="u1",
            stream_tokens=False,
        )

        msg = AIMessageChunk(content="Hello")
        event = (msg, {})

        events = handler.handle(event, no_stream_ctx)

        # Should return empty list when stream_tokens is False
        assert len(events) == 0

    def test_skips_messages_with_skip_stream_tag(self, tracker, stream_context):
        """Test that messages with skip_stream tag are filtered out."""
        from langchain_core.messages import AIMessageChunk

        handler = TokenEventHandler(tracker)

        msg = AIMessageChunk(content="Hello")
        event = (msg, {"tags": ["skip_stream"]})

        events = handler.handle(event, stream_context)

        assert len(events) == 0

    def test_filters_non_ai_message_chunks(self, tracker, stream_context):
        """Test that non-AIMessageChunk messages are filtered."""
        handler = TokenEventHandler(tracker)

        # Regular AIMessage (not chunk)
        msg = AIMessage(content="Hello")
        event = (msg, {})

        events = handler.handle(event, stream_context)

        assert len(events) == 0

    def test_filters_empty_content(self, tracker, stream_context):
        """Test that messages with empty content are filtered."""
        from langchain_core.messages import AIMessageChunk

        handler = TokenEventHandler(tracker)

        msg = AIMessageChunk(content="")
        event = (msg, {})

        events = handler.handle(event, stream_context)

        assert len(events) == 0

    def test_removes_tool_calls_from_content(self, tracker, stream_context):
        """Test that tool calls are removed from streamed content."""
        from langchain_core.messages import AIMessageChunk

        handler = TokenEventHandler(tracker)

        # Content that includes tool calls (which should be removed)
        msg = AIMessageChunk(
            content=[
                {"type": "text", "text": "Let me help you"},
                {"type": "tool_use", "name": "search", "id": "tc_1"},
            ]
        )
        event = (msg, {})

        events = handler.handle(event, stream_context)

        assert len(events) == 1
        assert events[0]["content"] == "Let me help you"

    def test_associates_tool_call_id_from_message(self, tracker, stream_context):
        """Test that tool call ID is extracted from message."""
        from langchain_core.messages import AIMessageChunk

        handler = TokenEventHandler(tracker)

        msg = AIMessageChunk(
            content="Searching...",
            tool_calls=[{"name": "search", "args": {}, "id": "tc_123"}],
        )
        event = (msg, {})

        events = handler.handle(event, stream_context)

        assert len(events) == 1
        assert events[0]["tool_call_id"] == "tc_123"

    def test_associates_tool_call_id_from_tracker(self, tracker, stream_context):
        """Test that tool call ID is taken from tracker if not in message."""
        from langchain_core.messages import AIMessageChunk

        handler = TokenEventHandler(tracker)

        # Set tracker's current_id
        tracker._current_tool_call_id = "tc_456"

        msg = AIMessageChunk(content="Result from tool")
        event = (msg, {})

        events = handler.handle(event, stream_context)

        assert len(events) == 1
        assert events[0]["tool_call_id"] == "tc_456"

    def test_no_tool_call_id_when_none_available(self, tracker, stream_context):
        """Test that tool_call_id is not added when none is available."""
        from langchain_core.messages import AIMessageChunk

        handler = TokenEventHandler(tracker)

        # Reset tracker to ensure no current_id
        tracker.reset()

        msg = AIMessageChunk(content="Hello")
        event = (msg, {})

        events = handler.handle(event, stream_context)

        assert len(events) == 1
        assert "tool_call_id" not in events[0]

    def test_prefers_message_tool_id_over_tracker(self, tracker, stream_context):
        """Test that message tool_call_id takes precedence over tracker."""
        from langchain_core.messages import AIMessageChunk

        handler = TokenEventHandler(tracker)

        # Set tracker's current_id
        tracker._current_tool_call_id = "tc_old"

        # Message has its own tool call
        msg = AIMessageChunk(
            content="Searching...",
            tool_calls=[{"name": "search", "args": {}, "id": "tc_new"}],
        )
        event = (msg, {})

        events = handler.handle(event, stream_context)

        assert len(events) == 1
        # Should use the message's tool_call_id, not tracker's
        assert events[0]["tool_call_id"] == "tc_new"

    def test_handles_tool_call_chunks(self, tracker, stream_context):
        """Test handling of tool_call_chunks during streaming."""
        from langchain_core.messages import AIMessageChunk

        handler = TokenEventHandler(tracker)

        msg = AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "search", "args": "{}", "id": "tc_789"}],
        )
        event = (msg, {})

        # Should filter out empty content even if tool_call_chunks present
        events = handler.handle(event, stream_context)

        assert len(events) == 0


class TestUpdateEventHandler:
    """Tests for UpdateEventHandler."""

    def test_handle_interrupt_event(self, deduplicator, stream_context):
        """Test handling of interrupt events."""
        handler = UpdateEventHandler(deduplicator)

        event = {
            "__interrupt__": [
                type("Interrupt", (), {"value": "Please confirm action"})()
            ]
        }

        events = handler.handle(event, stream_context)

        assert len(events) == 1
        assert events[0]["type"] == "message"
        assert events[0]["content"]["content"] == "Please confirm action"

    def test_handle_regular_messages(self, deduplicator, stream_context):
        """Test handling of regular message updates."""
        handler = UpdateEventHandler(deduplicator)

        msg = AIMessage(content="Hello", id="msg_1")
        event = {"agent": {"messages": [msg]}}

        events = handler.handle(event, stream_context)

        assert len(events) == 1
        assert events[0]["type"] == "message"
        assert events[0]["content"]["content"] == "Hello"
        assert events[0]["content"]["thread_id"] == "test_thread_1"

    def test_handle_overwrite_deduplication(self, deduplicator, stream_context):
        """Test that Overwrite events are properly deduplicated."""
        handler = UpdateEventHandler(deduplicator)

        msg1 = AIMessage(content="Hello", id="msg_1")
        msg2 = AIMessage(content="World", id="msg_2")

        # First, process msg1 normally
        event1 = {"agent": {"messages": [msg1]}}
        events = handler.handle(event1, stream_context)
        assert len(events) == 1

        # Then send Overwrite with full history
        overwrite_event = {"agent": {"messages": Overwrite([msg1, msg2])}}
        events = handler.handle(overwrite_event, stream_context)

        # Should only get msg2 since msg1 was already seen
        assert len(events) == 1
        assert events[0]["content"]["content"] == "World"

    def test_handle_empty_tool_message_logs_warning(self, deduplicator, stream_context):
        """Test that empty tool messages are skipped with warning."""
        handler = UpdateEventHandler(deduplicator)

        empty_tool = ToolMessage(content="", tool_call_id="tc_1", name="test_tool")
        event = {"agent": {"messages": [empty_tool]}}

        events = handler.handle(event, stream_context)

        # Should be filtered out
        assert len(events) == 0

    def test_handle_multiple_nodes(self, deduplicator, stream_context):
        """Test handling events from multiple nodes."""
        handler = UpdateEventHandler(deduplicator)

        event = {
            "node1": {"messages": [AIMessage(content="From node 1", id="msg_1")]},
            "node2": {"messages": [AIMessage(content="From node 2", id="msg_2")]},
        }

        events = handler.handle(event, stream_context)

        assert len(events) == 2
        contents = [e["content"]["content"] for e in events]
        assert "From node 1" in contents
        assert "From node 2" in contents
