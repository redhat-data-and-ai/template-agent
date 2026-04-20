"""Unit tests for history route."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from template_agent.src.api.routes.memory.history import (
    convert_with_metadata,
    history,
    is_subagent_checkpoint,
    rewrite_task_tool_calls,
)
from template_agent.src.schema import ChatHistoryResponse


def create_mock_state(
    messages=None,
    metadata=None,
    config=None,
):
    """Helper to create a mock StateSnapshot object."""
    mock_state = MagicMock()
    mock_state.values = {"messages": messages or []}
    mock_state.metadata = metadata or {}
    mock_state.config = config or {"configurable": {}}
    return mock_state


class TestHistory:
    """Tests for history endpoint."""

    @pytest.mark.asyncio
    async def test_successful_history_retrieval(self):
        """Test successful retrieval of chat history with metadata mapping."""
        # Create mock states representing the conversation history
        # State history is returned newest first, but we'll process oldest to newest
        state1 = create_mock_state(
            messages=[HumanMessage(content="Hello", id="msg1")],
            metadata={
                "user_id": "user123",
                "run_id": "run1",
                "trace_id": "trace1",
                "session_id": "session1",
            },
            config={"configurable": {"thread_id": "thread456"}},
        )

        state2 = create_mock_state(
            messages=[
                HumanMessage(content="Hello", id="msg1"),
                AIMessage(content="Hi there!", id="msg2"),
            ],
            metadata={
                "user_id": "user123",
                "run_id": "run2",
                "trace_id": "trace2",
                "session_id": "session2",
            },
            config={"configurable": {"thread_id": "thread456"}},
        )

        # Mock agent
        mock_agent = MagicMock()

        # aget_state_history returns newest first
        async def mock_state_history(config):
            for state in [state2, state1]:  # Newest first
                yield state

        mock_agent.aget_state_history = mock_state_history
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.api.routes.memory.history.get_template_agent"
        ) as mock_get_agent:
            mock_get_agent.return_value = mock_agent

            result = await history("user123", "thread456")

            assert isinstance(result, ChatHistoryResponse)
            assert len(result.messages) == 2
            assert result.messages[0].type == "human"
            assert result.messages[0].content == "Hello"
            assert result.messages[0].trace_id == "trace1"
            assert result.messages[0].thread_id == "thread456"
            assert result.messages[1].type == "ai"
            assert result.messages[1].content == "Hi there!"
            assert result.messages[1].trace_id == "trace2"

    @pytest.mark.asyncio
    async def test_ownership_check_fails(self):
        """Test when user doesn't own the thread."""
        # Create state with different user_id
        state1 = create_mock_state(
            messages=[HumanMessage(content="Hello", id="msg1")],
            metadata={
                "user_id": "other_user",
                "run_id": "run1",
                "trace_id": "trace1",
            },
            config={"configurable": {"thread_id": "thread456"}},
        )

        mock_agent = MagicMock()

        async def mock_state_history(config):
            yield state1

        mock_agent.aget_state_history = mock_state_history
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.api.routes.memory.history.get_template_agent"
        ) as mock_get_agent:
            mock_get_agent.return_value = mock_agent

            result = await history("user123", "thread456")

            assert isinstance(result, ChatHistoryResponse)
            assert result.messages == []

    @pytest.mark.asyncio
    async def test_no_state_history(self):
        """Test when no state history exists."""
        mock_agent = MagicMock()

        async def mock_state_history(config):
            # Empty generator
            return
            yield  # Make it a generator

        mock_agent.aget_state_history = mock_state_history
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.api.routes.memory.history.get_template_agent"
        ) as mock_get_agent:
            mock_get_agent.return_value = mock_agent

            result = await history("user123", "thread456")

            assert isinstance(result, ChatHistoryResponse)
            assert result.messages == []

    @pytest.mark.asyncio
    async def test_no_messages_in_state(self):
        """Test when state exists but has no messages."""
        state1 = create_mock_state(
            messages=[],
            metadata={
                "user_id": "user123",
                "run_id": "run1",
                "trace_id": "trace1",
            },
            config={"configurable": {"thread_id": "thread456"}},
        )

        mock_agent = MagicMock()

        async def mock_state_history(config):
            yield state1

        mock_agent.aget_state_history = mock_state_history
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.api.routes.memory.history.get_template_agent"
        ) as mock_get_agent:
            mock_get_agent.return_value = mock_agent

            result = await history("user123", "thread456")

            assert isinstance(result, ChatHistoryResponse)
            assert result.messages == []

    @pytest.mark.asyncio
    async def test_message_conversion_error_continues(self):
        """Test that message conversion errors don't stop processing."""
        state1 = create_mock_state(
            messages=[
                HumanMessage(content="Valid message", id="msg1"),
                MagicMock(),  # Invalid message that will fail conversion
                AIMessage(content="Another valid", id="msg3"),
            ],
            metadata={
                "user_id": "user123",
                "run_id": "run1",
                "trace_id": "trace1",
                "session_id": "session1",
            },
            config={"configurable": {"thread_id": "thread456"}},
        )

        mock_agent = MagicMock()

        async def mock_state_history(config):
            yield state1

        mock_agent.aget_state_history = mock_state_history
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.api.routes.memory.history.get_template_agent"
        ) as mock_get_agent:
            mock_get_agent.return_value = mock_agent

            result = await history("user123", "thread456")

            # Should have 2 valid messages, skipping the invalid one
            assert isinstance(result, ChatHistoryResponse)
            assert len(result.messages) == 2
            assert result.messages[0].content == "Valid message"
            assert result.messages[1].content == "Another valid"

    @pytest.mark.asyncio
    async def test_metadata_enrichment(self):
        """Test that messages get metadata from checkpoint where they were created."""
        state1 = create_mock_state(
            messages=[HumanMessage(content="Test", id="msg1")],
            metadata={
                "user_id": "user123",
                "run_id": "runxyz",
                "trace_id": "traceabc",
                "session_id": "sessionabc",
            },
            config={"configurable": {"thread_id": "thread456"}},
        )

        mock_agent = MagicMock()

        async def mock_state_history(config):
            yield state1

        mock_agent.aget_state_history = mock_state_history
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.api.routes.memory.history.get_template_agent"
        ) as mock_get_agent:
            mock_get_agent.return_value = mock_agent

            result = await history("user123", "thread456")

            assert len(result.messages) == 1
            assert result.messages[0].run_id == "runxyz"
            assert result.messages[0].trace_id == "traceabc"
            assert result.messages[0].session_id == "sessionabc"
            assert result.messages[0].thread_id == "thread456"

    @pytest.mark.asyncio
    async def test_subagent_checkpoints_filtered(self):
        """Test that subagent checkpoints are filtered out of metadata mapping."""
        # Main agent state
        state_main = create_mock_state(
            messages=[HumanMessage(content="Hello", id="msg1")],
            metadata={
                "user_id": "user123",
                "run_id": "mainrun",
                "trace_id": "maintrace",
                "session_id": "session1",
            },
            config={"configurable": {"thread_id": "thread456"}},
        )

        # Subagent state (should be skipped)
        state_subagent = create_mock_state(
            messages=[
                HumanMessage(content="Hello", id="msg1"),
                AIMessage(content="Subagent response", id="msg2"),
            ],
            metadata={
                "user_id": "user123",
                "run_id": "subagentrun",
                "trace_id": "subagenttrace",
                "lc_agent_name": "analyst",  # This marks it as subagent
            },
            config={
                "configurable": {"thread_id": "thread456", "checkpoint_ns": "analyst"}
            },
        )

        # Final main agent state
        state_final = create_mock_state(
            messages=[
                HumanMessage(content="Hello", id="msg1"),
                AIMessage(content="Final response", id="msg3"),
            ],
            metadata={
                "user_id": "user123",
                "run_id": "finalrun",
                "trace_id": "finaltrace",
                "session_id": "session1",
            },
            config={"configurable": {"thread_id": "thread456"}},
        )

        mock_agent = MagicMock()

        async def mock_state_history(config):
            # Newest first
            for state in [state_final, state_subagent, state_main]:
                yield state

        mock_agent.aget_state_history = mock_state_history
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.api.routes.memory.history.get_template_agent"
        ) as mock_get_agent:
            mock_get_agent.return_value = mock_agent

            result = await history("user123", "thread456")

            assert len(result.messages) == 2
            # First message should have maintrace (from state_main)
            assert result.messages[0].trace_id == "maintrace"
            # Second message should have finaltrace (from state_final)
            # NOT subagenttrace because subagent state was filtered
            assert result.messages[1].trace_id == "finaltrace"

    @pytest.mark.asyncio
    async def test_tool_messages_preserve_tool_call_id(self):
        """Test that tool messages preserve their tool_call_id."""
        state1 = create_mock_state(
            messages=[
                HumanMessage(content="Hello", id="msg1"),
                AIMessage(
                    content="",
                    id="msg2",
                    tool_calls=[{"name": "test_tool", "args": {}, "id": "tool-123"}],
                ),
                ToolMessage(content="Tool result", tool_call_id="tool-123", id="msg3"),
            ],
            metadata={
                "user_id": "user123",
                "run_id": "run1",
                "trace_id": "trace1",
                "session_id": "session1",
            },
            config={"configurable": {"thread_id": "thread456"}},
        )

        mock_agent = MagicMock()

        async def mock_state_history(config):
            yield state1

        mock_agent.aget_state_history = mock_state_history
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.api.routes.memory.history.get_template_agent"
        ) as mock_get_agent:
            mock_get_agent.return_value = mock_agent

            result = await history("user123", "thread456")

            assert len(result.messages) == 3
            # Tool message should have tool_call_id
            assert result.messages[2].type == "tool"
            assert result.messages[2].tool_call_id == "tool-123"

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_unique_trace_ids(self):
        """Test that multi-turn conversations have unique trace_id per turn."""
        # Turn 1: User asks, AI responds
        state1 = create_mock_state(
            messages=[HumanMessage(content="First question", id="msg1")],
            metadata={
                "user_id": "user123",
                "run_id": "run1",
                "trace_id": "trace1",
                "session_id": "session1",
            },
            config={"configurable": {"thread_id": "thread456"}},
        )

        state2 = create_mock_state(
            messages=[
                HumanMessage(content="First question", id="msg1"),
                AIMessage(content="First answer", id="msg2"),
            ],
            metadata={
                "user_id": "user123",
                "run_id": "run2",
                "trace_id": "trace1",  # Same trace for same turn
                "session_id": "session1",
            },
            config={"configurable": {"thread_id": "thread456"}},
        )

        # Turn 2: User asks again, AI responds
        state3 = create_mock_state(
            messages=[
                HumanMessage(content="First question", id="msg1"),
                AIMessage(content="First answer", id="msg2"),
                HumanMessage(content="Second question", id="msg3"),
            ],
            metadata={
                "user_id": "user123",
                "run_id": "run3",
                "trace_id": "trace2",  # Different trace for new turn
                "session_id": "session1",
            },
            config={"configurable": {"thread_id": "thread456"}},
        )

        state4 = create_mock_state(
            messages=[
                HumanMessage(content="First question", id="msg1"),
                AIMessage(content="First answer", id="msg2"),
                HumanMessage(content="Second question", id="msg3"),
                AIMessage(content="Second answer", id="msg4"),
            ],
            metadata={
                "user_id": "user123",
                "run_id": "run4",
                "trace_id": "trace2",  # Same trace for same turn
                "session_id": "session1",
            },
            config={"configurable": {"thread_id": "thread456"}},
        )

        mock_agent = MagicMock()

        async def mock_state_history(config):
            # Newest first
            for state in [state4, state3, state2, state1]:
                yield state

        mock_agent.aget_state_history = mock_state_history
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.api.routes.memory.history.get_template_agent"
        ) as mock_get_agent:
            mock_get_agent.return_value = mock_agent

            result = await history("user123", "thread456")

            assert len(result.messages) == 4
            # Turn 1 messages should have trace1
            assert result.messages[0].trace_id == "trace1"
            assert result.messages[1].trace_id == "trace1"
            # Turn 2 messages should have trace2
            assert result.messages[2].trace_id == "trace2"
            assert result.messages[3].trace_id == "trace2"

    @pytest.mark.asyncio
    async def test_agent_error_raises_http_exception(self):
        """Test that agent errors raise HTTPException."""
        mock_agent = MagicMock()

        async def mock_state_history_with_error(config):
            raise Exception("Agent connection failed")
            yield  # Make it a generator

        mock_agent.aget_state_history = mock_state_history_with_error
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.api.routes.memory.history.get_template_agent"
        ) as mock_get_agent:
            mock_get_agent.return_value = mock_agent

            with pytest.raises(HTTPException) as exc_info:
                await history("user123", "thread456")

            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Failed to retrieve chat history"


class TestHistoryHelpers:
    """Tests for history helper functions."""

    def test_is_subagent_checkpoint_with_lc_agent_name(self):
        """Test subagent detection via lc_agent_name."""
        state = create_mock_state(
            metadata={"lc_agent_name": "analyst"},
            config={"configurable": {}},
        )

        assert is_subagent_checkpoint(state) is True

    def test_is_subagent_checkpoint_with_checkpoint_ns(self):
        """Test subagent detection via checkpoint_ns."""
        state = create_mock_state(
            metadata={},
            config={"configurable": {"checkpoint_ns": "subagent"}},
        )

        assert is_subagent_checkpoint(state) is True

    def test_is_subagent_checkpoint_main_agent(self):
        """Test that main agent checkpoints are not flagged."""
        state = create_mock_state(
            metadata={"user_id": "user123"},
            config={"configurable": {"thread_id": "thread123"}},
        )

        assert is_subagent_checkpoint(state) is False

    def test_is_subagent_checkpoint_none_metadata(self):
        """Test handling of None metadata."""
        state = create_mock_state(
            metadata=None,
            config={"configurable": {}},
        )

        assert is_subagent_checkpoint(state) is False

    def test_convert_with_metadata_success(self):
        """Test successful message conversion with metadata."""
        msg = HumanMessage(content="Test")
        metadata_map = {
            0: {
                "run_id": "run123",
                "trace_id": "trace123",
                "session_id": "session123",
            }
        }

        result = convert_with_metadata(msg, 0, "thread123", metadata_map)

        assert result is not None
        assert result.content == "Test"
        assert result.run_id == "run123"
        assert result.trace_id == "trace123"
        assert result.session_id == "session123"
        assert result.thread_id == "thread123"

    def test_convert_with_metadata_no_metadata_available(self):
        """Test conversion when no metadata is available for index."""
        msg = HumanMessage(content="Test")
        metadata_map = {}  # Empty map

        result = convert_with_metadata(msg, 0, "thread123", metadata_map)

        assert result is not None
        assert result.content == "Test"
        assert result.thread_id == "thread123"
        # Metadata fields should be None when not in map
        assert result.run_id is None
        assert result.trace_id is None
        assert result.session_id is None

    def test_convert_with_metadata_invalid_message(self):
        """Test that invalid messages return None."""
        invalid_msg = MagicMock()  # Will fail conversion

        result = convert_with_metadata(invalid_msg, 0, "thread123", {})

        assert result is None

    def test_rewrite_task_tool_calls_rewrites_subagent(self):
        """Test that 'task' tool calls are rewritten to subagent names."""
        from template_agent.src.schema import ChatMessage

        msg = ChatMessage(
            type="ai",
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"subagent_type": "analyst", "description": "Analyze data"},
                    "id": "tc123",
                    "type": "tool_call",
                }
            ],
        )

        rewrite_task_tool_calls(msg)

        assert msg.tool_calls[0]["name"] == "analyst"
        assert msg.tool_calls[0]["args"]["subagent_type"] == "analyst"

    def test_rewrite_task_tool_calls_preserves_non_task_tools(self):
        """Test that non-task tool calls are not modified."""
        from template_agent.src.schema import ChatMessage

        msg = ChatMessage(
            type="ai",
            content="",
            tool_calls=[
                {
                    "name": "execute",
                    "args": {"command": "ls"},
                    "id": "tc456",
                    "type": "tool_call",
                }
            ],
        )

        rewrite_task_tool_calls(msg)

        assert msg.tool_calls[0]["name"] == "execute"

    def test_rewrite_task_tool_calls_handles_no_tool_calls(self):
        """Test that messages without tool calls are handled gracefully."""
        from template_agent.src.schema import ChatMessage

        msg = ChatMessage(type="ai", content="Hello", tool_calls=[])

        # Should not raise exception
        rewrite_task_tool_calls(msg)

        assert msg.tool_calls == []

    def test_convert_with_metadata_applies_rewriting(self):
        """Test that convert_with_metadata applies tool call rewriting."""
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"subagent_type": "publisher", "description": "Send email"},
                    "id": "tc789",
                }
            ],
        )

        result = convert_with_metadata(msg, 0, "thread123", {})

        assert result is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "publisher"
