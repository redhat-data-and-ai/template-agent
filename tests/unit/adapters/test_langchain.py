"""Unit tests for message conversion utilities."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deep_agent.src.adapters.langchain import (
    convert_message_content_to_string,
    langchain_to_chat_message,
)
from deep_agent.src.schema import ChatMessage


class TestConvertMessageContentToString:
    """Tests for convert_message_content_to_string function."""

    def test_string_content_passthrough(self):
        """Test that string content is returned unchanged."""
        content = "Hello, world!"
        result = convert_message_content_to_string(content)

        assert result == "Hello, world!"
        assert isinstance(result, str)

    def test_empty_string(self):
        """Test that empty string is handled correctly."""
        content = ""
        result = convert_message_content_to_string(content)

        assert result == ""

    def test_list_with_strings(self):
        """Test that list of strings is concatenated."""
        content = ["Hello", ", ", "world", "!"]
        result = convert_message_content_to_string(content)

        assert result == "Hello, world!"

    def test_list_with_text_dicts(self):
        """Test that list with text dicts extracts text."""
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " world"},
        ]
        result = convert_message_content_to_string(content)

        assert result == "Hello world"

    def test_mixed_list_strings_and_dicts(self):
        """Test that mixed list of strings and dicts is handled."""
        content = [
            "Hello",
            {"type": "text", "text": " beautiful"},
            " world",
            {"type": "text", "text": "!"},
        ]
        result = convert_message_content_to_string(content)

        assert result == "Hello beautiful world!"

    def test_empty_list(self):
        """Test that empty list returns empty string."""
        content = []
        result = convert_message_content_to_string(content)

        assert result == ""

    def test_list_with_non_text_items_ignored(self):
        """Test that non-text dict items are ignored."""
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "name": "search", "id": "tc_1"},
            {"type": "text", "text": " world"},
            {"type": "image", "url": "http://example.com/image.png"},
        ]
        result = convert_message_content_to_string(content)

        assert result == "Hello world"

    def test_complex_mixed_content(self):
        """Test complex content with multiple formats."""
        content = [
            "Starting text",
            {"type": "text", "text": " middle text"},
            {"type": "tool_use", "name": "tool1"},
            " more string",
            {"type": "text", "text": " ending"},
        ]
        result = convert_message_content_to_string(content)

        assert result == "Starting text middle text more string ending"


class TestLangchainToChatMessage:
    """Tests for langchain_to_chat_message function."""

    def test_human_message_simple(self):
        """Test conversion of simple HumanMessage."""
        msg = HumanMessage(content="Hello, AI!")
        result = langchain_to_chat_message(msg)

        assert isinstance(result, ChatMessage)
        assert result.type == "human"
        assert result.content == "Hello, AI!"

    def test_human_message_with_complex_content(self):
        """Test HumanMessage with list content."""
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "What is"},
                " this image?",
            ]
        )
        result = langchain_to_chat_message(msg)

        assert result.type == "human"
        assert result.content == "What is this image?"

    def test_ai_message_simple(self):
        """Test conversion of simple AIMessage."""
        msg = AIMessage(content="I am an AI assistant.")
        result = langchain_to_chat_message(msg)

        assert isinstance(result, ChatMessage)
        assert result.type == "ai"
        assert result.content == "I am an AI assistant."
        assert result.tool_calls == []

    def test_ai_message_with_tool_calls(self):
        """Test AIMessage with tool calls."""
        msg = AIMessage(
            content="Let me search for that.",
            tool_calls=[
                {
                    "name": "search",
                    "args": {"query": "test query"},
                    "id": "tc_123",
                }
            ],
        )
        result = langchain_to_chat_message(msg)

        assert result.type == "ai"
        assert result.content == "Let me search for that."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"
        assert result.tool_calls[0]["args"] == {"query": "test query"}
        assert result.tool_calls[0]["id"] == "tc_123"
        assert result.tool_calls[0]["type"] == "tool_call"

    def test_ai_message_with_multiple_tool_calls(self):
        """Test AIMessage with multiple tool calls."""
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "tool1", "args": {"param": "value1"}, "id": "tc_1"},
                {"name": "tool2", "args": {"param": "value2"}, "id": "tc_2"},
            ],
        )
        result = langchain_to_chat_message(msg)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "tool1"
        assert result.tool_calls[1]["name"] == "tool2"

    def test_ai_message_with_tool_call_with_none_id(self):
        """Test AIMessage with tool call that has None as ID."""
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "search", "args": {"query": "test"}, "id": None},
            ],
        )
        result = langchain_to_chat_message(msg)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] is None

    def test_ai_message_with_response_metadata(self):
        """Test AIMessage with response metadata."""
        msg = AIMessage(
            content="Response",
            response_metadata={
                "model": "test-model",
                "finish_reason": "stop",
                "token_usage": {"total": 100},
            },
        )
        result = langchain_to_chat_message(msg)

        assert result.response_metadata == {
            "model": "test-model",
            "finish_reason": "stop",
            "token_usage": {"total": 100},
        }

    def test_ai_message_empty_response_metadata(self):
        """Test AIMessage with empty response_metadata."""
        msg = AIMessage(content="Test")
        result = langchain_to_chat_message(msg)

        # Should have default empty dict
        assert result.response_metadata == {}

    def test_ai_message_with_complex_content(self):
        """Test AIMessage with complex content."""
        msg = AIMessage(
            content=[
                {"type": "text", "text": "Here is the answer: "},
                "42",
            ]
        )
        result = langchain_to_chat_message(msg)

        assert result.content == "Here is the answer: 42"

    def test_tool_message_simple(self):
        """Test conversion of simple ToolMessage."""
        msg = ToolMessage(
            content="Search result",
            tool_call_id="tc_123",
            name="search",
        )
        result = langchain_to_chat_message(msg)

        assert isinstance(result, ChatMessage)
        assert result.type == "tool"
        assert result.content == "Search result"
        assert result.tool_call_id == "tc_123"

    def test_tool_message_with_complex_content(self):
        """Test ToolMessage with complex content."""
        msg = ToolMessage(
            content=[
                {"type": "text", "text": "Result: "},
                "Success",
            ],
            tool_call_id="tc_456",
            name="test_tool",
        )
        result = langchain_to_chat_message(msg)

        assert result.type == "tool"
        assert result.content == "Result: Success"
        assert result.tool_call_id == "tc_456"

    def test_tool_message_empty_content(self):
        """Test ToolMessage with empty content."""
        msg = ToolMessage(
            content="",
            tool_call_id="tc_789",
            name="empty_tool",
        )
        result = langchain_to_chat_message(msg)

        assert result.type == "tool"
        assert result.content == ""

    def test_unsupported_message_type_raises_error(self):
        """Test that unsupported message types raise ValueError."""
        msg = SystemMessage(content="System message")

        with pytest.raises(ValueError) as exc_info:
            langchain_to_chat_message(msg)

        assert "Unsupported message type" in str(exc_info.value)
        assert "SystemMessage" in str(exc_info.value)

    def test_ai_message_with_empty_tool_calls_list(self):
        """Test AIMessage with empty tool_calls list."""
        msg = AIMessage(content="Test", tool_calls=[])
        result = langchain_to_chat_message(msg)

        # Empty tool_calls list should result in empty list (not None)
        assert result.tool_calls == []

    def test_ai_message_formats_tool_call_types(self):
        """Test that tool calls are formatted with proper type field."""
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "tool1", "args": {"p": "v"}, "id": "tc_1"},
            ],
        )
        result = langchain_to_chat_message(msg)

        # Verify the type field is added
        assert result.tool_calls[0]["type"] == "tool_call"
        assert result.tool_calls[0]["name"] == "tool1"
        assert result.tool_calls[0]["args"] == {"p": "v"}
        assert result.tool_calls[0]["id"] == "tc_1"

    def test_preserves_message_id_reference(self):
        """Test that original message ID is preserved if needed for debugging."""
        msg = AIMessage(content="Test", id="original_msg_123")

        result = langchain_to_chat_message(msg)

        # Our ChatMessage doesn't store the original LangChain message ID,
        # but we can verify the conversion works regardless
        assert result.type == "ai"
        assert result.content == "Test"
