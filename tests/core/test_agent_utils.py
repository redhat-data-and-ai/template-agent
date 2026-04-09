"""Tests for the agent_utils module."""

from unittest.mock import Mock

import pytest

from template_agent.src.core.agent_utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)
from template_agent.src.schema import ChatMessage


class TestAgentUtils:
    """Test cases for agent utility functions."""

    def test_convert_message_content_to_string_simple(self):
        """Test converting simple string content."""
        content = "Hello world"
        result = convert_message_content_to_string(content)
        assert result == "Hello world"

    def test_convert_message_content_to_string_list(self):
        """Test converting list content with text items."""
        content = ["Hello", " ", "world"]
        result = convert_message_content_to_string(content)
        assert result == "Hello world"

    def test_convert_message_content_to_string_mixed(self):
        """Test converting mixed content with text and dict items."""
        content = ["Hello", {"type": "text", "text": " world"}]
        result = convert_message_content_to_string(content)
        assert result == "Hello world"

    def test_convert_message_content_to_string_ignores_non_text(self):
        """Test that non-text dict items are ignored."""
        content = ["Hello", {"type": "image", "url": "test.jpg"}, " world"]
        result = convert_message_content_to_string(content)
        assert result == "Hello world"

    def test_remove_tool_calls_string(self):
        """Test remove_tool_calls with string content."""
        content = "Hello world"
        result = remove_tool_calls(content)
        assert result == "Hello world"

    def test_remove_tool_calls_list_without_tools(self):
        """Test remove_tool_calls with list content without tool calls."""
        content = ["Hello", " world"]
        result = remove_tool_calls(content)
        assert result == ["Hello", " world"]

    def test_remove_tool_calls_list_with_tools(self):
        """Test remove_tool_calls with list content containing tool calls."""
        content = ["Hello", {"type": "tool_use", "tool_use": {}}, " world"]
        result = remove_tool_calls(content)
        assert result == ["Hello", " world"]

    def test_langchain_to_chat_message_human(self):
        """Test converting HumanMessage to ChatMessage."""
        from langchain_core.messages import HumanMessage

        human_msg = HumanMessage(content="Hello")
        result = langchain_to_chat_message(human_msg)

        assert isinstance(result, ChatMessage)
        assert result.type == "human"
        assert result.content == "Hello"

    def test_langchain_to_chat_message_ai(self):
        """Test converting AIMessage to ChatMessage."""
        from langchain_core.messages import AIMessage

        ai_msg = AIMessage(content="Hello", tool_calls=[])
        result = langchain_to_chat_message(ai_msg)

        assert isinstance(result, ChatMessage)
        assert result.type == "ai"
        assert result.content == "Hello"

    def test_langchain_to_chat_message_tool(self):
        """Test converting ToolMessage to ChatMessage."""
        from langchain_core.messages import ToolMessage

        tool_msg = ToolMessage(content="Tool result", tool_call_id="call_123")
        result = langchain_to_chat_message(tool_msg)

        assert isinstance(result, ChatMessage)
        assert result.type == "tool"
        assert result.content == "Tool result"
        assert result.tool_call_id == "call_123"

    def test_langchain_to_chat_message_unsupported(self):
        """Test that unsupported message types raise ValueError."""
        mock_msg = Mock()
        mock_msg.__class__.__name__ = "UnsupportedMessage"

        with pytest.raises(ValueError, match="Unsupported message type"):
            langchain_to_chat_message(mock_msg)
