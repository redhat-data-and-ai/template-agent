"""Utility functions for handling agent messages and conversions.

This module provides utility functions for converting between different message
formats, handling message content, and managing tool calls in the template agent.
"""

from typing import Any, Dict, List, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import ChatMessage as LangchainChatMessage

from template_agent.src.schema import ChatMessage, ToolCall


def convert_message_content_to_string(
    content: Union[str, List[Union[str, Dict[str, Any]]]],
) -> str:
    """Convert message content to string format.

    This function handles the conversion of message content from various formats
    (string or list of strings/dicts) to a unified string format. It processes
    content items and extracts text from structured content.

    Args:
        content: The content to convert. Can be a string or a list containing
            strings and dictionaries with content information.

    Returns:
        The converted string content, concatenating all text elements.
    """
    if isinstance(content, str):
        return content

    text: List[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item.get("type") == "text":
            text.append(content_item.get("text", ""))

    return "".join(text)


def langchain_to_chat_message(message: BaseMessage) -> ChatMessage:
    """Create a ChatMessage from a LangChain message.

    This function converts LangChain message objects to the internal ChatMessage
    format used by the template agent. It handles different message types and
    preserves relevant metadata.

    Args:
        message: The LangChain message to convert. Must be one of the supported
            message types (HumanMessage, AIMessage, ToolMessage, or ChatMessage).

    Returns:
        The converted ChatMessage with appropriate type and content.

    Raises:
        ValueError: If the message type is not supported or has an invalid role.
    """
    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                content=convert_message_content_to_string(message.content),
            )
            return human_message

        case AIMessage():
            ai_message = ChatMessage(
                type="ai",
                content=convert_message_content_to_string(message.content),
            )
            # Handle tool calls from both direct attribute and additional_kwargs
            tool_calls = message.tool_calls or []
            if message.additional_kwargs and "tool_calls" in message.additional_kwargs:
                tool_calls.extend(message.additional_kwargs["tool_calls"])

            if tool_calls:
                # Ensure tool calls have the correct structure
                formatted_tool_calls = []
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        # Ensure required fields are present and properly typed
                        if "name" in tool_call and "args" in tool_call:
                            # Create a proper ToolCall object
                            formatted_call: ToolCall = {
                                "name": str(tool_call["name"]),
                                "args": dict(tool_call["args"]),
                                "id": str(tool_call.get("id"))
                                if tool_call.get("id")
                                else None,
                                "type": "tool_call",
                            }
                            formatted_tool_calls.append(formatted_call)
                ai_message.tool_calls = formatted_tool_calls

            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata
            if message.additional_kwargs:
                extra_metadata = message.additional_kwargs.get("response_metadata")
                if isinstance(extra_metadata, dict):
                    ai_message.response_metadata.update(extra_metadata)
                ai_message.ai_call_id = message.additional_kwargs.get("ai_call_id")
            return ai_message

        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                content=convert_message_content_to_string(message.content),
                tool_call_id=message.tool_call_id,
            )
            return tool_message

        case LangchainChatMessage():
            if message.role == "custom":
                custom_data = message.content[0] if message.content else {}
                custom_message = ChatMessage(
                    type="custom",
                    content="",
                    custom_data=custom_data,
                )
                return custom_message
            else:
                raise ValueError(f"Unsupported chat message role: {message.role}")

        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")


def remove_tool_calls(
    content: Union[str, List[Union[str, Dict[str, Any]]]],
) -> Union[str, List[Union[str, Dict[str, Any]]]]:
    """Remove tool calls from message content.

    This function filters out tool call content from message content, particularly
    useful for handling streaming responses from models that include tool calls
    in their content stream.

    Args:
        content: The content to process. Can be a string or a list containing
            strings and dictionaries with content information.

    Returns:
        The content with tool calls removed. Returns the same type as input.
    """
    if isinstance(content, str):
        return content

    # Currently only Anthropic models stream tool calls, using content item type tool_use
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item.get("type") != "tool_use"
    ]
