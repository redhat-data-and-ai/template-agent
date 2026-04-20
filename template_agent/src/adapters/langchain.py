"""LangChain message adapter.

This module adapts LangChain's message format to our internal ChatMessage schema.
It serves as the boundary layer between the external LangChain library and our
internal data structures defined in schema.py.

Functions:
    langchain_to_chat_message: Convert LangChain BaseMessage to ChatMessage
    convert_message_content_to_string: Normalize message content to string format
"""

from typing import Any, Dict, List, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

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
        if content_item["type"] == "text":
            text.append(content_item["text"])

    return "".join(text)


def langchain_to_chat_message(message: BaseMessage) -> ChatMessage:
    """Create a ChatMessage from a LangChain message.

    This function converts LangChain message objects to the internal ChatMessage
    format used by the template agent. It handles different message types and
    preserves relevant metadata including run_id, trace_id, and session_id from
    message metadata.

    Args:
        message: The LangChain message to convert. Must be one of the supported
            message types (HumanMessage, AIMessage, ToolMessage, or ChatMessage).

    Returns:
        The converted ChatMessage with appropriate type and content.

    Raises:
        ValueError: If the message type is not supported or has an invalid role.
    """
    # Extract common metadata fields from message.metadata
    metadata = getattr(message, "metadata", None) or {}
    run_id = metadata.get("run_id")
    trace_id = metadata.get("trace_id")
    session_id = metadata.get("session_id")

    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                content=convert_message_content_to_string(message.content),
                run_id=run_id,
                trace_id=trace_id,
                session_id=session_id,
            )
            return human_message

        case AIMessage():
            ai_message = ChatMessage(
                type="ai",
                content=convert_message_content_to_string(message.content),
                run_id=run_id,
                trace_id=trace_id,
                session_id=session_id,
            )

            # Handle tool calls from modern LangChain messages
            if message.tool_calls:
                formatted_tool_calls = []
                for tool_call in message.tool_calls:
                    if isinstance(tool_call, dict):
                        # Ensure required fields are present and properly typed
                        if "name" in tool_call and "args" in tool_call:
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

            # Copy response metadata
            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata

            return ai_message

        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                content=convert_message_content_to_string(message.content),
                tool_call_id=message.tool_call_id,
                run_id=run_id,
                trace_id=trace_id,
                session_id=session_id,
            )
            return tool_message

        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")
