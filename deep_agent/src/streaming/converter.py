"""Message format conversion for streaming API responses.

This module converts internal ChatMessage objects to the simplified JSON format
sent to clients via streaming endpoints. It handles special cases like tool call
rewrites and context metadata injection.
"""

from typing import Any, Dict, List, Union

from langchain_core.messages import BaseMessage

from deep_agent.src.streaming.context import StreamContext


def convert_message_to_api_format(
    chat_message: Any, ctx: StreamContext
) -> dict[str, Any]:
    """Convert ChatMessage to simplified API format.

    Args:
        chat_message: The chat message to convert.
        ctx: Stream context with metadata.

    Returns:
        Simplified message dictionary with type, content, and context metadata.
    """
    content = {
        "type": chat_message.type,
        "content": chat_message.content,
    }

    # Add optional message-specific fields
    if chat_message.tool_calls:
        # Rewrite "task" tool name to actual subagent name for better UI display
        content["tool_calls"] = [
            {**tc, "name": tc["args"]["subagent_type"]}
            if tc.get("name") == "task" and "subagent_type" in tc.get("args", {})
            else tc
            for tc in chat_message.tool_calls
        ]
    if chat_message.tool_call_id:
        content["tool_call_id"] = chat_message.tool_call_id
    if chat_message.response_metadata:
        content["response_metadata"] = chat_message.response_metadata

    # Add context metadata (always present, authoritative for the stream)
    content["run_id"] = ctx.run_id
    content["trace_id"] = ctx.trace_id
    content["thread_id"] = ctx.thread_id
    content["session_id"] = ctx.session_id
    content["user_id"] = ctx.user_id

    return content


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
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]


def should_skip_message(message: BaseMessage) -> tuple[bool, str | None]:
    """Determine if a message should be skipped.

    Args:
        message: The message to check.

    Returns:
        Tuple of (should_skip, reason).
    """
    from langchain_core.messages import AIMessage, ToolMessage

    # Skip empty tool messages
    if isinstance(message, ToolMessage) and not message.content:
        tool_name = message.name or "unknown"
        return (
            True,
            f"Subagent '{tool_name}' returned empty result (tool_call_id={message.tool_call_id})",
        )

    # Skip empty AI messages from malformed function calls
    if (
        isinstance(message, AIMessage)
        and not message.content
        and not message.tool_calls
    ):
        reason = message.response_metadata.get("finish_reason", "")
        if reason == "MALFORMED_FUNCTION_CALL":
            return True, "LLM returned MALFORMED_FUNCTION_CALL — skipping empty message"

    return False, None
