"""Message conversion utilities for streaming responses."""

from typing import Any

from template_agent.src.core.streaming.context import StreamContext


def convert_message_to_api_format(chat_message, ctx: StreamContext) -> dict[str, Any]:
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
    if chat_message.run_id:
        content["run_id"] = chat_message.run_id
    if chat_message.response_metadata:
        content["response_metadata"] = chat_message.response_metadata
    if chat_message.custom_data:
        content["custom_data"] = chat_message.custom_data

    # Add context metadata (always present)
    content["thread_id"] = ctx.thread_id
    content["session_id"] = ctx.session_id
    content["user_id"] = ctx.user_id

    return content


def should_skip_message(message) -> tuple[bool, str | None]:
    """Determine if a message should be skipped.

    Args:
        message: The message to check.

    Returns:
        Tuple of (should_skip, reason).
    """
    from langchain_core.messages import AIMessage, ToolMessage

    # Skip empty tool messages
    if isinstance(message, ToolMessage) and not message.content:
        tool_name = getattr(message, "name", "unknown")
        tool_id = getattr(message, "tool_call_id", "")
        return (
            True,
            f"Subagent '{tool_name}' returned empty result (tool_call_id={tool_id})",
        )

    # Skip empty AI messages from malformed function calls
    if (
        isinstance(message, AIMessage)
        and not message.content
        and not getattr(message, "tool_calls", None)
    ):
        metadata = getattr(message, "response_metadata", {}) or {}
        reason = metadata.get("finish_reason", "")
        if reason == "MALFORMED_FUNCTION_CALL":
            return True, "LLM returned MALFORMED_FUNCTION_CALL — skipping empty message"

    return False, None
