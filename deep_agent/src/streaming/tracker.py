"""Tool call tracking for enhanced UI feedback.

This module provides ToolCallTracker to accumulate tool call information from
streaming chunks and emit complete tool call events. This enables UIs to show
tool invocations with full context even during token streaming.
"""

from typing import Any

from langchain_core.messages import AIMessageChunk

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def extract_tool_call_id(msg: AIMessageChunk) -> str | None:
    """Extract tool call ID from an AIMessageChunk.

    Modern LangChain automatically populates tool_calls from tool_call_chunks
    during streaming, so we only need to check tool_calls.

    Args:
        msg: The message chunk to extract from.

    Returns:
        The tool call ID if available, None otherwise.
    """
    try:
        if msg.tool_calls:
            tool_call_id = msg.tool_calls[0].get("id")
            return tool_call_id if isinstance(tool_call_id, str) else None
        return None
    except (IndexError, KeyError) as e:
        logger.debug(f"Could not extract tool call ID: {e}")
        return None


class ToolCallTracker:
    """Tracks active tool calls to associate streaming tokens with tools.

    When a tool is invoked, streaming tokens that follow should be
    associated with that tool's response. This tracker maintains the
    current tool call ID for proper attribution in the UI.
    """

    def __init__(self) -> None:
        """Initialize the tracker."""
        self._current_tool_call_id: str | None = None

    def reset(self) -> None:
        """Clear the current tool call ID."""
        self._current_tool_call_id = None

    @property
    def current_id(self) -> str | None:
        """Get the current tool call ID being tracked."""
        return self._current_tool_call_id

    def update_from_stream_event(self, stream_mode: str, event: Any) -> None:
        """Update tracking based on a stream event.

        Args:
            stream_mode: The type of stream event (updates, messages, custom).
            event: The event data.
        """
        try:
            if stream_mode == "updates":
                self._update_from_updates(event)
            elif stream_mode == "messages":
                self._update_from_message_stream(event)
        except Exception as e:
            logger.debug(f"Tool call tracking error: {e}")

    def _update_from_updates(self, event: dict) -> None:
        """Update from an 'updates' mode event."""
        from langchain_core.messages import ToolMessage

        for _node, updates in event.items():
            if not updates or "messages" not in updates:
                continue
            for message in updates["messages"]:
                # ToolMessage responding to a tool call
                if isinstance(message, ToolMessage):
                    self._current_tool_call_id = message.tool_call_id
                    return
                # AIMessage with tool calls
                elif message.tool_calls:
                    self._current_tool_call_id = message.tool_calls[0].get("id")
                    return

    def _update_from_message_stream(self, event: tuple) -> None:
        """Update from a 'messages' mode event."""
        from langchain_core.messages import ToolMessage

        msg, _metadata = event
        # ToolMessage responding to a tool call
        if isinstance(msg, ToolMessage):
            self._current_tool_call_id = msg.tool_call_id
        # AIMessage with tool calls
        elif msg.tool_calls:
            self._current_tool_call_id = msg.tool_calls[0].get("id")
