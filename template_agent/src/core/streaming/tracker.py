"""Tool call tracking for enhanced UI feedback."""

from typing import Any

from langchain_core.messages import AIMessageChunk

from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def extract_tool_call_id(msg: AIMessageChunk) -> str | None:
    """Extract tool call ID from an AIMessageChunk.

    Args:
        msg: The message chunk to extract from.

    Returns:
        The tool call ID if available, None otherwise.
    """
    try:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            return msg.tool_calls[0].get("id")

        if hasattr(msg, "tool_call_chunks") and msg.tool_call_chunks:
            return msg.tool_call_chunks[0].get("id")

        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
            return msg.tool_call_id

        return None
    except (AttributeError, IndexError, KeyError) as e:
        logger.debug(f"Could not extract tool call ID: {e}")
        return None


class ToolCallTracker:
    """Tracks active tool calls to associate streaming tokens with tools.

    When a tool is invoked, streaming tokens that follow should be
    associated with that tool's response. This tracker maintains the
    current tool call ID for proper attribution in the UI.
    """

    def __init__(self):
        """Initialize the tracker."""
        self._current_tool_call_id: str | None = None

    def reset(self):
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
        for _node, updates in event.items():
            if not updates or "messages" not in updates:
                continue
            for message in updates["messages"]:
                if hasattr(message, "tool_calls") and message.tool_calls:
                    self._current_tool_call_id = message.tool_calls[0].get("id")
                    return
                elif hasattr(message, "tool_call_id") and message.tool_call_id:
                    self._current_tool_call_id = message.tool_call_id
                    return

    def _update_from_message_stream(self, event: tuple) -> None:
        """Update from a 'messages' mode event."""
        msg, _metadata = event
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            self._current_tool_call_id = msg.tool_calls[0].get("id")
        elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
            self._current_tool_call_id = msg.tool_call_id
