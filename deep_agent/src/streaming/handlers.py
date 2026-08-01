"""Event handlers for processing LangGraph stream events.

This module provides event handler classes (TokenEventHandler, UpdateEventHandler)
that process LangGraph streaming events and convert them into API-friendly formats.
Handles both token-level and update-level streaming modes.
"""

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.types import Overwrite

from deep_agent.src.adapters.langchain import (
    convert_message_content_to_string,
    langchain_to_chat_message,
)
from deep_agent.src.settings import settings
from deep_agent.src.streaming.context import StreamContext
from deep_agent.src.streaming.converter import (
    convert_message_to_api_format,
    remove_tool_calls,
    should_skip_message,
)
from deep_agent.src.streaming.deduplicator import MessageDeduplicator
from deep_agent.src.streaming.tracker import (
    ToolCallTracker,
    extract_tool_call_id,
)
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def _convert_interrupts_to_messages(interrupts: list) -> list:
    """Convert interrupt data to messages.

    Args:
        interrupts: List of interrupt objects.

    Returns:
        List of AIMessage objects.
    """
    messages = []
    for interrupt_data in interrupts:
        content = (
            interrupt_data.value
            if hasattr(interrupt_data, "value")
            else str(interrupt_data)
        )
        messages.append(AIMessage(content=content))
    return messages


def _convert_messages_to_events(
    messages: list, ctx: StreamContext
) -> list[dict[str, Any]]:
    """Convert messages to simplified event format.

    Args:
        messages: List of LangChain messages.
        ctx: Stream context with metadata.

    Returns:
        List of formatted events.
    """
    formatted_events = []

    for message in messages:
        try:
            # Check if message should be skipped
            should_skip, reason = should_skip_message(message)
            if should_skip:
                if reason:
                    logger.warning(reason)
                continue

            # Convert to chat message format
            chat_message = langchain_to_chat_message(message)
            chat_message.run_id = ctx.run_id

            # Convert to simplified format
            formatted_event = {
                "type": "message",
                "content": convert_message_to_api_format(chat_message, ctx),
            }
            formatted_events.append(formatted_event)

        except Exception as e:
            logger.error(f"Error formatting message: {e}")
            formatted_events.append(
                {
                    "type": "error",
                    "content": {
                        "message": "Message formatting error",
                        "recoverable": True,
                    },
                }
            )

    return formatted_events


class UpdateEventHandler:
    """Handles 'updates' stream mode events from LangGraph."""

    def __init__(self, deduplicator: MessageDeduplicator):
        """Initialize the handler.

        Args:
            deduplicator: Message deduplicator for handling replays.
        """
        self.deduplicator = deduplicator

    def handle(self, event: dict[str, Any], ctx: StreamContext) -> list[dict[str, Any]]:
        """Process update events and convert to simplified format.

        Args:
            event: Dictionary mapping node names to update data.
            ctx: Stream context with metadata.

        Returns:
            List of formatted message events.
        """
        messages = self._extract_and_deduplicate_messages(event)
        return _convert_messages_to_events(messages, ctx)

    def _extract_and_deduplicate_messages(self, event: dict[str, Any]) -> list:
        """Extract and deduplicate messages from update event.

        Args:
            event: Update event dictionary.

        Returns:
            List of messages to process.
        """
        all_messages = []

        for node, updates in event.items():
            if node == "__interrupt__":
                all_messages.extend(_convert_interrupts_to_messages(updates))
                continue

            updates = updates or {}
            raw_messages = updates.get("messages", [])
            is_overwrite = isinstance(raw_messages, Overwrite)
            update_messages = raw_messages.value if is_overwrite else raw_messages

            if is_overwrite:
                # Filter to only unseen messages
                update_messages = self.deduplicator.get_unseen_messages(update_messages)
            else:
                # Mark all messages as seen for future deduplication
                for msg in update_messages:
                    self.deduplicator.mark_seen(msg)

            all_messages.extend(update_messages)

        return all_messages


class TokenEventHandler:
    """Handles 'messages' stream mode events (token streaming)."""

    def __init__(self, tracker: ToolCallTracker):
        """Initialize the handler.

        Args:
            tracker: Tool call tracker for associating tokens with tools.
        """
        self.tracker = tracker

    def handle(self, event: tuple, ctx: StreamContext) -> list[dict[str, Any]]:
        """Process token streaming events.

        Args:
            event: Tuple of (message, metadata).
            ctx: Stream context with metadata.

        Returns:
            List containing a single token event, or empty list.
        """
        if not ctx.stream_tokens:
            return []

        msg, metadata = event
        if "skip_stream" in metadata.get("tags", []):
            return []

        if not isinstance(msg, AIMessageChunk):
            return []

        content = remove_tool_calls(msg.content)
        if not content:
            return []

        token_event = {
            "type": "token",
            "content": convert_message_content_to_string(content),
        }

        # Associate token with tool call if applicable
        tool_call_id = extract_tool_call_id(msg) or self.tracker.current_id
        if tool_call_id:
            token_event["tool_call_id"] = tool_call_id

        return [token_event]
