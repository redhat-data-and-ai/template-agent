"""Message deduplication for handling LangGraph checkpoint replays.

This module provides MessageDeduplicator to prevent duplicate messages when
LangGraph replays from checkpoints. It tracks message IDs and filters out
messages that have already been seen in the current stream.
"""

from langchain_core.messages import BaseMessage, ToolMessage


def extract_message_id(msg: BaseMessage) -> str | None:
    """Extract a stable identifier from a message.

    Args:
        msg: A LangChain message object.

    Returns:
        A stable ID string, or None if no stable ID exists.
    """
    msg_id: str | None
    if isinstance(msg.id, str):
        msg_id = msg.id
    elif isinstance(msg, ToolMessage):
        # ToolMessages may not have .id set; use tool_call_id as fallback
        msg_id = f"tool_{msg.tool_call_id}"
    else:
        msg_id = None
    return msg_id


class MessageDeduplicator:
    """Tracks and filters duplicate messages across checkpoint restores.

    LangGraph can replay message history via Overwrite updates when
    resuming from checkpoints. This class ensures we only emit new
    messages to avoid duplicate streaming.
    """

    def __init__(self) -> None:
        """Initialize the deduplicator."""
        self._seen_ids: set[str] = set()

    def reset(self) -> None:
        """Clear all seen message IDs."""
        self._seen_ids.clear()

    def mark_seen(self, msg: BaseMessage) -> None:
        """Mark a message as seen.

        Args:
            msg: A LangChain message object.
        """
        msg_id = extract_message_id(msg)
        if msg_id:
            self._seen_ids.add(msg_id)

    def is_seen(self, msg: BaseMessage) -> bool:
        """Check if a message has been seen before.

        Args:
            msg: A LangChain message object.

        Returns:
            True if the message was previously seen, False otherwise.
        """
        msg_id = extract_message_id(msg)
        if msg_id is None:
            # No stable ID - can't reliably deduplicate
            return False
        return msg_id in self._seen_ids

    def get_unseen_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Get only unseen messages from a list, marking them as seen.

        Args:
            messages: List of LangChain message objects.

        Returns:
            List of messages not previously seen.
        """
        unseen = []
        for msg in messages:
            msg_id = extract_message_id(msg)
            if msg_id is None:
                # No stable ID - always include to avoid data loss
                unseen.append(msg)
            elif msg_id not in self._seen_ids:
                unseen.append(msg)
                self._seen_ids.add(msg_id)
        return unseen

    def populate_from_history(self, messages: list[BaseMessage]) -> None:
        """Pre-populate seen IDs from existing message history.

        Used when resuming from a checkpoint to avoid replaying
        the full conversation history.

        Args:
            messages: List of messages from checkpoint state.
        """
        for msg in messages:
            self.mark_seen(msg)
