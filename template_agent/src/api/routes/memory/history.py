"""History route for the template agent API.

This module provides endpoints for retrieving chat history from the database,
allowing users to view previous conversations and continue ongoing threads.
"""

from typing import Dict, Optional

from fastapi import APIRouter, HTTPException

from template_agent.src.adapters.langchain import langchain_to_chat_message
from template_agent.src.agent.factory import get_template_agent
from template_agent.src.schema import ChatHistoryResponse, ChatMessage
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

router = APIRouter()

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def is_subagent_checkpoint(state) -> bool:
    """Check if checkpoint is from a subagent (should be skipped).

    Subagent checkpoints have either:
    - lc_agent_name in metadata (LangGraph internal field)
    - checkpoint_ns in configurable (namespace for subgraph)

    Args:
        state: State object from aget_state_history.

    Returns:
        True if this is a subagent checkpoint, False otherwise.
    """
    metadata = state.metadata or {}
    config = state.config or {}
    configurable = config.get("configurable", {})

    return bool(metadata.get("lc_agent_name") or configurable.get("checkpoint_ns"))


def rewrite_task_tool_calls(chat_message: ChatMessage) -> None:
    """Rewrite 'task' tool names to actual subagent names for better UI display.

    Modifies the chat_message.tool_calls in place to replace "task" with the
    actual subagent_type from args for better UI card display.

    Args:
        chat_message: ChatMessage to modify.
    """
    if not chat_message.tool_calls:
        return

    # Rewrite "task" tool name to actual subagent name
    for tc in chat_message.tool_calls:
        if tc.get("name") == "task" and "subagent_type" in tc.get("args", {}):
            tc["name"] = tc["args"]["subagent_type"]


def convert_with_metadata(
    message,
    idx: int,
    thread_id: str,
    metadata_map: Dict[int, Dict[str, Optional[str]]],
) -> Optional[ChatMessage]:
    """Convert message to ChatMessage and apply metadata if available.

    Args:
        message: LangChain message to convert.
        idx: Index of the message in the message list.
        thread_id: Thread ID to associate with the message.
        metadata_map: Mapping of message index to metadata dict.

    Returns:
        Converted ChatMessage or None if conversion fails.
    """
    try:
        chat_message = langchain_to_chat_message(message)
        chat_message.thread_id = thread_id

        # Apply metadata from the mapping if available
        if idx in metadata_map:
            msg_metadata = metadata_map[idx]
            chat_message.run_id = msg_metadata["run_id"]
            chat_message.trace_id = msg_metadata["trace_id"]
            chat_message.session_id = msg_metadata["session_id"]

        # Rewrite task tool calls for better UI display
        rewrite_task_tool_calls(chat_message)

        return chat_message
    except (ValueError, KeyError, AttributeError) as e:
        logger.warning(f"Could not convert message at index {idx}: {e}")
        return None


@router.get("/v1/users/{user_id}/history/{thread_id}")
async def history(
    user_id: str,
    thread_id: str,
) -> ChatHistoryResponse:
    """Get chat history for a specific thread.

    Args:
        user_id: User ID to verify thread ownership.
        thread_id: The unique identifier of the thread to retrieve history for.

    Returns:
        A ChatHistoryResponse containing the list of chat messages for the thread.
    """
    logger.info(f"Retrieving history for user_id={user_id}, thread_id={thread_id}")

    try:
        async with get_template_agent() as agent:
            config = {"configurable": {"thread_id": thread_id}}

            # Get state history using LangGraph's get_state_history
            state_history = [state async for state in agent.aget_state_history(config)]

            if not state_history:
                logger.info(f"No state history found for thread {thread_id}")
                return ChatHistoryResponse(messages=[])

            # Ownership check: verify thread belongs to user
            # Check metadata from newest checkpoint (first in list) to get the user_id
            first_state_metadata = state_history[0].metadata or {}
            thread_user_id = first_state_metadata.get("user_id")

            if thread_user_id != user_id:
                logger.warning(
                    f"Thread {thread_id} access denied for user {user_id} "
                    f"(belongs to {thread_user_id})"
                )
                return ChatHistoryResponse(messages=[])

            logger.debug(f"Ownership verified for thread {thread_id}")

            # Build message-to-metadata mapping by tracking message count changes
            # Process from oldest to newest (state_history is newest first, so reverse)
            message_to_metadata = {}
            previous_message_count = 0

            for state in reversed(state_history):
                # Skip subagent checkpoints
                if is_subagent_checkpoint(state):
                    continue

                checkpoint_metadata = state.metadata or {}

                # Get current messages in this state
                current_messages = state.values.get("messages", [])
                current_message_count = len(current_messages)

                # If message count increased, new messages were added
                if current_message_count > previous_message_count:
                    # Extract metadata once for all new messages
                    metadata_snapshot = {
                        "run_id": checkpoint_metadata.get("run_id"),
                        "trace_id": checkpoint_metadata.get("trace_id"),
                        "session_id": checkpoint_metadata.get("session_id"),
                    }

                    # Map each new message to this state's metadata
                    for idx in range(previous_message_count, current_message_count):
                        message_to_metadata[idx] = metadata_snapshot

                    logger.debug(
                        f"Messages[{previous_message_count}:{current_message_count}] → "
                        f"trace_id={metadata_snapshot['trace_id']}, "
                        f"run_id={metadata_snapshot['run_id']}"
                    )
                    previous_message_count = current_message_count

            # Get latest state for final message list (newest = first in list)
            latest_messages = state_history[0].values.get("messages", [])
            if not latest_messages:
                logger.info(f"No messages found in thread {thread_id}")
                return ChatHistoryResponse(messages=[])

            messages_with_metadata = len(message_to_metadata)
            messages_without_metadata = len(latest_messages) - messages_with_metadata
            logger.info(
                f"Processing {len(latest_messages)} messages "
                f"({messages_with_metadata} with metadata, "
                f"{messages_without_metadata} without)"
            )

            # Convert messages and apply metadata
            chat_messages = [
                msg
                for msg in (
                    convert_with_metadata(message, idx, thread_id, message_to_metadata)
                    for idx, message in enumerate(latest_messages)
                )
                if msg is not None
            ]

            logger.info(
                f"Retrieved {len(chat_messages)}/{len(latest_messages)} messages for thread {thread_id}"
            )
            return ChatHistoryResponse(messages=chat_messages)

    except Exception as e:
        logger.error(
            f"Error fetching history for user_id={user_id}, thread_id={thread_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")
