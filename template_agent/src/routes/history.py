"""History route for the template agent API.

This module provides endpoints for retrieving chat history from the database,
allowing users to view previous conversations and continue ongoing threads.
"""

from typing import List

from fastapi import APIRouter, HTTPException
from langchain_core.runnables import RunnableConfig

from template_agent.src.core.checkpointer import get_checkpointer
from template_agent.src.core.messages import langchain_to_chat_message
from template_agent.src.schema import ChatHistoryResponse, ChatMessage
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

router = APIRouter()

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


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

    chat_messages: List[ChatMessage] = []

    try:
        async with get_checkpointer() as checkpointer:
            config = RunnableConfig(configurable={"thread_id": thread_id})

            # Ownership check: verify thread belongs to user
            async with checkpointer.conn.cursor() as cur:
                await cur.execute(
                    "SELECT 1 FROM checkpoints WHERE thread_id = %s AND metadata->>'user_id' = %s LIMIT 1",
                    (thread_id, user_id),
                )
                row = await cur.fetchone()
                if row is None:
                    logger.warning(
                        f"Thread {thread_id} not found or access denied for user {user_id}"
                    )
                    return ChatHistoryResponse(messages=[])

            # Get the latest checkpoint using checkpointer API (decodes msgpack blobs)
            checkpoint_data = await checkpointer.aget(config)

            if checkpoint_data and "channel_values" in checkpoint_data:
                channel_values = checkpoint_data["channel_values"]

                if "messages" in channel_values:
                    checkpoint_messages = channel_values["messages"]
                    logger.info(
                        f"Found {len(checkpoint_messages)} messages in latest checkpoint"
                    )

                    # Get metadata for run_id and session_id
                    metadata_tuple = await checkpointer.aget_tuple(config)
                    metadata = metadata_tuple.metadata if metadata_tuple else {}
                    run_id = metadata.get("run_id") if metadata else None
                    session_id = metadata.get("session_id") if metadata else None

                    for message in checkpoint_messages:
                        try:
                            chat_message = langchain_to_chat_message(message)
                            if run_id:
                                chat_message.run_id = run_id
                            if thread_id:
                                chat_message.thread_id = thread_id
                            if session_id:
                                chat_message.session_id = session_id
                            chat_messages.append(chat_message)
                        except Exception as e:
                            logger.warning(f"Could not convert checkpoint message: {e}")
                            continue

                    logger.info(
                        f"Retrieved {len(chat_messages)} messages for user_id={user_id}, thread_id={thread_id}"
                    )
                    return ChatHistoryResponse(messages=chat_messages)

            # No messages found in latest checkpoint
            logger.info(
                f"No messages found for user_id={user_id}, thread_id={thread_id}"
            )
            return ChatHistoryResponse(messages=[])

    except Exception as e:
        logger.error(
            f"Database error while fetching history for user_id={user_id}, thread_id={thread_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve chat history: {str(e)}"
        )
