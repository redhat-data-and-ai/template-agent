"""History route for the template agent API.

This module provides endpoints for retrieving chat history from the database,
allowing users to view previous conversations and continue ongoing threads.
"""

from typing import List, Optional

import psycopg2
from fastapi import APIRouter, HTTPException, Query
from langchain_core.runnables import RunnableConfig

from template_agent.src.core.messages import langchain_to_chat_message
from template_agent.src.core.storage import get_shared_checkpointer
from template_agent.src.schema import ChatHistoryResponse, ChatMessage, ToolCall
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

router = APIRouter()

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


@router.get("/v1/history/{thread_id}")
async def history(
    thread_id: str,
    user_id: Optional[str] = Query(
        default=None, description="User ID for ownership verification"
    ),
) -> ChatHistoryResponse:
    """Get chat history for a specific thread.

    Args:
        thread_id: The unique identifier of the thread to retrieve history for.
        user_id: Optional user ID to verify thread ownership.

    Returns:
        A ChatHistoryResponse containing the list of chat messages for the thread.
    """
    logger.info(f"Retrieving history for thread_id: {thread_id}")

    chat_messages: List[ChatMessage] = []

    if settings.USE_INMEMORY_SAVER:
        logger.info(
            f"Using in-memory storage - retrieving history for thread_id: {thread_id}"
        )
        try:
            checkpointer = get_shared_checkpointer()
            config = RunnableConfig(
                configurable={"thread_id": thread_id, "checkpoint_ns": ""}
            )

            state_history = list(checkpointer.list(config))
            logger.info(
                f"Found {len(state_history)} checkpoints for thread_id: {thread_id}"
            )

            if len(state_history) == 0:
                logger.info(f"No checkpoints found for thread {thread_id}")
                return ChatHistoryResponse(messages=[])

            # Ownership check: verify user_id matches checkpoint metadata
            if user_id:
                latest = state_history[-1]
                checkpoint_metadata = getattr(latest, "metadata", None) or {}
                stored_user_id = checkpoint_metadata.get("user_id")
                if stored_user_id and stored_user_id != user_id:
                    logger.warning(
                        f"Ownership mismatch for thread {thread_id}: "
                        f"requested by {user_id}, owned by {stored_user_id}"
                    )
                    return ChatHistoryResponse(messages=[])

            # Read messages from the latest checkpoint
            latest_checkpoint = state_history[-1]
            if (
                latest_checkpoint.checkpoint
                and "channel_values" in latest_checkpoint.checkpoint
            ):
                channel_values = latest_checkpoint.checkpoint["channel_values"]
                if "messages" in channel_values:
                    messages = channel_values["messages"]
                    logger.info(f"Found {len(messages)} messages in latest checkpoint")
                    for message in messages:
                        try:
                            chat_message = langchain_to_chat_message(message)
                            chat_messages.append(chat_message)
                        except Exception as e:
                            logger.warning(
                                f"Could not convert message to ChatMessage: {e}"
                            )
                            continue

            # Fallback: collect from all checkpoints if latest had nothing
            if len(chat_messages) == 0:
                logger.info("Fallback: processing all checkpoints")
                for checkpoint_tuple in state_history:
                    if (
                        checkpoint_tuple.checkpoint
                        and "channel_values" in checkpoint_tuple.checkpoint
                    ):
                        channel_values = checkpoint_tuple.checkpoint["channel_values"]
                        if "messages" in channel_values:
                            for message in channel_values["messages"]:
                                try:
                                    chat_message = langchain_to_chat_message(message)
                                    is_duplicate = any(
                                        existing.type == chat_message.type
                                        and existing.content == chat_message.content
                                        for existing in chat_messages
                                    )
                                    if not is_duplicate:
                                        chat_messages.append(chat_message)
                                except Exception as e:
                                    logger.warning(f"Could not convert message: {e}")
                                    continue

            logger.info(
                f"Found {len(chat_messages)} messages for thread_id: {thread_id}"
            )
            return ChatHistoryResponse(messages=chat_messages)

        except Exception as e:
            logger.error(
                f"Error accessing in-memory storage for thread {thread_id}: {e}"
            )
            return ChatHistoryResponse(messages=[])

    try:
        with psycopg2.connect(settings.database_uri) as conn:
            cur = conn.cursor()

            # Ownership check for Postgres: verify user_id if provided
            if user_id:
                cur.execute(
                    "SELECT 1 FROM checkpoints WHERE thread_id = %s AND metadata->>'user_id' = %s LIMIT 1",
                    (thread_id, user_id),
                )
                if cur.fetchone() is None:
                    logger.warning(f"Thread {thread_id} not found for user {user_id}")
                    return ChatHistoryResponse(messages=[])

            # Get the latest checkpoint
            cur.execute(
                "SELECT checkpoint, metadata FROM checkpoints WHERE thread_id = %s ORDER BY checkpoint_id DESC LIMIT 1",
                (thread_id,),
            )
            latest_row = cur.fetchone()

            if latest_row:
                checkpoint_data, metadata = latest_row

                if checkpoint_data and "channel_values" in checkpoint_data:
                    channel_values = checkpoint_data["channel_values"]

                    if "messages" in channel_values:
                        checkpoint_messages = channel_values["messages"]
                        logger.info(
                            f"Found {len(checkpoint_messages)} messages in latest checkpoint"
                        )

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
                                logger.warning(
                                    f"Could not convert checkpoint message: {e}"
                                )
                                continue

                        logger.info(
                            f"Retrieved {len(chat_messages)} messages for thread_id: {thread_id}"
                        )
                        return ChatHistoryResponse(messages=chat_messages)

            # Fallback: process all checkpoints with writes
            logger.info(
                "Latest checkpoint didn't contain messages, falling back to writes"
            )
            cur.execute(
                "SELECT checkpoint, metadata FROM checkpoints WHERE thread_id = %s ORDER BY checkpoint_id ASC",
                (thread_id,),
            )
            rows = cur.fetchall()

            logger.info(f"Found {len(rows)} checkpoints for thread_id: {thread_id}")

            for row in rows:
                checkpoint_data, metadata = row
                run_id = metadata.get("run_id") if metadata else None
                session_id = metadata.get("session_id") if metadata else None

                writes = metadata.get("writes", {}) if metadata else {}
                if writes is None:
                    writes = {}

                messages = []
                if "__start__" in writes and "messages" in writes["__start__"]:
                    messages.extend(writes["__start__"]["messages"])
                if "agent" in writes and "messages" in writes["agent"]:
                    messages.extend(writes["agent"]["messages"])
                if "tools" in writes and "messages" in writes["tools"]:
                    messages.extend(writes["tools"]["messages"])

                for message_data in messages:
                    try:
                        if (
                            not isinstance(message_data, dict)
                            or "kwargs" not in message_data
                        ):
                            continue

                        kwargs = message_data.get("kwargs", {})
                        message_type = kwargs.get("type", "")
                        content = kwargs.get("content", "")
                        response_metadata = kwargs.get("response_metadata", {})

                        tool_calls = kwargs.get("tool_calls", [])
                        if not tool_calls and "additional_kwargs" in kwargs:
                            tool_calls = kwargs["additional_kwargs"].get(
                                "tool_calls", []
                            )

                        from langchain_core.messages import (
                            AIMessage,
                            HumanMessage,
                            ToolMessage,
                        )

                        if message_type == "human":
                            message = HumanMessage(content=content)
                        elif message_type == "ai":
                            message = AIMessage(
                                content=content,
                                tool_calls=tool_calls,
                                additional_kwargs={
                                    "response_metadata": response_metadata
                                },
                            )
                        elif message_type == "tool":
                            tool_call_id = kwargs.get("tool_call_id")
                            name = kwargs.get("name", "")
                            message = ToolMessage(
                                content=content,
                                tool_call_id=tool_call_id,
                                name=name,
                                additional_kwargs={
                                    "response_metadata": response_metadata
                                },
                            )
                        else:
                            continue

                        chat_message = langchain_to_chat_message(message)
                        if run_id:
                            chat_message.run_id = run_id
                        if thread_id:
                            chat_message.thread_id = thread_id
                        if session_id:
                            chat_message.session_id = session_id
                        if response_metadata:
                            chat_message.response_metadata = response_metadata

                        if tool_calls:
                            formatted_tool_calls = []
                            for tool_call in tool_calls:
                                if isinstance(tool_call, dict):
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
                            chat_message.tool_calls = formatted_tool_calls

                        chat_messages.append(chat_message)

                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        continue

            logger.info(
                f"Retrieved {len(chat_messages)} messages for thread_id: {thread_id}"
            )
            return ChatHistoryResponse(messages=chat_messages)

    except Exception as e:
        logger.error(
            f"Database error while fetching history for thread {thread_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve chat history: {str(e)}"
        )
