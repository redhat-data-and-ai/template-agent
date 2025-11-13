"""History route for the template agent API.

This module provides endpoints for retrieving chat history from the database,
allowing users to view previous conversations and continue ongoing threads.
"""

from typing import List

import psycopg2
from fastapi import APIRouter, HTTPException, Request
from langchain_core.runnables import RunnableConfig

from template_agent.src.core.agent_utils import langchain_to_chat_message
from template_agent.src.core.storage import get_shared_checkpointer
from template_agent.src.schema import ChatHistoryResponse, ChatMessage, ToolCall
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

router = APIRouter()

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


@router.get("/v1/history/{thread_id}")
async def history(thread_id: str, request: Request) -> ChatHistoryResponse:
    """Get chat history for a specific thread by reading from checkpoints table.

    This endpoint retrieves the complete conversation history for a given
    thread_id from the PostgreSQL database. When using in-memory storage,
    returns an empty history since conversations are not persisted.

    The function handles different message types (human, ai, tool) and
    converts them to the internal ChatMessage format for consistent
    representation across the application.

    Args:
        thread_id: The unique identifier of the thread to retrieve history for.
        request: The FastAPI request object, used to extract headers like
            X-Token for authentication.

    Returns:
        A ChatHistoryResponse containing the list of chat messages for the thread.
        Returns empty history when using in-memory storage.
        If there's an error, returns an empty message list instead of raising
        an exception.

    Note:
        - Messages are extracted from multiple locations in the checkpoint data
        - The function handles various message formats and gracefully skips
          invalid messages
        - Authentication tokens are logged but not currently used for validation
        - In-memory storage mode returns empty history as conversations are not persisted
    """
    access_token = request.headers.get("X-Token")
    logger.info(f"Retrieving history for thread_id: {thread_id}")
    logger.info(f"Access token present: {access_token is not None}")

    chat_messages: List[ChatMessage] = []

    # When using in-memory storage, get history from shared checkpointer
    if settings.USE_INMEMORY_SAVER:
        logger.info(
            f"Using in-memory storage - retrieving history from checkpointer for thread_id: {thread_id}"
        )
        try:
            checkpointer = get_shared_checkpointer()

            # Create a config for this thread (matching the format used by the agent)
            config = RunnableConfig(
                configurable={"thread_id": thread_id, "checkpoint_ns": ""}
            )

            # Get all checkpoints for this thread to understand the structure
            state_history = list(checkpointer.list(config))
            logger.info(
                f"Found {len(state_history)} checkpoints for thread_id: {thread_id}"
            )

            if len(state_history) == 0:
                logger.info(
                    f"No checkpoints found for thread {thread_id} - this means no conversations have happened in this thread yet."
                )
            else:
                # DEBUG: Log structure of all checkpoints to understand how messages are stored
                for i, checkpoint_tuple in enumerate(state_history):
                    logger.info(f"=== CHECKPOINT {i} DEBUG ===")
                    logger.info(
                        f"Checkpoint keys: {list(checkpoint_tuple.checkpoint.keys()) if checkpoint_tuple.checkpoint else 'None'}"
                    )

                    if (
                        checkpoint_tuple.checkpoint
                        and "channel_values" in checkpoint_tuple.checkpoint
                    ):
                        channel_values = checkpoint_tuple.checkpoint["channel_values"]
                        logger.info(
                            f"Channel values keys: {list(channel_values.keys())}"
                        )

                        if "messages" in channel_values:
                            messages = channel_values["messages"]
                            logger.info(
                                f"Messages count in checkpoint {i}: {len(messages)}"
                            )
                            for j, msg in enumerate(messages):
                                msg_type = (
                                    getattr(msg, "type", "unknown")
                                    if hasattr(msg, "type")
                                    else type(msg).__name__
                                )
                                msg_content = (
                                    getattr(msg, "content", str(msg)[:100])
                                    if hasattr(msg, "content")
                                    else str(msg)[:100]
                                )
                                logger.info(
                                    f"  Message {j}: {msg_type} - {msg_content}"
                                )
                        else:
                            logger.info(
                                f"No 'messages' key in channel_values for checkpoint {i}"
                            )
                    else:
                        logger.info(f"No channel_values in checkpoint {i}")

                # Try the latest checkpoint first (our current approach)
                latest_checkpoint = state_history[-1]
                logger.info(
                    f"=== PROCESSING LATEST CHECKPOINT (index {len(state_history) - 1}) ==="
                )

                if (
                    latest_checkpoint.checkpoint
                    and "channel_values" in latest_checkpoint.checkpoint
                ):
                    channel_values = latest_checkpoint.checkpoint["channel_values"]
                    if "messages" in channel_values:
                        messages = channel_values["messages"]
                        logger.info(
                            f"Found {len(messages)} messages in latest checkpoint"
                        )
                        for message in messages:
                            try:
                                chat_message = langchain_to_chat_message(message)
                                chat_messages.append(chat_message)
                                logger.info(
                                    f"Added message: {chat_message.type} - {chat_message.content[:50]}..."
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Could not convert message to ChatMessage: {e}"
                                )
                                continue

                # If latest checkpoint approach didn't work, try collecting from all checkpoints
                if len(chat_messages) == 0:
                    logger.info("=== FALLBACK: PROCESSING ALL CHECKPOINTS ===")
                    for i, checkpoint_tuple in enumerate(state_history):
                        if (
                            checkpoint_tuple.checkpoint
                            and "channel_values" in checkpoint_tuple.checkpoint
                        ):
                            channel_values = checkpoint_tuple.checkpoint[
                                "channel_values"
                            ]
                            if "messages" in channel_values:
                                messages = channel_values["messages"]
                                logger.info(
                                    f"Processing {len(messages)} messages from checkpoint {i}"
                                )
                                for message in messages:
                                    try:
                                        chat_message = langchain_to_chat_message(
                                            message
                                        )
                                        # Check for duplicates before adding
                                        is_duplicate = False
                                        for existing_msg in chat_messages:
                                            if (
                                                existing_msg.type == chat_message.type
                                                and existing_msg.content
                                                == chat_message.content
                                            ):
                                                is_duplicate = True
                                                break

                                        if not is_duplicate:
                                            chat_messages.append(chat_message)
                                            logger.info(
                                                f"Added unique message: {chat_message.type} - {chat_message.content[:50]}..."
                                            )
                                        else:
                                            logger.info(
                                                f"Skipped duplicate message: {chat_message.type} - {chat_message.content[:50]}..."
                                            )
                                    except Exception as e:
                                        logger.warning(
                                            f"Could not convert message to ChatMessage: {e}"
                                        )
                                        continue

            logger.info(
                f"Found {len(chat_messages)} messages in memory for thread_id: {thread_id}"
            )

            return ChatHistoryResponse(messages=chat_messages)
        except Exception as e:
            logger.error(
                f"Error accessing in-memory storage for thread {thread_id}: {e}"
            )
            return ChatHistoryResponse(messages=[])

    try:
        # Connect to PostgreSQL and read from checkpoints table
        with psycopg2.connect(settings.database_uri) as conn:
            cur = conn.cursor()

            # Query the checkpoints table for the specific thread_id
            # Get the latest checkpoint first (which should contain complete conversation state)
            cur.execute(
                "SELECT checkpoint, metadata FROM checkpoints WHERE thread_id = %s ORDER BY checkpoint_id DESC LIMIT 1",
                (thread_id,),
            )
            latest_row = cur.fetchone()

            if latest_row:
                logger.info(f"Found latest checkpoint for thread_id: {thread_id}")
                checkpoint_data, metadata = latest_row

                # DEBUG: Log the structure of the latest checkpoint
                logger.info("=== POSTGRESQL LATEST CHECKPOINT DEBUG ===")
                logger.info(
                    f"Checkpoint_data keys: {list(checkpoint_data.keys()) if checkpoint_data else 'None'}"
                )
                logger.info(
                    f"Metadata keys: {list(metadata.keys()) if metadata else 'None'}"
                )

                # Try to get complete conversation from latest checkpoint
                if checkpoint_data and "channel_values" in checkpoint_data:
                    channel_values = checkpoint_data["channel_values"]
                    logger.info(f"Channel values keys: {list(channel_values.keys())}")

                    if "messages" in channel_values:
                        checkpoint_messages = channel_values["messages"]
                        logger.info(
                            f"Found {len(checkpoint_messages)} messages in latest checkpoint channel_values"
                        )

                        # DEBUG: Log each message structure
                        for i, msg in enumerate(checkpoint_messages):
                            msg_type = (
                                getattr(msg, "type", "unknown")
                                if hasattr(msg, "type")
                                else type(msg).__name__
                            )
                            msg_content = (
                                getattr(msg, "content", str(msg)[:100])
                                if hasattr(msg, "content")
                                else str(msg)[:100]
                            )
                            logger.info(
                                f"  PostgreSQL Message {i}: {msg_type} - {msg_content}"
                            )

                        # Extract metadata for tracking
                        run_id = metadata.get("run_id") if metadata else None
                        session_id = metadata.get("session_id") if metadata else None
                        user_id = metadata.get("user_id") if metadata else None

                        # Convert LangChain messages directly (like in-memory version)
                        for message in checkpoint_messages:
                            try:
                                chat_message = langchain_to_chat_message(message)
                                # Set metadata from checkpoint for tracking
                                if run_id:
                                    chat_message.run_id = run_id
                                if thread_id:
                                    chat_message.thread_id = thread_id
                                if session_id:
                                    chat_message.session_id = session_id
                                chat_messages.append(chat_message)
                                logger.info(
                                    f"Successfully converted checkpoint message: {chat_message.type} - {chat_message.content[:50]}..."
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Could not convert checkpoint message to ChatMessage: {e}"
                                )
                                continue

                        logger.info(
                            f"Retrieved {len(chat_messages)} messages from latest checkpoint for thread_id: {thread_id}"
                        )
                        return ChatHistoryResponse(messages=chat_messages)
                    else:
                        logger.info(
                            "No 'messages' key found in channel_values of latest checkpoint"
                        )
                else:
                    logger.info("No 'channel_values' found in latest checkpoint_data")

            # Fallback: If latest checkpoint doesn't have messages, process all checkpoints with writes
            logger.info(
                "Latest checkpoint didn't contain messages, falling back to processing all checkpoints"
            )
            cur.execute(
                "SELECT checkpoint, metadata FROM checkpoints WHERE thread_id = %s ORDER BY checkpoint_id ASC",
                (thread_id,),
            )
            rows = cur.fetchall()

            logger.info(f"Found {len(rows)} checkpoints for thread_id: {thread_id}")

            total_messages_found = 0

            # Process each checkpoint to extract messages from writes (fallback approach)
            for row in rows:
                checkpoint_data, metadata = row

                # Extract run_id, thread_id, session_id from metadata for tracking
                run_id = metadata.get("run_id") if metadata else None
                session_id = metadata.get("session_id") if metadata else None
                user_id = metadata.get("user_id") if metadata else None

                logger.info(
                    f"Processing checkpoint with run_id: {run_id}, session_id: {session_id}, user_id: {user_id}"
                )

                # Get messages from metadata.writes (original logic)
                messages = []
                writes = metadata.get("writes", {}) if metadata else {}

                # Handle case where writes might be None
                if writes is None:
                    writes = {}
                    logger.info("Writes is None, using empty dict")

                # Check for messages in different write locations
                if "__start__" in writes and "messages" in writes["__start__"]:
                    messages.extend(writes["__start__"]["messages"])
                    logger.info(
                        f"Found {len(writes['__start__']['messages'])} messages in __start__"
                    )
                if "agent" in writes and "messages" in writes["agent"]:
                    messages.extend(writes["agent"]["messages"])
                    logger.info(
                        f"Found {len(writes['agent']['messages'])} messages in agent"
                    )
                if "tools" in writes and "messages" in writes["tools"]:
                    messages.extend(writes["tools"]["messages"])
                    logger.info(
                        f"Found {len(writes['tools']['messages'])} messages in tools"
                    )

                total_messages_found += len(messages)

                # Convert each message to ChatMessage format
                for message_data in messages:
                    try:
                        logger.info(f"Processing message_data: {message_data}")

                        # Validate message format - should be a dict with kwargs
                        if (
                            not isinstance(message_data, dict)
                            or "kwargs" not in message_data
                        ):
                            logger.info(
                                f"Skipping invalid message format: {message_data}"
                            )
                            continue

                        # Extract message components
                        kwargs = message_data.get("kwargs", {})
                        message_type = kwargs.get("type", "")
                        content = kwargs.get("content", "")
                        response_metadata = kwargs.get("response_metadata", {})

                        # Handle tool calls from both direct kwargs and additional_kwargs
                        tool_calls = kwargs.get("tool_calls", [])
                        if not tool_calls and "additional_kwargs" in kwargs:
                            tool_calls = kwargs["additional_kwargs"].get(
                                "tool_calls", []
                            )

                        logger.info(f"Message type: {message_type}, content: {content}")

                        # Import here to avoid circular imports
                        from langchain_core.messages import (
                            AIMessage,
                            HumanMessage,
                            ToolMessage,
                        )

                        # Create appropriate LangChain message based on type
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
                            logger.info(
                                f"Skipping unknown message type: {message_type}"
                            )
                            continue

                        # Convert to internal ChatMessage format
                        chat_message = langchain_to_chat_message(message)

                        # Set metadata from checkpoint for tracking
                        if run_id:
                            chat_message.run_id = run_id
                        if thread_id:
                            chat_message.thread_id = thread_id
                        if session_id:
                            chat_message.session_id = session_id

                        # Set metadata from the original message data
                        if response_metadata:
                            chat_message.response_metadata = response_metadata

                        # Set tool calls if present
                        if tool_calls:
                            # Ensure tool calls have the correct structure
                            formatted_tool_calls = []
                            for tool_call in tool_calls:
                                if isinstance(tool_call, dict):
                                    # Ensure required fields are present and properly typed
                                    if "name" in tool_call and "args" in tool_call:
                                        # Create a proper ToolCall object
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
                            logger.info(
                                f"Added {len(formatted_tool_calls)} tool calls to message"
                            )

                        logger.info(
                            f"Successfully converted message: {chat_message.type} - {chat_message.content[:50]}..."
                        )
                        logger.info(
                            "Message metadata: "
                            f"tool_calls={bool(chat_message.tool_calls)}, "
                            f"response_metadata={bool(chat_message.response_metadata)}"
                        )
                        chat_messages.append(chat_message)

                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        continue

            logger.info(
                f"Retrieved {len(chat_messages)} messages for thread_id: {thread_id}"
            )
            logger.info(f"Total messages found: {total_messages_found}")
            logger.info(f"Final chat_messages: {[msg.type for msg in chat_messages]}")
            return ChatHistoryResponse(messages=chat_messages)

    except Exception as e:
        logger.error(
            f"Database error while fetching history for thread {thread_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve chat history: {str(e)}"
        )
