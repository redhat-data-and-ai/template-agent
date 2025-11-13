"""Agent Manager for the template agent system.

This module provides the AgentManager class that orchestrates agent operations,
handles streaming responses, and manages the conversion between LangGraph events
and simplified streaming.
"""

import inspect
from collections.abc import AsyncGenerator
from typing import Any, Dict
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langfuse.callback import CallbackHandler
from langgraph.pregel import Pregel
from langgraph.types import Command, Interrupt

from template_agent.src.core.agent import get_template_agent
from template_agent.src.core.agent_utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)
from template_agent.src.core.storage import register_thread
from template_agent.src.schema import StreamRequest
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler(
    trace_name="template-agent", environment=settings.LANGFUSE_TRACING_ENVIRONMENT
)

app_logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


class AgentManager:
    """Manager class for handling agent operations and streaming responses.

    This class provides a simplified interface for agent interactions while
    preserving all enterprise features like authentication, tracing, and
    error handling from the original implementation.
    """

    def __init__(self, redhat_sso_token: str | None = None):
        """Initialize the AgentManager.

        Args:
            redhat_sso_token: Optional SSO token for enterprise authentication.
        """
        self.redhat_sso_token = redhat_sso_token
        self._agent: Pregel | None = None
        self._current_tool_call_id: str | None = None  # Track current active tool call

    async def stream_response(
        self, request: StreamRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream agent response with simplified event structure.

        This method provides streaming functionality while ensuring that conversation
        state is saved only once at the end, not during intermediate streaming.

        Args:
            request: The streaming request containing user input and configuration.

        Yields:
            Simplified event dictionaries with 'type' and 'content' fields.
        """
        # Use persistent agent for both streaming and state persistence
        # This ensures LangGraph handles state management automatically
        async with get_template_agent(
            self.redhat_sso_token, enable_checkpointing=True
        ) as persistent_agent:
            try:
                # Prepare input for the persistent agent
                kwargs, run_id, thread_id = await self._handle_input(
                    request, persistent_agent
                )

                app_logger.info(
                    f"AgentManager streaming response for run_id: {run_id}, thread_id: {thread_id}"
                )

                # Reset tool call tracking for this stream
                self._current_tool_call_id = None

                # Use persistent agent for streaming - LangGraph will handle state automatically
                async for stream_event in persistent_agent.astream(
                    **kwargs, stream_mode=["updates", "messages", "custom"]
                ):
                    if not isinstance(stream_event, tuple):
                        continue

                    stream_mode, event = stream_event

                    # Update tool call tracking based on stream events
                    self._update_tool_call_tracking(stream_mode, event)

                    # Convert LangGraph events to simplified format
                    effective_session_id = request.session_id or thread_id
                    formatted_events = self._format_events(
                        stream_mode,
                        event,
                        request.stream_tokens,
                        run_id,
                        thread_id,
                        effective_session_id,
                    )

                    for formatted_event in formatted_events:
                        if formatted_event:
                            yield formatted_event

                # No manual state saving needed - LangGraph handles this automatically
                app_logger.info(
                    f"Conversation completed and auto-saved for thread {thread_id}"
                )

            except Exception as e:
                app_logger.error(f"Error in AgentManager stream_response: {e}")
                yield {
                    "type": "error",
                    "content": {
                        "message": "Internal server error",
                        "recoverable": False,
                        "error_type": "agent_error",
                    },
                }

    async def _handle_input(
        self, request: StreamRequest, agent: Pregel
    ) -> tuple[Dict[str, Any], str, str]:
        """Handle input preparation and configuration (preserving existing logic)."""
        run_id = uuid4()

        # Generate default thread_id if not provided
        thread_id = request.thread_id
        if thread_id is None:
            thread_id = str(uuid4())
            app_logger.info(
                f"Assigning auto-generated thread_id '{thread_id}' as thread_id is missing in user request"
            )

        # Configure tracing and session management (preserved from original)
        # If session_id is not provided, use thread_id as session_id
        effective_session_id = request.session_id or thread_id
        effective_user_id = request.user_id or "anonymous"

        # Register thread for user (for in-memory storage tracking)
        if settings.USE_INMEMORY_SAVER:
            register_thread(effective_user_id, thread_id)

        # Generate AI call ID
        ai_call_id = f"ai_call_{str(uuid4())}"

        configurable = {
            "thread_id": thread_id,
            "session_id": effective_session_id,
            "run_id": str(run_id),
            "user_id": effective_user_id,
            "ai_call_id": ai_call_id,
            "langfuse_session_id": effective_session_id,
            "langfuse_user_id": effective_user_id,
            "langfuse_observation_id": thread_id,
        }

        config = RunnableConfig(
            configurable=configurable,
            run_id=run_id,
            callbacks=[langfuse_handler],
        )

        # Check for interrupts that need to be resumed (preserved from original)
        state = await agent.aget_state(config=config)
        interrupted_tasks = [
            task
            for task in state.tasks
            if hasattr(task, "interrupts") and task.interrupts
        ]

        # Prepare input based on whether we're resuming from an interrupt
        user_input_message: Command | Dict[str, Any]
        if interrupted_tasks:
            user_input_message = Command(resume=request.message)
        else:
            user_input_message = {"messages": [HumanMessage(content=request.message)]}

        kwargs = {
            "input": user_input_message,
            "config": config,
        }

        app_logger.info(
            f"AgentManager configured with run_id: {run_id}, thread_id: {thread_id}, session_id: {effective_session_id}"
        )
        return kwargs, str(run_id), thread_id

    async def _prepare_streaming_input_with_history(
        self, request: StreamRequest, existing_state, run_id: str, thread_id: str
    ) -> Dict[str, Any]:
        """Prepare streaming input with conversation history for non-checkpointing agent."""
        from langchain_core.messages import HumanMessage
        from langchain_core.runnables import RunnableConfig

        # Get existing messages from state
        existing_messages = existing_state.values.get("messages", [])

        # Create new message list with history + current user message
        all_messages = list(existing_messages)
        all_messages.append(HumanMessage(content=request.message))

        # Configure for streaming agent (no checkpointing)
        effective_session_id = request.session_id or thread_id
        effective_user_id = request.user_id or "anonymous"

        configurable = {
            "thread_id": thread_id,
            "session_id": effective_session_id,
            "run_id": run_id,
            "user_id": effective_user_id,
            "langfuse_session_id": effective_session_id,
            "langfuse_user_id": effective_user_id,
            "langfuse_observation_id": thread_id,
        }

        config = RunnableConfig(
            configurable=configurable,
            run_id=run_id,
            callbacks=[langfuse_handler],
        )

        return {
            "input": {"messages": all_messages},
            "config": config,
        }

    async def _save_final_conversation_state(
        self, persistent_agent, config, all_messages: list, thread_id: str
    ) -> None:
        """Save the final conversation state once after streaming completes."""
        try:
            app_logger.info(
                f"Saving {len(all_messages)} messages for thread {thread_id}"
            )

            # Log message types for debugging
            message_types = [
                getattr(msg, "type", type(msg).__name__) for msg in all_messages
            ]
            app_logger.info(f"Message types being saved: {message_types}")

            # Update the persistent agent's state with all messages
            await persistent_agent.aupdate_state(
                config=config, values={"messages": all_messages}
            )
            app_logger.info(
                f"Successfully saved conversation state for thread {thread_id}"
            )

        except Exception as e:
            app_logger.error(f"Error saving final conversation state: {e}")
            # Don't re-raise - streaming already completed successfully

    def _format_events(
        self,
        stream_mode: str,
        event: Any,
        stream_tokens: bool,
        run_id: str,
        thread_id: str,
        session_id: str | None,
    ) -> list[Dict[str, Any]]:
        """Convert LangGraph events to simplified streaming format.

        This method implements the proposed event format while preserving
        all the business logic from the original implementation.
        """
        formatted_events = []

        if stream_mode == "updates":
            formatted_events.extend(
                self._handle_update_events(event, run_id, thread_id, session_id)
            )
        elif stream_mode == "messages" and stream_tokens:
            token_event = self._handle_token_events(event)
            if token_event:
                formatted_events.append(token_event)
        elif stream_mode == "custom":
            custom_event = self._handle_custom_events(
                event, run_id, thread_id, session_id
            )
            if custom_event:
                formatted_events.append(custom_event)

        return formatted_events

    def _handle_update_events(
        self, event: Dict[str, Any], run_id: str, thread_id: str, session_id: str | None
    ) -> list[Dict[str, Any]]:
        """Handle update events from LangGraph (preserving existing logic)."""
        formatted_events = []
        new_messages = []

        for node, updates in event.items():
            # Handle agent interrupts with structured messages (preserved)
            if node == "__interrupt__":
                interrupt: Interrupt
                for interrupt in updates:
                    new_messages.append(AIMessage(content=interrupt.value))
                continue

            updates = updates or {}
            update_messages = updates.get("messages", [])

            # Special cases for using langgraph-supervisor library (preserved)
            if node == "supervisor":
                ai_messages = [
                    msg for msg in update_messages if isinstance(msg, AIMessage)
                ]
                if ai_messages:
                    update_messages = [ai_messages[-1]]

            if node in ("research_expert", "math_expert"):
                # Convert sub-agent output to ToolMessage for UI display (preserved)
                msg = ToolMessage(
                    content=update_messages[0].content,
                    name=node,
                    tool_call_id="",
                )
                update_messages = [msg]

            new_messages.extend(update_messages)

        # Process messages and convert to simplified format
        processed_messages = self._process_message_tuples(new_messages)

        for message in processed_messages:
            try:
                chat_message = langchain_to_chat_message(message)
                chat_message.run_id = run_id

                # Convert to simplified format
                formatted_event = {
                    "type": "message",
                    "content": self._convert_chat_message_to_simple_format(
                        chat_message, thread_id, session_id
                    ),
                }
                formatted_events.append(formatted_event)

            except Exception as e:
                app_logger.error(f"Error formatting message: {e}")
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

    def _handle_token_events(self, event: tuple) -> Dict[str, Any] | None:
        """Handle token streaming events with tool call ID tracking."""
        msg, metadata = event
        if "skip_stream" in metadata.get("tags", []):
            return None

        # Filter out non-LLM node messages (preserved logic)
        if not isinstance(msg, AIMessageChunk):
            return None

        content = remove_tool_calls(msg.content)
        if content:
            token_event = {
                "type": "token",
                "content": convert_message_content_to_string(content),
            }

            # Add tool call ID if this token is part of a tool call response
            tool_call_id = (
                self._extract_tool_call_id_from_message(msg)
                or self._current_tool_call_id
            )
            if tool_call_id:
                token_event["tool_call_id"] = tool_call_id

            return token_event
        return None

    def _handle_custom_events(
        self, event: Any, run_id: str, thread_id: str, session_id: str | None
    ) -> Dict[str, Any] | None:
        """Handle custom events from LangGraph."""
        try:
            chat_message = langchain_to_chat_message(event)
            chat_message.run_id = run_id

            return {
                "type": "message",
                "content": self._convert_chat_message_to_simple_format(
                    chat_message, thread_id, session_id
                ),
            }
        except Exception as e:
            app_logger.error(f"Error handling custom event: {e}")
            return None

    def _process_message_tuples(self, new_messages: list) -> list:
        """Process LangGraph streaming tuples and accumulate message parts (preserved logic)."""
        processed_messages = []
        current_message: Dict[str, Any] = {}

        for message in new_messages:
            if isinstance(message, tuple):
                key, value = message
                current_message[key] = value
            else:
                # Add complete message if we have one in progress
                if current_message:
                    processed_messages.append(self._create_ai_message(current_message))
                    current_message = {}
                processed_messages.append(message)

        # Add any remaining message parts
        if current_message:
            processed_messages.append(self._create_ai_message(current_message))

        return processed_messages

    def _create_ai_message(self, parts: Dict[str, Any]) -> AIMessage:
        """Create an AIMessage from a dictionary of parts (preserved from original)."""
        sig = inspect.signature(AIMessage)
        valid_keys = set(sig.parameters)
        filtered = {k: v for k, v in parts.items() if k in valid_keys}
        return AIMessage(**filtered)

    def _convert_chat_message_to_simple_format(
        self, chat_message, thread_id: str, session_id: str | None
    ) -> Dict[str, Any]:
        """Convert ChatMessage to simplified content format for the proposed API."""
        content = {
            "type": chat_message.type,
            "content": chat_message.content,
        }

        # Add optional fields only if present
        if chat_message.tool_calls:
            content["tool_calls"] = chat_message.tool_calls
        if chat_message.tool_call_id:
            content["tool_call_id"] = chat_message.tool_call_id
        if chat_message.run_id:
            content["run_id"] = chat_message.run_id
        if thread_id:
            content["thread_id"] = thread_id
        if session_id:
            content["session_id"] = session_id
        if chat_message.ai_call_id:
            content["ai_call_id"] = chat_message.ai_call_id
        if chat_message.response_metadata:
            content["response_metadata"] = chat_message.response_metadata
        if chat_message.custom_data:
            content["custom_data"] = chat_message.custom_data

        return content

    def _extract_tool_call_id_from_message(self, msg: AIMessageChunk) -> str | None:
        """Extract tool call ID from an AIMessageChunk if available.

        Args:
            msg: The AIMessageChunk to extract tool call ID from

        Returns:
            The tool call ID if available, None otherwise
        """
        try:
            # Check if the message has tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                # Return the ID of the first tool call
                return msg.tool_calls[0].get("id")

            # Check if the message has tool_call_chunks (streaming tool calls)
            if hasattr(msg, "tool_call_chunks") and msg.tool_call_chunks:
                # Return the ID of the first tool call chunk
                return msg.tool_call_chunks[0].get("id")

            # Check if this is a response to a tool call (has tool_call_id)
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                return msg.tool_call_id

            return None
        except (AttributeError, IndexError, KeyError) as e:
            app_logger.debug(f"Could not extract tool call ID from message: {e}")
            return None

    def _update_tool_call_tracking(self, stream_mode: str, event: Any) -> None:
        """Update the current tool call ID based on streaming events.

        Args:
            stream_mode: The type of stream event
            event: The event data
        """
        try:
            if stream_mode == "updates":
                # Look for tool calls in update events
                for node, updates in event.items():
                    if updates and "messages" in updates:
                        for message in updates["messages"]:
                            if hasattr(message, "tool_calls") and message.tool_calls:
                                # Found a new tool call, update tracking
                                self._current_tool_call_id = message.tool_calls[0].get(
                                    "id"
                                )
                                app_logger.debug(
                                    f"Tracking tool call ID: {self._current_tool_call_id}"
                                )
                                return
                            elif (
                                hasattr(message, "tool_call_id")
                                and message.tool_call_id
                            ):
                                # This is a tool response, track its ID
                                self._current_tool_call_id = message.tool_call_id
                                app_logger.debug(
                                    f"Tracking tool response ID: {self._current_tool_call_id}"
                                )
                                return

            elif stream_mode == "messages":
                # Check message stream for tool calls
                msg, metadata = event
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    self._current_tool_call_id = msg.tool_calls[0].get("id")
                    app_logger.debug(
                        f"Tracking tool call ID from message: {self._current_tool_call_id}"
                    )
                elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    self._current_tool_call_id = msg.tool_call_id
                    app_logger.debug(
                        f"Tracking tool response ID from message: {self._current_tool_call_id}"
                    )

        except Exception as e:
            app_logger.debug(f"Error updating tool call tracking: {e}")
            # Don't fail streaming due to tracking issues
