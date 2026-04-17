"""Agent Manager for the template agent system.

This module provides the AgentManager class that orchestrates agent operations,
handles streaming responses, and coordinates the streaming pipeline components.
"""

from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
from langgraph.types import Command

from template_agent.src.core.agent import get_template_agent
from template_agent.src.core.storage import register_thread
from template_agent.src.core.streaming import (
    MessageDeduplicator,
    StreamContext,
    TokenEventHandler,
    ToolCallTracker,
    UpdateEventHandler,
)
from template_agent.src.schema import StreamRequest
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


class AgentManager:
    """Manager class for handling agent operations and streaming responses.

    Orchestrates the streaming pipeline using modular components for:
    - Message deduplication across checkpoints
    - Tool call tracking for UI feedback
    - Event handling and formatting
    - Authentication and tracing
    """

    def __init__(self, redhat_sso_token: str | None = None):
        """Initialize the AgentManager.

        Args:
            redhat_sso_token: Optional SSO token for MCP authentication.
        """
        self.redhat_sso_token = redhat_sso_token

        # Initialize streaming components
        self.deduplicator = MessageDeduplicator()
        self.tracker = ToolCallTracker()

        # Register event handlers
        self.handlers: dict[str, UpdateEventHandler | TokenEventHandler] = {
            "updates": UpdateEventHandler(self.deduplicator),
            "messages": TokenEventHandler(self.tracker),
        }

    async def stream_response(
        self, request: StreamRequest
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream agent response with simplified event structure.

        LangGraph automatically handles state persistence at the end of streaming.

        Args:
            request: The streaming request containing user input and configuration.

        Yields:
            Simplified event dictionaries with 'type' and 'content' fields.
        """
        async with get_template_agent(self.redhat_sso_token) as agent:
            try:
                # Reset per-stream state
                self.deduplicator.reset()
                self.tracker.reset()

                # Prepare input and configuration
                config, ctx = await self._prepare_stream(request, agent)

                logger.info(
                    f"Streaming response for run_id={ctx.run_id}, thread_id={ctx.thread_id}"
                )

                # Stream events from agent
                async for stream_event in agent.astream(
                    **config, stream_mode=["updates", "messages"]
                ):
                    if not isinstance(stream_event, tuple):
                        continue

                    stream_mode, event = stream_event

                    # Update tool call tracking
                    self.tracker.update_from_stream_event(stream_mode, event)

                    # Route to appropriate handler
                    handler = self.handlers.get(stream_mode)
                    if not handler:
                        continue

                    # Process and yield formatted events
                    formatted_events = handler.handle(event, ctx)
                    for formatted_event in formatted_events:
                        if formatted_event:
                            yield formatted_event

                logger.info(f"Conversation auto-saved for thread {ctx.thread_id}")

            except Exception as e:
                logger.error(f"Error in stream_response: {e}", exc_info=True)
                yield {
                    "type": "error",
                    "content": {
                        "message": "Internal server error",
                        "recoverable": False,
                        "error_type": "agent_error",
                    },
                }

    async def _prepare_stream(
        self, request: StreamRequest, agent
    ) -> tuple[dict[str, Any], StreamContext]:
        """Prepare streaming configuration and context.

        Args:
            request: The stream request.
            agent: The agent instance.

        Returns:
            Tuple of (config dict for astream, StreamContext).
        """
        run_id = uuid4()
        thread_id = request.thread_id or str(uuid4())

        if not request.thread_id:
            logger.info(f"Auto-generated thread_id: {thread_id}")

        effective_session_id = request.session_id or thread_id
        effective_user_id = request.user_id or "anonymous"

        # Register thread for in-memory storage
        if settings.USE_INMEMORY_SAVER:
            register_thread(effective_user_id, thread_id)

        # Configure Langfuse tracing
        langfuse_handler = CallbackHandler()
        config = RunnableConfig(
            configurable={
                "thread_id": thread_id,
                "langfuse_session_id": effective_session_id,
                "run_id": str(run_id),
                "langfuse_user_id": effective_user_id,
            },
            run_id=run_id,
            run_name="template-agent",
            callbacks=[langfuse_handler],
        )

        # Get current state and pre-populate seen messages
        state = await agent.aget_state(config=config)
        self.deduplicator.populate_from_history(state.values.get("messages", []))

        # Check for interrupt resumption
        interrupted_tasks = [
            task
            for task in state.tasks
            if hasattr(task, "interrupts") and task.interrupts
        ]

        user_input: Command | dict[str, Any]
        if interrupted_tasks:
            user_input = Command(resume=request.message)
        else:
            user_input = {"messages": [HumanMessage(content=request.message)]}

        logger.info(
            f"Configured run_id={run_id}, thread_id={thread_id}, session_id={effective_session_id}"
        )

        # Create stream context
        ctx = StreamContext(
            run_id=str(run_id),
            thread_id=thread_id,
            session_id=effective_session_id,
            user_id=effective_user_id,
            stream_tokens=request.stream_tokens,
        )

        return {"input": user_input, "config": config}, ctx
