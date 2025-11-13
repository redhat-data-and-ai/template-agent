"""Stream route for the template agent API.

This module provides streaming endpoints for real-time agent interactions,
handling message streaming, token generation, and conversation management.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from template_agent.src.core.manager import AgentManager
from template_agent.src.schema import StreamRequest
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

router = APIRouter()
app_logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


async def message_generator(
    user_input: StreamRequest, agent_manager: AgentManager
) -> AsyncGenerator[str, None]:
    """Generate a stream of messages from the agent using the simplified format.

    This function uses the AgentManager to handle streaming with features like
    SSO authentication, tracing, and error handling. The AgentManager is
    initialized before streaming begins to allow proper HTTP error responses.

    Args:
        user_input: The streaming input from the user containing the message
            and configuration.
        agent_manager: Pre-initialized AgentManager instance.

    Yields:
        JSON-formatted SSE messages as strings in the simplified event format.

    Note:
        - Uses simplified event format: {"type": "message"|"token"|"error", "content": ...}
        - Preserves enterprise features: SSO auth, Langfuse tracing, error handling
        - Errors during streaming are sent as error events in the stream
        - Initialization errors are handled before streaming starts
    """
    try:
        app_logger.info(f"Starting stream for message: {user_input.message[:100]}...")

        # Stream events using the simplified AgentManager
        async for event in agent_manager.stream_response(user_input):
            # Filter out duplicate human messages
            if (
                event.get("type") == "message"
                and event.get("content", {}).get("type") == "human"
                and event.get("content", {}).get("content") == user_input.message
            ):
                continue

            # Yield the simplified event format
            yield f"{json.dumps(event, separators=(',', ':'))}\n\n"

    except Exception as e:
        app_logger.error(f"Error in message generator: {e}")
        error_event = {
            "type": "error",
            "content": {
                "message": "Internal server error",
                "recoverable": False,
                "error_type": "stream_error",
            },
        }
        yield f"{json.dumps(error_event)}\n\n"
    finally:
        # Send completion marker
        yield "[DONE]\n\n"


def _sse_response_example() -> dict[int | str, Any]:
    """Generate example response for SSE endpoint documentation.

    Returns:
        A dictionary containing the example SSE response format for
        the simplified streaming API.
    """
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response - Simplified Format",
            "content": {
                "text/event-stream": {
                    "example": '{"type": "message", "content": {"type": "ai", "content": "", "tool_calls": [{"name": "multiply", "args": {"a": 3, "b": 2}, "id": "call_123"}], "run_id": "12345", "thread_id": "thread-123", "session_id": "session-456"}}\n\n{"type": "message", "content": {"type": "tool", "content": "6", "tool_call_id": "call_123", "run_id": "12345", "thread_id": "thread-123", "session_id": "session-456"}}\n\n{"type": "token", "content": "The"}\n\n{"type": "token", "content": " answer"}\n\n{"type": "token", "content": " is"}\n\n{"type": "token", "content": " 6"}\n\n{"type": "message", "content": {"type": "ai", "content": "The answer is 6", "run_id": "12345", "thread_id": "thread-123", "session_id": "session-456"}}\n\n[DONE]\n\n',
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/v1/stream", response_class=StreamingResponse, responses=_sse_response_example()
)
async def stream(user_input: StreamRequest, request: Request) -> StreamingResponse:
    """Stream AI agent responses in real-time using simplified event format.

    This endpoint provides the core streaming functionality following the
    simplified API design with features like SSO
    authentication, Langfuse tracing, and comprehensive error handling.

    **Event Types:**
    - `message` - Tool calls, tool results, and final responses
    - `token` - Individual tokens (only when `stream_tokens: true`)
    - `error` - Error messages with recovery information
    - `[DONE]` - Stream completion marker

    **Request Fields:**
    - `message`: User's input message (required)
    - `thread_id`: Conversation thread identifier (optional - auto-generated if not provided)
    - `session_id`: Session identifier (required)
    - `user_id`: User identifier for tracking and personalization (required)
    - `stream_tokens`: Whether to stream individual tokens (`true`) or just complete messages (`false`) (optional)

    **Enterprise Features (Preserved):**
    - SSO authentication via X-Token header
    - Langfuse tracing and analytics
    - PostgreSQL checkpointing for conversation persistence
    - Comprehensive error handling and logging

    Args:
        user_input: The streaming request with simplified structure.
        request: FastAPI request object for extracting authentication headers.

    Returns:
        StreamingResponse with simplified event format:
        ```
        {"type": "message", "content": {"type": "ai", "content": "Hello", "run_id": "12345", "thread_id": "thread-123", "session_id": "session-456"}}
        {"type": "token", "content": "world"}
        [DONE]
        ```

    Raises:
        HTTPException: If initialization fails (returns 500 status code).
    """
    # Get token from request headers
    access_token = request.headers.get("X-Token")
    app_logger.info(f"Received token: {'Yes' if access_token else 'No'}")

    # Initialize AgentManager BEFORE streaming to catch initialization errors
    try:
        agent_manager = AgentManager(redhat_sso_token=access_token)
    except Exception as e:
        app_logger.error(f"Failed to initialize AgentManager: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize agent: {str(e)}"
        )

    return StreamingResponse(
        message_generator(user_input, agent_manager),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
