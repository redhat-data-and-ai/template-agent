"""Stream route for the template agent API.

This module provides streaming endpoints for real-time agent interactions,
handling message streaming, token generation, and conversation management.
"""

import asyncio
import hashlib
import json
import time as _time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from template_agent.src.core.manager import AgentManager
from template_agent.src.schema import StreamRequest
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger
from template_agent.utils.tracing import StreamTracer, langfuse_handler

router = APIRouter()
app_logger = get_python_logger(settings.PYTHON_LOG_LEVEL)

_MAX_CONCURRENT_DEEP_RESEARCH_PER_USER = 2
_user_semaphores: dict[str, asyncio.Semaphore] = {}


def _get_user_semaphore(user_id: str) -> asyncio.Semaphore:
    """Return a per-user semaphore, creating one on first access."""
    if user_id not in _user_semaphores:
        _user_semaphores[user_id] = asyncio.Semaphore(
            _MAX_CONCURRENT_DEEP_RESEARCH_PER_USER
        )
    return _user_semaphores[user_id]


async def message_generator(
    user_input: StreamRequest,
    agent_manager: AgentManager,
    stream_tracer: StreamTracer | None = None,
) -> AsyncGenerator[str, None]:
    """Generate a stream of messages from the agent using the simplified format.

    This function uses the AgentManager to handle streaming with features like
    SSO authentication, tracing, and error handling. The AgentManager is
    initialized before streaming begins to allow proper HTTP error responses.

    Args:
        user_input: The streaming input from the user containing the message
            and configuration.
        agent_manager: Pre-initialized AgentManager instance.
        stream_tracer: Optional StreamTracer for Langfuse span tracking.

    Yields:
        JSON-formatted SSE messages as strings in the simplified event format.

    Note:
        - Uses simplified event format: {"type": "message"|"token"|"error", "content": ...}
        - Preserves enterprise features: SSO auth, Langfuse tracing, error handling
        - Errors during streaming are sent as error events in the stream
        - Initialization errors are handled before streaming starts
    """
    start_time = _time.monotonic()
    try:
        msg_hash = hashlib.sha256(user_input.message.encode()).hexdigest()[:12]
        app_logger.info(
            f"Starting stream for message hash={msg_hash}, length={len(user_input.message)}"
        )

        async for event in agent_manager.stream_response(user_input):
            if (
                event.get("type") == "message"
                and event.get("content", {}).get("type") == "human"
                and event.get("content", {}).get("content") == user_input.message
            ):
                continue

            if stream_tracer and event.get("type") == "message":
                content = event.get("content", {}).get("content", "")
                stream_tracer.track_message(str(content))

            yield f"{json.dumps(event, separators=(',', ':'))}\n\n"

    except Exception as e:
        app_logger.error(f"Error in message generator: {e}")
        if stream_tracer:
            stream_tracer.track_error(e)
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
        duration_ms = (_time.monotonic() - start_time) * 1000
        if stream_tracer:
            stream_tracer.end_stream(duration_ms=duration_ms)
        if langfuse_handler:
            langfuse_handler.flush()
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
    access_token = request.headers.get("X-Token")
    app_logger.info(f"Received token: {'Yes' if access_token else 'No'}")

    if user_input.deep_research_enabled and user_input.user_id:
        sem = _get_user_semaphore(user_input.user_id)
        if sem.locked():
            raise HTTPException(
                status_code=429,
                detail="Too many concurrent deep research requests. Please wait for an existing request to finish.",
            )
        await sem.acquire()

    root_tracer = getattr(request.state, "root_tracer", None)
    if root_tracer:
        root_tracer.update(
            user_id=user_input.user_id,
            session_id=user_input.session_id,
            metadata={"message_length": len(user_input.message)},
        )

    stream_tracer = StreamTracer(parent_tracer=root_tracer, name="stream_response")

    try:
        agent_manager = AgentManager(
            redhat_sso_token=access_token,
            root_tracer=root_tracer,
        )
    except Exception as e:
        app_logger.error(f"Failed to initialize AgentManager: {e}")
        if user_input.deep_research_enabled and user_input.user_id:
            _get_user_semaphore(user_input.user_id).release()
        raise HTTPException(status_code=500, detail="Failed to initialize agent")

    async def _guarded_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in message_generator(
                user_input, agent_manager, stream_tracer=stream_tracer
            ):
                yield chunk
        finally:
            if user_input.deep_research_enabled and user_input.user_id:
                _get_user_semaphore(user_input.user_id).release()

    return StreamingResponse(
        _guarded_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
