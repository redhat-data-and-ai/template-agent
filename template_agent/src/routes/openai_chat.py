"""OpenAI-compatible chat completions API endpoint.

This module provides OpenAI API compatibility for chat completions,
allowing existing OpenAI clients to interact with the template agent.
"""

import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from template_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from template_agent.src.core.manager import AgentManager
from template_agent.src.schema import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChatStreamChoice,
    OpenAIChatStreamDelta,
    OpenAIChatStreamResponse,
    OpenAIChoice,
    OpenAIMessage,
    OpenAIUsage,
    StreamRequest,
)
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

router = APIRouter()
logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def _app_exception_to_http_exception(app_exception: AppException) -> HTTPException:
    """Convert AppException to HTTPException for FastAPI compatibility.

    Args:
        app_exception: The AppException to convert.

    Returns:
        HTTPException with appropriate status code and detail.
    """
    return HTTPException(
        status_code=app_exception.response_code,
        detail=f"{app_exception.detail_message} (Error: {app_exception.error_code})",
    )


def _convert_openai_to_internal(openai_request: OpenAIChatRequest) -> StreamRequest:
    """Convert OpenAI request format to internal StreamRequest format.

    Args:
        openai_request: OpenAI-compatible chat request.

    Returns:
        Internal StreamRequest format.
    """
    # Extract the last user message as the primary message
    user_messages = [msg for msg in openai_request.messages if msg.role == "user"]
    if not user_messages:
        app_exception = AppException(
            "No user message found in request", AppExceptionCode.BAD_REQUEST_ERROR
        )
        raise _app_exception_to_http_exception(app_exception)

    last_user_message = user_messages[-1]
    message_content = last_user_message.content or ""

    # Create internal request format
    return StreamRequest(
        message=message_content,
        thread_id=None,  # OpenAI doesn't have thread concept by default
        session_id=openai_request.user,  # Use user field as session_id
        user_id=openai_request.user,
        stream_tokens=openai_request.stream,
    )


def _convert_internal_to_openai_message(
    internal_message: dict[str, Any],
) -> OpenAIMessage:
    """Convert internal message format to OpenAI message format.

    Args:
        internal_message: Internal message format.

    Returns:
        OpenAI-compatible message.
    """
    role_mapping = {
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
        "system": "system",
    }

    message_type = internal_message.get("type", "assistant")
    role = role_mapping.get(message_type, "assistant")

    return OpenAIMessage(
        role=cast(Literal["system", "user", "assistant", "tool"], role),
        content=internal_message.get("content", ""),
        tool_calls=internal_message.get("tool_calls"),
        tool_call_id=internal_message.get("tool_call_id"),
    )


async def _stream_openai_response(
    openai_request: OpenAIChatRequest, request: Request
) -> AsyncGenerator[str, None]:
    """Generate OpenAI-compatible streaming response.

    Args:
        openai_request: OpenAI chat completion request.
        request: FastAPI request object.

    Yields:
        OpenAI-compatible streaming chunks.
    """
    # Convert to internal format
    internal_request = _convert_openai_to_internal(openai_request)

    # Get token from request headers
    auth_header = request.headers.get("Authorization", "")
    access_token = auth_header.replace("Bearer ", "") if auth_header else None
    if not access_token:
        x_token = request.headers.get("X-Token", "")
        access_token = x_token if x_token else None

    logger.info(f"OpenAI chat request with model: {openai_request.model}")

    # Initialize AgentManager
    agent_manager = AgentManager(redhat_sso_token=access_token)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created_time = int(time.time())

    try:
        # Send initial chunk
        initial_chunk = OpenAIChatStreamResponse(
            id=completion_id,
            created=created_time,
            model=openai_request.model,
            choices=[
                OpenAIChatStreamChoice(
                    index=0,
                    delta=OpenAIChatStreamDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"

        # Stream content from agent
        async for event in agent_manager.stream_response(internal_request):
            if event.get("type") == "token":
                # Stream individual tokens
                token_chunk = OpenAIChatStreamResponse(
                    id=completion_id,
                    created=created_time,
                    model=openai_request.model,
                    choices=[
                        OpenAIChatStreamChoice(
                            index=0,
                            delta=OpenAIChatStreamDelta(
                                content=event.get("content", "")
                            ),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {token_chunk.model_dump_json()}\n\n"

            elif event.get("type") == "message":
                content = event.get("content", {})
                if content.get("type") == "ai" and content.get("content"):
                    # Final message - send completion
                    final_chunk = OpenAIChatStreamResponse(
                        id=completion_id,
                        created=created_time,
                        model=openai_request.model,
                        choices=[
                            OpenAIChatStreamChoice(
                                index=0,
                                delta=OpenAIChatStreamDelta(),
                                finish_reason="stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                    break

    except Exception as e:
        logger.error(f"Error in OpenAI streaming: {e}")
        error_chunk = OpenAIChatStreamResponse(
            id=completion_id,
            created=created_time,
            model=openai_request.model,
            choices=[
                OpenAIChatStreamChoice(
                    index=0,
                    delta=OpenAIChatStreamDelta(),
                    finish_reason="content_filter",
                )
            ],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"

    # Send final done marker
    yield "data: [DONE]\n\n"


async def _get_openai_response(
    openai_request: OpenAIChatRequest, request: Request
) -> OpenAIChatResponse:
    """Generate OpenAI-compatible non-streaming response.

    Args:
        openai_request: OpenAI chat completion request.
        request: FastAPI request object.

    Returns:
        OpenAI-compatible chat response.
    """
    # Convert to internal format
    internal_request = _convert_openai_to_internal(openai_request)
    internal_request.stream_tokens = False  # Force non-streaming

    # Get token from request headers
    auth_header = request.headers.get("Authorization", "")
    access_token = auth_header.replace("Bearer ", "") if auth_header else None
    if not access_token:
        x_token = request.headers.get("X-Token", "")
        access_token = x_token if x_token else None

    logger.info(
        f"OpenAI chat request (non-streaming) with model: {openai_request.model}"
    )

    # Initialize AgentManager
    agent_manager = AgentManager(redhat_sso_token=access_token)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created_time = int(time.time())

    try:
        # Collect all events to build final response
        response_content = ""

        async for event in agent_manager.stream_response(internal_request):
            if event.get("type") == "message":
                content = event.get("content", {})
                if content.get("type") == "ai" and content.get("content"):
                    response_content = content.get("content", "")
                    break

        # Build OpenAI response
        choice = OpenAIChoice(
            index=0,
            message=OpenAIMessage(
                role="assistant",
                content=response_content,
            ),
            finish_reason="stop",
        )

        return OpenAIChatResponse(
            id=completion_id,
            created=created_time,
            model=openai_request.model,
            choices=[choice],
            usage=OpenAIUsage(
                prompt_tokens=0,  # Would need tokenizer to calculate
                completion_tokens=0,  # Would need tokenizer to calculate
                total_tokens=0,
            ),
        )

    except Exception as e:
        logger.error(f"Error in OpenAI non-streaming: {e}")
        app_exception = AppException(
            "Internal server error during OpenAI non-streaming response",
            AppExceptionCode.INTERNAL_SERVER_ERROR,
        )
        raise _app_exception_to_http_exception(app_exception)


@router.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    request_data: OpenAIChatRequest,
    request: Request,
):
    """Create a chat completion, compatible with OpenAI API.

    This endpoint provides OpenAI API compatibility for chat completions,
    supporting both streaming and non-streaming responses.

    Args:
        request_data: OpenAI-compatible chat completion request.
        request: FastAPI request object for authentication.

    Returns:
        OpenAI-compatible chat completion response or streaming response.
    """
    try:
        if request_data.stream:
            return StreamingResponse(
                _stream_openai_response(request_data, request),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            return await _get_openai_response(request_data, request)

    except HTTPException:
        # Re-raise HTTP exceptions to preserve status codes (400, 422, etc.)
        raise
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        app_exception = AppException(
            "Internal server error during chat completion",
            AppExceptionCode.INTERNAL_SERVER_ERROR,
        )
        raise _app_exception_to_http_exception(app_exception)
