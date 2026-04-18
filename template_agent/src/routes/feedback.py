"""Feedback route for the template agent API.

This module provides endpoints for recording user feedback on agent responses
using Langfuse for analytics and monitoring purposes.
"""

from asyncio import to_thread
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from langfuse import Langfuse

from template_agent.src.schema import FeedbackRequest, FeedbackResponse
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

router = APIRouter()
logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def get_langfuse_client(request: Request) -> Optional[Langfuse]:
    """Get Langfuse client from app state.

    Args:
        request: FastAPI request object.

    Returns:
        Langfuse client instance or None if not configured.
    """
    return request.app.state.langfuse_client


@router.post("/v1/feedback")
async def feedback(
    feedback_request: FeedbackRequest,
    client: Optional[Langfuse] = Depends(get_langfuse_client),
) -> FeedbackResponse:
    """Record feedback for a specific agent run to Langfuse.

    This endpoint serves as a wrapper for the Langfuse create_score API,
    allowing credentials to be stored and managed in the service rather than
    requiring client-side credential management.

    The function maps the feedback request parameters to Langfuse's expected
    format:
    - run_id -> trace_id
    - key -> name
    - score -> value

    Args:
        feedback_request: The feedback request containing run_id, key, score, and
            optional kwargs for additional metadata.
        client: Langfuse client instance injected by FastAPI.

    Returns:
        A FeedbackResponse indicating successful feedback recording.

    Raises:
        HTTPException: If Langfuse is not configured or if there are API issues.

    See Also:
        https://langfuse.com/docs/scores/api
    """
    if not client:
        raise HTTPException(
            status_code=503,
            detail="Langfuse feedback service not configured. Please configure Langfuse.",
        )

    logger.info(
        f"Recording feedback: run_id={feedback_request.run_id}, "
        f"key={feedback_request.key}, score={feedback_request.score}"
    )

    kwargs = feedback_request.kwargs or {}

    try:
        # Run blocking I/O operations in thread pool to avoid blocking event loop
        # Langfuse uses different parameter names than our schema
        await to_thread(
            client.create_score,
            trace_id=feedback_request.run_id,  # Assuming run_id maps to trace_id
            name=feedback_request.key,  # 'key' becomes 'name' in Langfuse
            value=feedback_request.score,  # 'score' becomes 'value' in Langfuse
            **kwargs,
        )

        # Flush to ensure score is sent to Langfuse immediately
        await to_thread(client.flush)

        logger.info(
            f"Successfully recorded feedback for run_id={feedback_request.run_id}"
        )
        return FeedbackResponse()

    except Exception as e:
        logger.error(
            f"Failed to submit feedback for run_id={feedback_request.run_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record feedback: {str(e)}",
        )
