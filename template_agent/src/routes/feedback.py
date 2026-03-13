"""Feedback route for the template agent API.

This module provides endpoints for recording user feedback on agent responses
using Langfuse for analytics and monitoring purposes.
"""

from fastapi import APIRouter, HTTPException

from template_agent.src.schema import FeedbackRequest, FeedbackResponse
from template_agent.utils.tracing import client

router = APIRouter()


@router.post("/v1/feedback")
async def feedback(feedback: FeedbackRequest) -> FeedbackResponse:
    """Record feedback for a specific agent run to Langfuse.

    This endpoint serves as a wrapper for the Langfuse create_feedback API,
    allowing credentials to be stored and managed in the service rather than
    requiring client-side credential management.

    The function maps the feedback request parameters to Langfuse's expected
    format:
    - run_id -> trace_id
    - key -> name
    - score -> value

    Args:
        feedback: The feedback request containing run_id, key, score, and
            optional kwargs for additional metadata.

    Returns:
        A FeedbackResponse indicating successful feedback recording.

    Raises:
        HTTPException: If Langfuse is not configured.

    See Also:
        https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Langfuse is not configured; feedback cannot be recorded.",
        )

    kwargs = feedback.kwargs or {}

    client.score(
        trace_id=feedback.run_id,
        name=feedback.key,
        value=feedback.score,
        **kwargs,
    )

    return FeedbackResponse()
