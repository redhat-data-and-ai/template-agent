"""Feedback route for the template agent API.

This module provides endpoints for recording user feedback on agent responses
using Langfuse for analytics and monitoring purposes.
"""

from fastapi import APIRouter
from langfuse import Langfuse

from template_agent.src.schema import FeedbackRequest, FeedbackResponse

router = APIRouter()

# Lazy initialization to ensure env vars are loaded first
_langfuse_client = None


def get_langfuse_client() -> Langfuse:
    """Get or create the Langfuse client singleton.

    Lazy initialization ensures environment variables are loaded before
    the client is created, avoiding initialization issues if this module
    is imported before env vars are set.

    Returns:
        Langfuse client instance.
    """
    global _langfuse_client
    if _langfuse_client is None:
        # Environment is auto-read from LANGFUSE_TRACING_ENVIRONMENT env var
        _langfuse_client = Langfuse()
    return _langfuse_client


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
        Exception: If there are issues with the Langfuse API call.

    See Also:
        https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    kwargs = feedback.kwargs or {}

    # Langfuse uses different parameter names than our schema
    client = get_langfuse_client()
    client.score(
        trace_id=feedback.run_id,  # Assuming run_id maps to trace_id
        name=feedback.key,  # 'key' becomes 'name' in Langfuse
        value=feedback.score,  # 'score' becomes 'value' in Langfuse
        **kwargs,
    )

    return FeedbackResponse()
