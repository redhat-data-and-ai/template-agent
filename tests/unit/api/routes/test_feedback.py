"""Unit tests for feedback route."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from template_agent.src.api.routes.agent.feedback import feedback
from template_agent.src.schema import FeedbackRequest, FeedbackResponse


class TestFeedback:
    """Tests for feedback endpoint."""

    @pytest.mark.asyncio
    async def test_successful_feedback_submission(self):
        """Test successful feedback submission to Langfuse."""
        mock_client = MagicMock()
        mock_client.create_score = MagicMock()
        mock_client.flush = MagicMock()

        request = FeedbackRequest(
            trace_id="trace123",
            name="user-rating",
            value=0.85,
        )

        response = await feedback(request, client=mock_client)

        assert isinstance(response, FeedbackResponse)
        mock_client.create_score.assert_called_once_with(
            trace_id="trace123",
            name="user-rating",
            value=0.85,
        )
        mock_client.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_feedback_with_kwargs(self):
        """Test feedback submission with additional kwargs."""
        mock_client = MagicMock()
        mock_client.create_score = MagicMock()
        mock_client.flush = MagicMock()

        request = FeedbackRequest(
            trace_id="trace456",
            name="thumbs-up",
            value=1.0,
            kwargs={"comment": "Excellent response!", "category": "helpful"},
        )

        response = await feedback(request, client=mock_client)

        assert isinstance(response, FeedbackResponse)
        mock_client.create_score.assert_called_once_with(
            trace_id="trace456",
            name="thumbs-up",
            value=1.0,
            comment="Excellent response!",
            category="helpful",
        )
        mock_client.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_langfuse_not_configured(self):
        """Test that HTTPException is raised when Langfuse is not configured."""
        request = FeedbackRequest(
            trace_id="tracexyz",
            name="test",
            value=0.5,
        )

        with pytest.raises(HTTPException) as exc_info:
            await feedback(request, client=None)

        assert exc_info.value.status_code == 503
        assert "not configured" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_langfuse_client_dependency(self):
        """Test that get_langfuse_client extracts client from app state."""
        from template_agent.src.api.routes.agent.feedback import get_langfuse_client

        mock_request = MagicMock()
        mock_client = MagicMock()
        mock_request.app.state.langfuse_client = mock_client

        client = get_langfuse_client(mock_request)

        assert client is mock_client

    @pytest.mark.asyncio
    async def test_feedback_api_error_handling(self):
        """Test that API errors are handled and logged properly."""
        mock_client = MagicMock()
        # create_score is called via to_thread, so it should be sync MagicMock
        mock_client.create_score = MagicMock(
            side_effect=Exception("Langfuse API error")
        )
        mock_client.flush = MagicMock()

        request = FeedbackRequest(
            trace_id="traceerror",
            name="test",
            value=0.5,
        )

        with pytest.raises(HTTPException) as exc_info:
            await feedback(request, client=mock_client)

        assert exc_info.value.status_code == 500
        assert "Failed to record feedback" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_trace_id_hex_format_validation(self):
        """Test that trace_id is properly formatted (hex without hyphens)."""
        mock_client = MagicMock()
        mock_client.create_score = MagicMock()
        mock_client.flush = MagicMock()

        # Use realistic hex format (32 chars, no hyphens)
        hex_trace_id = "847c62858fc94560a83f4e6285809254"
        request = FeedbackRequest(
            trace_id=hex_trace_id,
            name="test",
            value=1.0,
        )

        await feedback(request, client=mock_client)

        # Verify trace_id is passed correctly in hex format
        call_args = mock_client.create_score.call_args
        assert call_args.kwargs["trace_id"] == hex_trace_id
        assert "-" not in call_args.kwargs["trace_id"]
        assert len(call_args.kwargs["trace_id"]) == 32
