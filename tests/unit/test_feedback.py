"""Unit tests for feedback route."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from template_agent.src.routes.feedback import feedback
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
            run_id="run-123",
            key="user-rating",
            score=0.85,
        )

        response = await feedback(request, client=mock_client)

        assert isinstance(response, FeedbackResponse)
        mock_client.create_score.assert_called_once_with(
            trace_id="run-123",
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
            run_id="run-456",
            key="thumbs-up",
            score=1.0,
            kwargs={"comment": "Excellent response!", "category": "helpful"},
        )

        response = await feedback(request, client=mock_client)

        assert isinstance(response, FeedbackResponse)
        mock_client.create_score.assert_called_once_with(
            trace_id="run-456",
            name="thumbs-up",
            value=1.0,
            comment="Excellent response!",
            category="helpful",
        )
        mock_client.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_feedback_with_empty_kwargs(self):
        """Test feedback submission with empty kwargs."""
        mock_client = MagicMock()
        mock_client.create_score = MagicMock()
        mock_client.flush = MagicMock()

        request = FeedbackRequest(
            run_id="run-789",
            key="rating",
            score=0.5,
            kwargs={},
        )

        response = await feedback(request, client=mock_client)

        assert isinstance(response, FeedbackResponse)
        mock_client.create_score.assert_called_once_with(
            trace_id="run-789",
            name="rating",
            value=0.5,
        )
        mock_client.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_flush_called(self):
        """Test that client.flush() is called to ensure immediate submission."""
        mock_client = MagicMock()
        mock_client.create_score = MagicMock()
        mock_client.flush = MagicMock()

        request = FeedbackRequest(
            run_id="run-abc",
            key="feedback",
            score=0.9,
        )

        await feedback(request, client=mock_client)

        # Verify flush is called after create_score
        mock_client.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_langfuse_not_configured(self):
        """Test that HTTPException is raised when Langfuse is not configured."""
        request = FeedbackRequest(
            run_id="run-xyz",
            key="test",
            score=0.5,
        )

        with pytest.raises(HTTPException) as exc_info:
            await feedback(request, client=None)

        assert exc_info.value.status_code == 503
        assert "not configured" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_langfuse_client_dependency(self):
        """Test that get_langfuse_client extracts client from app state."""
        from template_agent.src.routes.feedback import get_langfuse_client

        mock_request = MagicMock()
        mock_client = MagicMock()
        mock_request.app.state.langfuse_client = mock_client

        client = get_langfuse_client(mock_request)

        assert client is mock_client

    @pytest.mark.asyncio
    async def test_feedback_api_error_handling(self):
        """Test that API errors are handled and logged properly."""
        mock_client = MagicMock()
        mock_client.create_score = MagicMock(
            side_effect=Exception("Langfuse API error")
        )
        mock_client.flush = MagicMock()

        request = FeedbackRequest(
            run_id="run-error",
            key="test",
            score=0.5,
        )

        with pytest.raises(HTTPException) as exc_info:
            await feedback(request, client=mock_client)

        assert exc_info.value.status_code == 500
        assert "Failed to record feedback" in exc_info.value.detail
