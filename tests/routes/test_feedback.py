"""Tests for the feedback route."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from template_agent.src.routes.feedback import router


class TestFeedbackRoute:
    """Test cases for feedback endpoint."""

    @patch("template_agent.src.routes.feedback.client")
    def test_feedback_endpoint_success(self, mock_client):
        """Test feedback endpoint with successful Langfuse call."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        # Mock the Langfuse client
        mock_client.score.return_value = None

        feedback_data = {
            "run_id": "run_123",
            "key": "response_quality",
            "score": 4.5,
            "kwargs": {"comment": "Great response"},
        }

        response = client.post("/v1/feedback", json=feedback_data)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"

        # Verify Langfuse was called correctly
        mock_client.score.assert_called_once_with(
            trace_id="run_123",
            name="response_quality",
            value=4.5,
            comment="Great response",
        )

    @patch("template_agent.src.routes.feedback.client")
    def test_feedback_endpoint_minimal_data(self, mock_client):
        """Test feedback endpoint with minimal required data."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        # Mock the Langfuse client
        mock_client.score.return_value = None

        feedback_data = {"run_id": "run_123", "key": "response_quality", "score": 4.5}

        response = client.post("/v1/feedback", json=feedback_data)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"

        # Verify Langfuse was called correctly
        mock_client.score.assert_called_once_with(
            trace_id="run_123", name="response_quality", value=4.5
        )
