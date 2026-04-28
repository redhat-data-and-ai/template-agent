"""Tests for route handlers - stream and history endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from template_agent.src.routes.history import router as history_router
from template_agent.src.routes.stream import router as stream_router


@pytest.fixture
def app_with_routes(monkeypatch):
    """Create test app with routes."""
    monkeypatch.setenv("USE_INMEMORY_SAVER", "true")
    monkeypatch.setattr(
        "template_agent.src.routes.history.settings.USE_INMEMORY_SAVER", True
    )

    app = FastAPI()
    app.include_router(stream_router)
    app.include_router(history_router)
    return app


class TestStreamEndpoint:
    """Tests for /v1/stream endpoint."""

    def test_requires_auth_token(self, app_with_routes):
        """Returns 401 when no auth token provided."""
        client = TestClient(app_with_routes)
        resp = client.post(
            "/v1/stream",
            json={"message": "hello", "session_id": "s1", "user_id": "u1"},
        )
        assert resp.status_code == 401
        assert "Missing credentials" in resp.json()["detail"]

    def test_accepts_x_token_header(self, app_with_routes):
        """Accepts X-Token header for authentication."""
        client = TestClient(app_with_routes)

        with patch("template_agent.src.routes.stream.AgentManager") as MockManager:
            instance = MockManager.return_value

            async def _mock_stream(request):
                yield {"type": "message", "content": {"type": "ai", "content": "hello"}}

            instance.stream_response = _mock_stream

            resp = client.post(
                "/v1/stream",
                json={"message": "hello", "session_id": "s1", "user_id": "u1"},
                headers={"X-Token": "test-token"},
            )

        assert resp.status_code == 200

    def test_accepts_bearer_token(self, app_with_routes):
        """Accepts Authorization: Bearer header for authentication."""
        client = TestClient(app_with_routes)

        with patch("template_agent.src.routes.stream.AgentManager") as MockManager:
            instance = MockManager.return_value

            async def _mock_stream(request):
                yield {"type": "message", "content": {"type": "ai", "content": "hello"}}

            instance.stream_response = _mock_stream

            resp = client.post(
                "/v1/stream",
                json={"message": "hello", "session_id": "s1", "user_id": "u1"},
                headers={"Authorization": "Bearer test-token"},
            )

        assert resp.status_code == 200

    def test_handles_agent_manager_init_failure(self, app_with_routes):
        """Returns 500 when AgentManager initialization fails."""
        client = TestClient(app_with_routes, raise_server_exceptions=False)

        with patch(
            "template_agent.src.routes.stream.AgentManager",
            side_effect=Exception("Init failed"),
        ):
            resp = client.post(
                "/v1/stream",
                json={"message": "hello", "session_id": "s1", "user_id": "u1"},
                headers={"X-Token": "test-token"},
            )

        assert resp.status_code == 500
        assert "Failed to initialize agent" in resp.json()["detail"]


class TestHistoryEndpoint:
    """Tests for /v1/history/{thread_id} endpoint."""

    def test_requires_auth_token(self, app_with_routes):
        """Returns 401 when no auth token provided."""
        client = TestClient(app_with_routes)
        resp = client.get("/v1/history/thread-123")
        assert resp.status_code == 401
        assert "Missing credentials" in resp.json()["detail"]

    def test_returns_empty_history_for_new_thread(self, app_with_routes):
        """Returns empty history for non-existent thread."""
        client = TestClient(app_with_routes)

        with patch(
            "template_agent.src.routes.history.get_shared_checkpointer"
        ) as mock_checkpointer:
            mock_cp = MagicMock()
            mock_cp.list.return_value = []
            mock_checkpointer.return_value = mock_cp

            resp = client.get(
                "/v1/history/thread-123",
                headers={"X-Token": "test-token"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["messages"] == []

    def test_accepts_bearer_token(self, app_with_routes):
        """Accepts Authorization: Bearer header for authentication."""
        client = TestClient(app_with_routes)

        with patch(
            "template_agent.src.routes.history.get_shared_checkpointer"
        ) as mock_checkpointer:
            mock_cp = MagicMock()
            mock_cp.list.return_value = []
            mock_checkpointer.return_value = mock_cp

            resp = client.get(
                "/v1/history/thread-123",
                headers={"Authorization": "Bearer test-token"},
            )

        assert resp.status_code == 200


class TestStreamResponseFormat:
    """Tests for stream response format."""

    def test_stream_yields_json_events(self, app_with_routes):
        """Stream yields JSON-formatted events."""
        client = TestClient(app_with_routes)

        with patch("template_agent.src.routes.stream.AgentManager") as MockManager:
            instance = MockManager.return_value

            async def _mock_stream(request):
                yield {"type": "token", "content": "Hello"}
                yield {"type": "token", "content": " world"}
                yield {
                    "type": "message",
                    "content": {"type": "ai", "content": "Hello world"},
                }

            instance.stream_response = _mock_stream

            resp = client.post(
                "/v1/stream",
                json={
                    "message": "hi",
                    "session_id": "s1",
                    "user_id": "u1",
                    "stream_tokens": True,
                },
                headers={"X-Token": "test-token"},
            )

        assert resp.status_code == 200
        content = resp.text
        assert "[DONE]" in content

    def test_stream_filters_duplicate_human_messages(self, app_with_routes):
        """Stream filters out duplicate human messages."""
        client = TestClient(app_with_routes)

        with patch("template_agent.src.routes.stream.AgentManager") as MockManager:
            instance = MockManager.return_value

            async def _mock_stream(request):
                yield {
                    "type": "message",
                    "content": {"type": "human", "content": "test message"},
                }
                yield {
                    "type": "message",
                    "content": {"type": "ai", "content": "response"},
                }

            instance.stream_response = _mock_stream

            resp = client.post(
                "/v1/stream",
                json={"message": "test message", "session_id": "s1", "user_id": "u1"},
                headers={"X-Token": "test-token"},
            )

        assert resp.status_code == 200
        content = resp.text
        lines = [
            line for line in content.split("\n\n") if line.strip() and line != "[DONE]"
        ]
        assert len(lines) == 1
        assert "response" in lines[0]
