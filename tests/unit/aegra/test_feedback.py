"""Unit tests for Langfuse feedback recording and HTTP handler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from starlette.requests import Request
from starlette.testclient import TestClient

from deep_agent.aegra.feedback import feedback_handler, record_feedback
from deep_agent.aegra.http_app import app


class TestRecordFeedback:
    @pytest.mark.asyncio
    async def test_records_score_when_langfuse_configured(self):
        mock_client = MagicMock()
        payload = {
            "trace_id": "abcd1234" * 4,
            "name": "user-rating",
            "value": 1.0,
            "kwargs": {"comment": "great"},
        }

        with patch(
            "deep_agent.aegra.feedback.get_langfuse_client",
            return_value=mock_client,
        ):
            result = await record_feedback(payload)

        assert result.status == "success"
        mock_client.create_score.assert_called_once_with(
            trace_id=payload["trace_id"],
            name="user-rating",
            value=1.0,
            data_type="BOOLEAN",
            comment="great",
        )

    @pytest.mark.asyncio
    async def test_graceful_degradation_when_langfuse_unconfigured(self):
        payload = {
            "trace_id": "abcd1234" * 4,
            "name": "thumbs-up",
            "value": 1.0,
        }

        with patch(
            "deep_agent.aegra.feedback.get_langfuse_client",
            return_value=None,
        ):
            result = await record_feedback(payload)

        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_validation_error_on_missing_fields(self):
        with pytest.raises(ValidationError):
            await record_feedback({})

    @pytest.mark.asyncio
    async def test_gracefully_handles_score_failure(self):
        mock_client = MagicMock()
        mock_client.create_score.side_effect = RuntimeError("network")

        payload = {
            "trace_id": "abcd1234" * 4,
            "name": "user-rating",
            "value": 0.5,
        }

        with patch(
            "deep_agent.aegra.feedback.get_langfuse_client",
            return_value=mock_client,
        ):
            result = await record_feedback(payload)

        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_persists_postgres_when_thread_and_message_present(self):
        payload = {
            "trace_id": "a" * 32,
            "name": "user-rating",
            "value": 1.0,
            "thread_id": "thread-1",
            "message_id": "msg-1",
            "user_id": "user-42",
        }
        mock_upsert = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.upsert_feedback = mock_upsert

        with patch(
            "deep_agent.aegra.feedback.get_langfuse_client",
            return_value=None,
        ):
            with patch(
                "deep_agent.aegra.feedback.FeedbackRepository",
                return_value=mock_repo,
            ):
                result = await record_feedback(payload)

        assert result.status == "success"
        mock_upsert.assert_awaited_once_with(
            "thread-1",
            "msg-1",
            "user-42",
            "up",
            "a" * 32,
        )

    @pytest.mark.asyncio
    async def test_skips_postgres_when_thread_or_message_missing(self):
        payload = {
            "trace_id": "a" * 32,
            "name": "user-rating",
            "value": 0.2,
        }

        with patch(
            "deep_agent.aegra.feedback.get_langfuse_client",
            return_value=None,
        ):
            with patch(
                "deep_agent.aegra.feedback.FeedbackRepository",
            ) as mock_repo_cls:
                result = await record_feedback(payload)

        assert result.status == "success"
        mock_repo_cls.assert_not_called()


class TestFeedbackHandler:
    @pytest.mark.asyncio
    async def test_validation_error_response_shape(self):
        scope = {
            "type": "http",
            "asgi": {"spec_version": "2.0", "version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/feedback",
            "raw_path": b"/feedback",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 80),
        }

        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        request = Request(scope, receive)
        response = await feedback_handler(request)
        assert response.status_code == 422

    def test_post_feedback_via_test_client(self):
        client = TestClient(app)
        payload = {
            "trace_id": "a" * 32,
            "name": "user-rating",
            "value": 1.0,
        }
        with patch(
            "deep_agent.aegra.feedback.get_langfuse_client",
            return_value=None,
        ):
            res = client.post("/feedback", json=payload)
        assert res.status_code == 200
        assert res.json() == {"status": "success"}

    def test_get_thread_feedback(self):
        client = TestClient(app)
        thread_uuid = "00000000-0000-0000-0000-000000000001"

        mock_repo = MagicMock()
        mock_repo.list_feedback = AsyncMock(
            return_value=[{"message_id": "m1", "feedback": "up"}]
        )
        with patch(
            "deep_agent.aegra.feedback.FeedbackRepository",
            return_value=mock_repo,
        ):
            res = client.get(
                f"/feedback/{thread_uuid}",
                params={"user_id": "u1"},
            )
        assert res.status_code == 200
        assert res.json() == {"feedback": [{"message_id": "m1", "feedback": "up"}]}
        mock_repo.list_feedback.assert_awaited_once_with(thread_uuid, "u1")


class TestTokenUsageEndpoint:
    def test_get_thread_token_usage_success(self) -> None:
        from deep_agent.src.token_budget.service import ThreadTokenUsage

        client = TestClient(app)
        thread_uuid = "00000000-0000-0000-0000-000000000001"

        with patch(
            "deep_agent.src.token_budget.service.get_thread_token_usage",
            new=AsyncMock(
                return_value=ThreadTokenUsage(
                    thread_id=thread_uuid,
                    used=150,
                    input_tokens=100,
                    output_tokens=50,
                )
            ),
        ):
            res = client.get(f"/threads/{thread_uuid}/token-usage")

        assert res.status_code == 200
        assert res.json() == {
            "thread_id": thread_uuid,
            "used": 150,
            "input_tokens": 100,
            "output_tokens": 50,
        }

    def test_get_thread_token_usage_not_found(self) -> None:
        from deep_agent.src.token_budget.service import TokenUsageNotFoundError

        client = TestClient(app)
        thread_uuid = "00000000-0000-0000-0000-000000000001"
        with patch(
            "deep_agent.src.token_budget.service.get_thread_token_usage",
            new=AsyncMock(side_effect=TokenUsageNotFoundError(thread_uuid)),
        ):
            res = client.get(f"/threads/{thread_uuid}/token-usage")

        assert res.status_code == 404

    def test_get_thread_token_usage_unavailable(self) -> None:
        from deep_agent.src.token_budget.service import TokenUsageUnavailableError

        client = TestClient(app)
        thread_uuid = "00000000-0000-0000-0000-000000000001"
        with patch(
            "deep_agent.src.token_budget.service.get_thread_token_usage",
            new=AsyncMock(side_effect=TokenUsageUnavailableError("down")),
        ):
            res = client.get(f"/threads/{thread_uuid}/token-usage")

        assert res.status_code == 503
