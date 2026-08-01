"""Tests for aegra.state module."""

from deep_agent.aegra.state import (
    AegraMetadata,
    HealthStatus,
    make_health_status,
    serialize_metadata,
)


class TestAegraMetadata:
    """Tests for AegraMetadata TypedDict operations."""

    def test_full_metadata_creation(self):
        meta: AegraMetadata = {
            "run_id": "run-123",
            "trace_id": "trace-456",
            "thread_id": "thread-789",
            "session_id": "session-abc",
            "user_id": "user-def",
            "stream_tokens": True,
            "error_count": 0,
            "last_error": None,
        }
        assert meta["run_id"] == "run-123"
        assert meta["error_count"] == 0

    def test_partial_metadata_creation(self):
        meta: AegraMetadata = {"run_id": "run-123", "thread_id": "thread-456"}
        assert meta["run_id"] == "run-123"
        assert "user_id" not in meta


class TestSerializeMetadata:
    """Tests for serialize_metadata helper."""

    def test_strips_none_values(self):
        meta: AegraMetadata = {
            "run_id": "run-123",
            "last_error": None,
        }
        result = serialize_metadata(meta)
        assert "run_id" in result
        assert "last_error" not in result

    def test_preserves_falsy_non_none_values(self):
        meta: AegraMetadata = {"error_count": 0, "stream_tokens": False}
        result = serialize_metadata(meta)
        assert result["error_count"] == 0
        assert result["stream_tokens"] is False

    def test_empty_metadata(self):
        result = serialize_metadata({})
        assert result == {}


class TestMakeHealthStatus:
    """Tests for make_health_status factory."""

    def test_produces_valid_health_status(self):
        status: HealthStatus = make_health_status(
            agent_name="orchestrator",
            model="gemini-3.1-pro-preview",
            mcp_tools_count=4,
            subagents_count=2,
            backend_ready=True,
        )
        assert status["status"] == "healthy"
        assert status["agent_name"] == "orchestrator"
        assert status["mcp_tools_loaded"] == 4
        assert status["subagents_loaded"] == 2
        assert status["backend_ready"] is True

    def test_zero_tools_and_subagents(self):
        status = make_health_status(
            agent_name="test",
            model="test-model",
            mcp_tools_count=0,
            subagents_count=0,
            backend_ready=False,
        )
        assert status["mcp_tools_loaded"] == 0
        assert status["backend_ready"] is False
