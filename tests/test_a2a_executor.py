"""Tests for A2A executor - bridging template agent to A2A protocol."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import Task, TaskArtifactUpdateEvent, TaskState, TaskStatusUpdateEvent

from template_agent.src.a2a.context import A2ARequestContext, a2a_request_ctx
from template_agent.src.core.a2a_executor import (
    TemplateAgentA2AExecutor,
    _meta_str,
    _optional_thread_id,
)


class TestMetaStrFunction:
    """Tests for _meta_str helper function."""

    def test_returns_string_value(self):
        """Returns trimmed string when key exists with valid string."""
        metadata = {"user_id": "  test-user  "}
        assert _meta_str(metadata, "user_id", "default") == "test-user"

    def test_returns_default_when_key_missing(self):
        """Returns default when key is not in metadata."""
        metadata = {}
        assert _meta_str(metadata, "user_id", "default-user") == "default-user"

    def test_returns_default_when_value_empty(self):
        """Returns default when value is empty string."""
        metadata = {"user_id": ""}
        assert _meta_str(metadata, "user_id", "default-user") == "default-user"

    def test_returns_default_when_value_whitespace_only(self):
        """Returns default when value is only whitespace."""
        metadata = {"user_id": "   "}
        assert _meta_str(metadata, "user_id", "default-user") == "default-user"

    def test_returns_default_when_value_not_string(self):
        """Returns default when value is not a string."""
        metadata = {"user_id": 123}
        assert _meta_str(metadata, "user_id", "default-user") == "default-user"

    def test_returns_default_when_value_is_none(self):
        """Returns default when value is None."""
        metadata = {"user_id": None}
        assert _meta_str(metadata, "user_id", "default-user") == "default-user"


class TestOptionalThreadIdFunction:
    """Tests for _optional_thread_id helper function."""

    def test_returns_string_value(self):
        """Returns trimmed string when thread_id exists."""
        metadata = {"thread_id": "  thread-123  "}
        assert _optional_thread_id(metadata) == "thread-123"

    def test_returns_none_when_key_missing(self):
        """Returns None when thread_id is not in metadata."""
        metadata = {}
        assert _optional_thread_id(metadata) is None

    def test_returns_none_when_value_empty(self):
        """Returns None when thread_id is empty string."""
        metadata = {"thread_id": ""}
        assert _optional_thread_id(metadata) is None

    def test_returns_none_when_value_whitespace_only(self):
        """Returns None when thread_id is only whitespace."""
        metadata = {"thread_id": "   "}
        assert _optional_thread_id(metadata) is None

    def test_returns_none_when_value_not_string(self):
        """Returns None when thread_id is not a string."""
        metadata = {"thread_id": 123}
        assert _optional_thread_id(metadata) is None


class TestTemplateAgentA2AExecutor:
    """Tests for TemplateAgentA2AExecutor."""

    @pytest.fixture
    def executor(self):
        """Create executor instance."""
        return TemplateAgentA2AExecutor()

    @pytest.fixture
    def mock_context(self):
        """Create mock request context."""
        ctx = MagicMock()
        ctx.get_user_input.return_value = "test prompt"
        ctx.task_id = "task-123"
        ctx.context_id = "ctx-123"
        ctx.metadata = {}
        return ctx

    @pytest.fixture
    def spy_event_queue(self):
        """Create event queue that captures events."""
        events = []

        class SpyQueue:
            async def enqueue_event(self, event):
                events.append(event)

        return SpyQueue(), events

    @pytest.fixture(autouse=True)
    def setup_a2a_context(self):
        """Setup A2A request context for tests."""
        ctx = A2ARequestContext(access_token="test-token")
        token = a2a_request_ctx.set(ctx)
        yield
        a2a_request_ctx.reset(token)

    async def test_empty_prompt_completes_immediately(
        self, executor, mock_context, spy_event_queue
    ):
        """Empty prompt returns completed task with message."""
        mock_context.get_user_input.return_value = "   "
        queue, events = spy_event_queue

        await executor.execute(mock_context, queue)

        assert len(events) == 2
        task_event = events[0]
        assert isinstance(task_event, Task)
        assert task_event.status.state == TaskState.TASK_STATE_COMPLETED

        status_event = events[1]
        assert isinstance(status_event, TaskStatusUpdateEvent)
        assert "No text message" in status_event.status.message.parts[0].text

    async def test_agent_manager_init_failure(
        self, executor, mock_context, spy_event_queue
    ):
        """AgentManager initialization failure results in FAILED task."""
        queue, events = spy_event_queue

        with patch(
            "template_agent.src.core.a2a_executor.AgentManager",
            side_effect=Exception("Init failed"),
        ):
            await executor.execute(mock_context, queue)

        task_events = [e for e in events if isinstance(e, Task)]
        assert len(task_events) == 1
        assert task_events[0].status.state == TaskState.TASK_STATE_FAILED

        status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
        assert len(status_events) == 1
        assert "initialization failed" in status_events[0].status.message.parts[0].text

    async def test_stream_response_exception(
        self, executor, mock_context, spy_event_queue
    ):
        """Exception during stream_response results in FAILED status."""
        queue, events = spy_event_queue

        async def _failing_stream(request):
            yield {"type": "token", "content": "partial"}
            raise RuntimeError("Stream crashed")

        with patch("template_agent.src.core.a2a_executor.AgentManager") as MockManager:
            instance = MockManager.return_value
            instance.stream_response = _failing_stream
            await executor.execute(mock_context, queue)

        status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
        failed = [
            e for e in status_events if e.status.state == TaskState.TASK_STATE_FAILED
        ]
        assert len(failed) == 1
        assert "Stream crashed" in failed[0].status.message.parts[0].text

    async def test_no_ai_response_returns_placeholder(
        self, executor, mock_context, spy_event_queue
    ):
        """When no AI response is received, returns placeholder text."""
        queue, events = spy_event_queue

        async def _empty_stream(request):
            yield {"type": "token", "content": ""}

        with patch("template_agent.src.core.a2a_executor.AgentManager") as MockManager:
            instance = MockManager.return_value
            instance.stream_response = _empty_stream
            await executor.execute(mock_context, queue)

        artifact_events = [e for e in events if isinstance(e, TaskArtifactUpdateEvent)]
        final = [e for e in artifact_events if e.last_chunk]
        assert len(final) == 1
        assert "(no assistant response)" in final[0].artifact.parts[0].text

    async def test_metadata_extraction(self, executor, spy_event_queue):
        """User ID, session ID, and thread ID are extracted from metadata."""
        mock_context = MagicMock()
        mock_context.get_user_input.return_value = "test"
        mock_context.task_id = "task-1"
        mock_context.context_id = "ctx-1"
        mock_context.metadata = {
            "user_id": "custom-user",
            "session_id": "custom-session",
            "thread_id": "custom-thread",
        }
        queue, events = spy_event_queue

        captured_request = None

        async def _capture_stream(request):
            nonlocal captured_request
            captured_request = request
            yield {"type": "message", "content": {"type": "ai", "content": "ok"}}

        with patch("template_agent.src.core.a2a_executor.AgentManager") as MockManager:
            instance = MockManager.return_value
            instance.stream_response = _capture_stream
            await executor.execute(mock_context, queue)

        assert captured_request is not None
        assert captured_request.user_id == "custom-user"
        assert captured_request.session_id == "custom-session"
        assert captured_request.thread_id == "custom-thread"

    async def test_correlation_id_logged(self, executor, mock_context, spy_event_queue):
        """Correlation ID from context is logged."""
        queue, events = spy_event_queue

        ctx = A2ARequestContext(
            access_token="token",
            correlation_id="corr-test-123",
            calling_agent_id="test-agent",
        )
        token = a2a_request_ctx.set(ctx)

        async def _simple_stream(request):
            yield {"type": "message", "content": {"type": "ai", "content": "ok"}}

        try:
            with patch(
                "template_agent.src.core.a2a_executor.AgentManager"
            ) as MockManager:
                instance = MockManager.return_value
                instance.stream_response = _simple_stream
                with patch(
                    "template_agent.src.core.a2a_executor.logger"
                ) as mock_logger:
                    await executor.execute(mock_context, queue)
                    mock_logger.info.assert_any_call(
                        "a2a_execute",
                        correlation_id="corr-test-123",
                        calling_agent=("test-agent"),
                    )
        finally:
            a2a_request_ctx.reset(token)

    async def test_cancel_is_noop(self, executor, mock_context, spy_event_queue):
        """Cancel method is a no-op (best effort)."""
        queue, _ = spy_event_queue
        await executor.cancel(mock_context, queue)

    async def test_token_streaming(self, executor, mock_context, spy_event_queue):
        """Token events are streamed as artifact updates."""
        queue, events = spy_event_queue

        async def _token_stream(request):
            yield {"type": "token", "content": "Hello"}
            yield {"type": "token", "content": " world"}
            yield {
                "type": "message",
                "content": {"type": "ai", "content": "Hello world"},
            }

        with patch("template_agent.src.core.a2a_executor.AgentManager") as MockManager:
            instance = MockManager.return_value
            instance.stream_response = _token_stream
            await executor.execute(mock_context, queue)

        artifact_events = [e for e in events if isinstance(e, TaskArtifactUpdateEvent)]
        append_events = [e for e in artifact_events if e.append]
        assert len(append_events) == 2

    async def test_error_event_in_stream(self, executor, mock_context, spy_event_queue):
        """Error event in stream results in FAILED status."""
        queue, events = spy_event_queue

        async def _error_stream(request):
            yield {"type": "error", "content": {"message": "Something broke"}}

        with patch("template_agent.src.core.a2a_executor.AgentManager") as MockManager:
            instance = MockManager.return_value
            instance.stream_response = _error_stream
            await executor.execute(mock_context, queue)

        status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
        failed = [
            e for e in status_events if e.status.state == TaskState.TASK_STATE_FAILED
        ]
        assert len(failed) == 1
        assert "Something broke" in failed[0].status.message.parts[0].text
