"""Unit tests for schema models."""

import pytest
from pydantic import ValidationError

from deep_agent.src.schema import (
    ChatHistoryResponse,
    ChatMessage,
    FeedbackRequest,
    FeedbackResponse,
    Project,
    ProjectCreate,
    ProjectUpdate,
    StreamRequest,
    UserInput,
)


class TestUserInput:
    def test_required_message(self):
        inp = UserInput(message="hello")
        assert inp.message == "hello"

    def test_optional_fields_default_none(self):
        inp = UserInput(message="hi")
        assert inp.thread_id is None
        assert inp.session_id is None
        assert inp.user_id is None

    def test_all_fields(self):
        inp = UserInput(
            message="hello",
            thread_id="t1",
            session_id="s1",
            user_id="u1",
        )
        assert inp.thread_id == "t1"
        assert inp.session_id == "s1"
        assert inp.user_id == "u1"


class TestStreamRequest:
    def test_inherits_user_input(self):
        req = StreamRequest(message="test")
        assert isinstance(req, UserInput)

    def test_default_stream_tokens(self):
        req = StreamRequest(message="test")
        assert req.stream_tokens is True

    def test_stream_tokens_false(self):
        req = StreamRequest(message="test", stream_tokens=False)
        assert req.stream_tokens is False


class TestChatMessage:
    def test_minimal_message(self):
        msg = ChatMessage(type="human", content="hello")
        assert msg.type == "human"
        assert msg.content == "hello"
        assert msg.tool_calls == []
        assert msg.tool_call_id is None
        assert msg.run_id is None
        assert msg.response_metadata == {}
        assert msg.custom_data == {}

    def test_ai_message_with_tool_calls(self):
        msg = ChatMessage(
            type="ai",
            content="",
            tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "tc1"}],
        )
        assert msg.tool_calls[0]["name"] == "search"

    def test_allowed_types(self):
        for t in ("human", "ai", "tool", "custom"):
            msg = ChatMessage(type=t, content="x")
            assert msg.type == t


class TestFeedbackRequest:
    def test_required_fields(self):
        fb = FeedbackRequest(trace_id="abc", name="thumbs-up", value=1.0)
        assert fb.trace_id == "abc"
        assert fb.name == "thumbs-up"
        assert fb.value == 1.0
        assert fb.kwargs == {}

    def test_with_kwargs(self):
        fb = FeedbackRequest(
            trace_id="abc",
            name="rating",
            value=0.8,
            kwargs={"comment": "good"},
        )
        assert fb.kwargs["comment"] == "good"


class TestFeedbackResponse:
    def test_default_status(self):
        resp = FeedbackResponse()
        assert resp.status == "success"


class TestChatHistoryResponse:
    def test_empty_messages(self):
        resp = ChatHistoryResponse(messages=[])
        assert resp.messages == []

    def test_with_messages(self):
        msg = ChatMessage(type="human", content="hi")
        resp = ChatHistoryResponse(messages=[msg])
        assert len(resp.messages) == 1


class TestProjectModels:
    def test_create_requires_name(self):
        with pytest.raises(ValidationError):
            ProjectCreate()

    def test_create_strips_and_rejects_blank_name(self):
        body = ProjectCreate(project_name="  Alpha  ")
        assert body.project_name == "Alpha"
        with pytest.raises(ValidationError):
            ProjectCreate(project_name="   ")

    def test_update_strips_name(self):
        body = ProjectUpdate(project_name="  Beta  ")
        assert body.project_name == "Beta"

    def test_project_defaults_thread_count(self):
        project = Project(
            project_id="p1",
            project_name="Alpha",
            username="u1",
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert project.thread_count == 0
        assert project.project_description is None
