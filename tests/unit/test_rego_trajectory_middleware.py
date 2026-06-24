"""Unit tests for RegoTrajectoryMiddleware."""

from unittest.mock import MagicMock, patch

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from deep_agent.src.agent.config.middleware import RegoTrajectoryConfig, ResolvedMiddlewareConfig
from deep_agent.src.infrastructure.middleware import build_middleware_list
from deep_agent.src.infrastructure.rego_trajectory_middleware import (
    POLICY_DENIAL_MESSAGE,
    RegoTrajectoryMiddleware,
)


class TestParseTrajectory:
    def test_parses_ai_tool_calls_and_tool_responses(self):
        middleware = RegoTrajectoryMiddleware()
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_web", "args": {"q": "bmi"}, "id": "1", "type": "tool_call"}
                ],
            ),
            ToolMessage(content="ok", name="search_web", tool_call_id="1"),
        ]

        trajectory = middleware._parse_trajectory(messages)

        assert trajectory == [
            {
                "type": "agent_action",
                "tools": [{"name": "search_web", "args": {"q": "bmi"}}],
            },
            {"type": "tool_response", "name": "search_web", "status": "completed"},
        ]

    def test_ignores_non_tool_messages(self):
        middleware = RegoTrajectoryMiddleware()
        assert middleware._parse_trajectory([]) == []


class TestEvaluatePolicy:
    def test_returns_true_when_opa_allows(self):
        middleware = RegoTrajectoryMiddleware(opa_url="http://opa:8181/v1/data/agent/authz")
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"allow": True}}
        mock_response.raise_for_status = MagicMock()

        with patch(
            "deep_agent.src.infrastructure.rego_trajectory_middleware.httpx.post",
            return_value=mock_response,
        ) as mock_post:
            allowed = middleware._evaluate_policy([], {"action": "llm_request"})

        assert allowed is True
        mock_post.assert_called_once()

    def test_returns_false_when_opa_denies(self):
        middleware = RegoTrajectoryMiddleware()
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"allow": False}}
        mock_response.raise_for_status = MagicMock()

        with patch(
            "deep_agent.src.infrastructure.rego_trajectory_middleware.httpx.post",
            return_value=mock_response,
        ):
            assert middleware._evaluate_policy([], {"action": "llm_request"}) is False

    def test_fail_closed_on_http_error(self):
        middleware = RegoTrajectoryMiddleware()

        with patch(
            "deep_agent.src.infrastructure.rego_trajectory_middleware.httpx.post",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            assert middleware._evaluate_policy([], {"action": "llm_request"}) is False


class TestHooks:
    def test_before_model_returns_denial_message_on_policy_violation(self):
        middleware = RegoTrajectoryMiddleware()
        with patch.object(middleware, "_evaluate_policy", return_value=False):
            result = middleware.before_model({"messages": []}, runtime=None)

        assert result == {
            "jump_to": "end",
            "messages": [AIMessage(content=POLICY_DENIAL_MESSAGE)],
        }

    def test_wrap_tool_call_returns_denial_command_on_policy_violation(self):
        middleware = RegoTrajectoryMiddleware()
        request = MagicMock()
        request.state = {"messages": []}
        request.tool_call = {"name": "send_email", "args": {"to": "a@b.com"}, "id": "tc-1"}
        handler = MagicMock()

        with patch.object(middleware, "_evaluate_policy", return_value=False):
            result = middleware.wrap_tool_call(request, handler)

        assert isinstance(result, Command)
        assert result.goto == "end"
        messages = result.update["messages"]
        assert messages[0] == ToolMessage(
            content=POLICY_DENIAL_MESSAGE,
            tool_call_id="tc-1",
            name="send_email",
            status="error",
        )
        assert messages[1] == AIMessage(content=POLICY_DENIAL_MESSAGE)
        handler.assert_not_called()

    def test_wrap_tool_call_delegates_when_allowed(self):
        middleware = RegoTrajectoryMiddleware()
        request = MagicMock()
        request.state = {"messages": []}
        request.tool_call = {"name": "send_email", "args": {}}
        handler = MagicMock(return_value="tool-result")

        with patch.object(middleware, "_evaluate_policy", return_value=True):
            result = middleware.wrap_tool_call(request, handler)

        assert result == "tool-result"
        handler.assert_called_once_with(request)


class TestMiddlewareBuilder:
    def test_includes_rego_trajectory_when_enabled(self):
        resolved = ResolvedMiddlewareConfig(
            summarization_tool_enabled=False,
            rego_trajectory=RegoTrajectoryConfig(enabled=True),
        )
        with patch(
            "deep_agent.src.infrastructure.middleware.settings"
        ) as mock_settings:
            mock_settings.MIDDLEWARE_ENABLED = True
            mock_settings.OPA_URL = "http://opa:8181/v1/data/agent/authz"
            result = build_middleware_list(resolved)

        assert any(isinstance(mw, RegoTrajectoryMiddleware) for mw in result)

    def test_skips_rego_trajectory_when_disabled(self):
        resolved = ResolvedMiddlewareConfig(
            summarization_tool_enabled=False,
            rego_trajectory=RegoTrajectoryConfig(enabled=False),
            extra_middleware=[],
        )
        with patch(
            "deep_agent.src.infrastructure.middleware.settings"
        ) as mock_settings:
            mock_settings.MIDDLEWARE_ENABLED = True
            result = build_middleware_list(resolved)

        assert not any(isinstance(mw, RegoTrajectoryMiddleware) for mw in result)
