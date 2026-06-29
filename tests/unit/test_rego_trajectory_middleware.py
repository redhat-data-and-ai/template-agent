"""Unit tests for RegoTrajectoryMiddleware."""

import pytest
from unittest.mock import MagicMock, patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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

    def test_parses_user_and_agent_messages(self):
        middleware = RegoTrajectoryMiddleware()
        messages = [
            HumanMessage(content="What is bmi?"),
            AIMessage(content="BMI stands for Body Mass Index"),
        ]

        trajectory = middleware._parse_trajectory(messages)

        assert trajectory == [
            {"type": "user_message", "content": "What is bmi?"},
            {"type": "agent_response", "content": "BMI stands for Body Mass Index"},
        ]

    def test_ignores_empty_messages(self):
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
            allowed, reasons = middleware._evaluate_policy([], {"action": "llm_request"})

        assert allowed is True
        assert reasons == []
        mock_post.assert_called_once()
        # Verify payload does not include user_settings
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert "user_settings" not in payload["input"]
        assert "trajectory" in payload["input"]
        assert "current_intent" in payload["input"]

    def test_returns_false_when_opa_denies(self):
        middleware = RegoTrajectoryMiddleware()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "allow": False,
                "deny_reasons": ["Banned word 'bmi' found in user message"]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch(
            "deep_agent.src.infrastructure.rego_trajectory_middleware.httpx.post",
            return_value=mock_response,
        ):
            allowed, reasons = middleware._evaluate_policy([], {"action": "llm_request"})
            assert allowed is False
            assert reasons == ["Banned word 'bmi' found in user message"]

    def test_fail_closed_on_http_error(self):
        middleware = RegoTrajectoryMiddleware()

        with patch(
            "deep_agent.src.infrastructure.rego_trajectory_middleware.httpx.post",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            allowed, reasons = middleware._evaluate_policy([], {"action": "llm_request"})
            assert allowed is False
            assert "technical error" in reasons[0].lower()


class TestModelCallWrapping:
    @pytest.mark.asyncio
    async def test_wrap_model_call_denies_banned_content(self):
        """Test that awrap_model_call intercepts and replaces denied LLM responses."""
        from langchain.agents.middleware.types import ModelResponse, ModelRequest
        from langchain_core.messages import AIMessage as LCCoreAIMessage

        middleware = RegoTrajectoryMiddleware()

        # Create a mock request
        request = MagicMock(spec=ModelRequest)
        request.state = {"messages": [HumanMessage(content="What is BMI?")]}

        # Create a mock handler that returns a response with banned content
        async def mock_handler(_req):
            return ModelResponse(
                result=[LCCoreAIMessage(content="BMI is Body Mass Index")],
                structured_response=None,
            )

        denial_reasons = ["Banned word 'BMI' found in agent response"]
        with patch.object(middleware, "_evaluate_policy", return_value=(False, denial_reasons)):
            result = await middleware.awrap_model_call(request, mock_handler)

        # Check that the response was replaced with a denial
        assert isinstance(result, ModelResponse)
        assert len(result.result) == 1
        denied_msg = result.result[0]
        assert POLICY_DENIAL_MESSAGE in denied_msg.content
        assert "Banned word 'BMI' found in agent response" in denied_msg.content

    @pytest.mark.asyncio
    async def test_wrap_model_call_allows_compliant_content(self):
        """Test that awrap_model_call passes through allowed LLM responses."""
        from langchain.agents.middleware.types import ModelResponse, ModelRequest
        from langchain_core.messages import AIMessage as LCCoreAIMessage

        middleware = RegoTrajectoryMiddleware()

        request = MagicMock(spec=ModelRequest)
        request.state = {"messages": [HumanMessage(content="Hello")]}

        expected_response = ModelResponse(
            result=[LCCoreAIMessage(content="Hello! How can I help you?")],
            structured_response=None,
        )

        async def mock_handler(_req):
            return expected_response

        with patch.object(middleware, "_evaluate_policy", return_value=(True, [])):
            result = await middleware.awrap_model_call(request, mock_handler)

        # Check that the original response was returned unchanged
        assert result == expected_response


class TestHooks:
    @pytest.mark.asyncio
    async def test_before_model_returns_denial_message_on_policy_violation(self):
        middleware = RegoTrajectoryMiddleware()
        state = {
            "messages": [HumanMessage(content="What is bmi?")]
        }
        denial_reasons = ["Banned word 'bmi' found in user message"]
        with patch.object(middleware, "_evaluate_policy", return_value=(False, denial_reasons)):
            result = await middleware.abefore_model(state, runtime=None)

        assert result["jump_to"] == "end"
        assert len(result["messages"]) == 1
        message_content = result["messages"][0].content
        assert POLICY_DENIAL_MESSAGE in message_content
        assert "Banned word 'bmi' found in user message" in message_content

    @pytest.mark.asyncio
    async def test_wrap_tool_call_returns_denial_command_on_policy_violation(self):
        middleware = RegoTrajectoryMiddleware()
        request = MagicMock()
        request.state = {"messages": []}
        request.tool_call = {"name": "send_email", "args": {"to": "a@b.com"}, "id": "tc-1"}
        handler = MagicMock()

        denial_reasons = ["Email tool is not allowed"]
        with patch.object(middleware, "_evaluate_policy", return_value=(False, denial_reasons)):
            result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, Command)
        assert result.goto == "end"
        messages = result.update["messages"]

        # Check tool message has denial reasons
        tool_msg = messages[0]
        assert tool_msg.tool_call_id == "tc-1"
        assert tool_msg.name == "send_email"
        assert tool_msg.status == "error"
        assert POLICY_DENIAL_MESSAGE in tool_msg.content
        assert "Email tool is not allowed" in tool_msg.content

        # Check AI message has denial reasons
        ai_msg = messages[1]
        assert isinstance(ai_msg, AIMessage)
        assert POLICY_DENIAL_MESSAGE in ai_msg.content
        assert "Email tool is not allowed" in ai_msg.content

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_wrap_tool_call_delegates_when_allowed(self):
        middleware = RegoTrajectoryMiddleware()
        request = MagicMock()
        request.state = {"messages": []}
        request.tool_call = {"name": "send_email", "args": {}}

        async def async_handler(_req):
            return "tool-result"

        with patch.object(middleware, "_evaluate_policy", return_value=(True, [])):
            result = await middleware.awrap_tool_call(request, async_handler)

        assert result == "tool-result"

    @pytest.mark.asyncio
    async def test_after_model_returns_denial_message_on_policy_violation(self):
        middleware = RegoTrajectoryMiddleware()
        state = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Here is some info about BMI")
            ]
        }
        denial_reasons = ["Banned word 'BMI' found in agent response"]
        with patch.object(middleware, "_evaluate_policy", return_value=(False, denial_reasons)):
            result = await middleware.aafter_model(state, runtime=None)

        assert result["jump_to"] == "end"
        # Should have: HumanMessage + Denial AIMessage (original AI message removed)
        assert len(result["messages"]) == 2
        assert result["messages"][0].content == "Hello"  # Original human message preserved
        denial_content = result["messages"][1].content
        assert POLICY_DENIAL_MESSAGE in denial_content
        assert "Banned word 'BMI' found in agent response" in denial_content

    @pytest.mark.asyncio
    async def test_after_model_allows_compliant_response(self):
        middleware = RegoTrajectoryMiddleware()
        state = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hello! How can I help you?")
            ]
        }
        with patch.object(middleware, "_evaluate_policy", return_value=(True, [])):
            result = await middleware.aafter_model(state, runtime=None)

        assert result is None

    @pytest.mark.asyncio
    async def test_after_tool_call_returns_denial_on_policy_violation(self):
        middleware = RegoTrajectoryMiddleware()
        request = MagicMock()
        request.state = {"messages": []}
        request.tool_call = {"name": "search", "args": {}, "id": "tc-1"}

        tool_result = ToolMessage(
            content="Results contain BMI information",
            tool_call_id="tc-1",
            name="search"
        )

        denial_reasons = ["Banned word 'BMI' found in tool result"]
        with patch.object(middleware, "_evaluate_policy", return_value=(False, denial_reasons)):
            result = await middleware.aafter_tool_call(tool_result, request)

        assert isinstance(result, Command)
        assert result.goto == "end"
        messages = result.update["messages"]
        assert POLICY_DENIAL_MESSAGE in messages[0].content
        assert "Banned word 'BMI' found in tool result" in messages[0].content

    @pytest.mark.asyncio
    async def test_after_tool_call_allows_compliant_result(self):
        middleware = RegoTrajectoryMiddleware()
        request = MagicMock()
        request.state = {"messages": []}
        request.tool_call = {"name": "search", "args": {}}

        tool_result = ToolMessage(
            content="Search results",
            tool_call_id="tc-1",
            name="search"
        )

        with patch.object(middleware, "_evaluate_policy", return_value=(True, [])):
            result = await middleware.aafter_tool_call(tool_result, request)

        assert result == tool_result


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
