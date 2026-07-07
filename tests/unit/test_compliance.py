"""Unit tests for compliance middleware."""

import pytest
from unittest.mock import MagicMock, patch

import httpx
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from deep_agent.src.agent.config.middleware import RegoTrajectoryConfig, ResolvedMiddlewareConfig
from deep_agent.src.infrastructure.middleware import build_middleware_list
from deep_agent.src.infrastructure.compliance import (
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
            "deep_agent.src.infrastructure.compliance.httpx.post",
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
            "deep_agent.src.infrastructure.compliance.httpx.post",
            return_value=mock_response,
        ):
            allowed, reasons = middleware._evaluate_policy([], {"action": "llm_request"})
            assert allowed is False
            assert reasons == ["Banned word 'bmi' found in user message"]

    def test_fail_closed_on_http_error(self):
        middleware = RegoTrajectoryMiddleware()

        with patch(
            "deep_agent.src.infrastructure.compliance.httpx.post",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            allowed, reasons = middleware._evaluate_policy([], {"action": "llm_request"})
            assert allowed is False
            assert "technical error" in reasons[0].lower()


class TestHooks:
    @pytest.mark.asyncio
    async def test_before_model_validates_entire_trajectory(self):
        """Test that abefore_model evaluates the complete trajectory."""
        middleware = RegoTrajectoryMiddleware()
        state = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
                HumanMessage(content="Tell me about BMI"),
            ]
        }
        denial_reasons = ["Trajectory contains banned content"]
        with patch.object(middleware, "_evaluate_policy", return_value=(False, denial_reasons)) as mock_eval:
            result = await middleware.abefore_model(state, runtime=None)

        # Verify the policy was called with trajectory validation intent
        call_args = mock_eval.call_args
        assert call_args[0][1]["action"] == "trajectory_validation"
        assert call_args[0][1]["trajectory_length"] == 3

        # Verify denial response
        assert result["jump_to"] == "end"
        assert len(result["messages"]) == 1
        message_content = result["messages"][0].content
        assert POLICY_DENIAL_MESSAGE in message_content
        assert "Trajectory contains banned content" in message_content

    @pytest.mark.asyncio
    async def test_before_model_allows_compliant_trajectory(self):
        """Test that abefore_model allows compliant trajectories."""
        middleware = RegoTrajectoryMiddleware()
        state = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi! How can I help?"),
            ]
        }
        with patch.object(middleware, "_evaluate_policy", return_value=(True, [])) as mock_eval:
            result = await middleware.abefore_model(state, runtime=None)

        # Verify the policy was called
        call_args = mock_eval.call_args
        assert call_args[0][1]["action"] == "trajectory_validation"

        # Verify no blocking occurred
        assert result is None

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
    async def test_after_tool_call_offers_retry_on_first_violation(self):
        """First tool violation should offer retry prompt."""
        middleware = RegoTrajectoryMiddleware(enable_retry=True)
        request = MagicMock()
        request.state = {"messages": []}
        request.tool_call = {"name": "search", "args": {}, "id": "tc-1"}

        tool_result = ToolMessage(
            content="Results contain BMI information",
            tool_call_id="tc-1",
            name="search"
        )

        # Mock handler that returns the tool result
        async def mock_handler(req):
            return tool_result

        denial_reasons = ["Banned word 'BMI' found in tool result"]
        with patch.object(middleware, "_evaluate_policy", return_value=(False, denial_reasons)):
            result = await middleware.awrap_tool_call(request, mock_handler)

        assert isinstance(result, Command)
        assert result.goto == "end"
        messages = result.update["messages"]
        assert "Retry the prompt :" in messages[0].content
        assert "search" in messages[0].content
        assert "Banned word 'BMI' found in tool result" in messages[0].content
        assert messages[0].content == messages[1].content
        # Should store violation context
        assert result.update["policy_violation_context"]["retry_available"] is True
        assert result.update["policy_violation_context"]["checkpoint"] == "awrap_tool_call"

    @pytest.mark.asyncio
    async def test_after_tool_call_no_retry_on_second_violation(self):
        """Second tool violation should show final denial."""
        middleware = RegoTrajectoryMiddleware(enable_retry=True)
        request = MagicMock()
        request.state = {
            "messages": [],
            "policy_violation_context": {
                "retry_available": False,
                "checkpoint": "awrap_tool_call",
                "denial_reasons": ["Previous violation"]
            }
        }
        request.tool_call = {"name": "search", "args": {}, "id": "tc-1"}

        tool_result = ToolMessage(
            content="Results contain BMI information again",
            tool_call_id="tc-1",
            name="search"
        )

        # Mock handler that returns the tool result
        async def mock_handler(req):
            return tool_result

        denial_reasons = ["Banned word 'BMI' found in tool result"]
        with patch.object(middleware, "_evaluate_policy", return_value=(False, denial_reasons)):
            result = await middleware.awrap_tool_call(request, mock_handler)

        assert isinstance(result, Command)
        assert result.goto == "end"
        messages = result.update["messages"]
        # Should contain final denial, not retry prompt with blocked input
        assert POLICY_DENIAL_MESSAGE in messages[0].content
        assert "Retry the prompt :" not in messages[0].content
        assert result.update["policy_violation_context"]["retry_available"] is False

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

        # Mock handler that returns the tool result
        async def mock_handler(req):
            return tool_result

        with patch.object(middleware, "_evaluate_policy", return_value=(True, [])):
            result = await middleware.awrap_tool_call(request, mock_handler)

        assert result == tool_result


class TestWrapModelCall:
    @pytest.mark.asyncio
    async def test_awrap_model_call_passes_nostream_tags_to_handler(self):
        middleware = RegoTrajectoryMiddleware()
        request = ModelRequest(
            model=MagicMock(),
            tools=[],
            system_message=None,
            response_format=None,
            messages=[HumanMessage(content="Hello")],
            tool_choice=None,
            state={"messages": [HumanMessage(content="Hello")]},
            runtime=MagicMock(),
            model_settings={},
        )
        handler_request: ModelRequest | None = None

        async def mock_handler(req: ModelRequest) -> ModelResponse:
            nonlocal handler_request
            handler_request = req
            return ModelResponse(result=[AIMessage(content="Hi there!")])

        with patch.object(middleware, "_evaluate_policy", return_value=(True, [])):
            await middleware.awrap_model_call(request, mock_handler)

        assert handler_request is not None
        assert "nostream" in handler_request.model_settings["tags"]
        assert "langsmith:nostream" in handler_request.model_settings["tags"]

    @pytest.mark.asyncio
    async def test_awrap_model_call_allows_compliant_response(self):
        middleware = RegoTrajectoryMiddleware()
        request = ModelRequest(
            model=MagicMock(),
            tools=[],
            system_message=None,
            response_format=None,
            messages=[HumanMessage(content="Hello")],
            tool_choice=None,
            state={"messages": [HumanMessage(content="Hello")]},
            runtime=MagicMock(),
        )
        original = ModelResponse(result=[AIMessage(content="Hello! How can I help you?")])

        async def mock_handler(_req: ModelRequest) -> ModelResponse:
            return original

        with patch.object(middleware, "_evaluate_policy", return_value=(True, [])):
            result = await middleware.awrap_model_call(request, mock_handler)

        assert result is original

    @pytest.mark.asyncio
    async def test_awrap_model_call_replaces_violating_response(self):
        middleware = RegoTrajectoryMiddleware(enable_retry=True)
        request = ModelRequest(
            model=MagicMock(),
            tools=[],
            system_message=None,
            response_format=None,
            messages=[HumanMessage(content="Tell me about BMI")],
            tool_choice=None,
            state={"messages": [HumanMessage(content="Tell me about BMI")]},
            runtime=MagicMock(),
        )
        denial_reasons = ["Banned word 'BMI' found in agent response"]

        async def mock_handler(_req: ModelRequest) -> ModelResponse:
            return ModelResponse(result=[AIMessage(content="BMI stands for Body Mass Index")])

        with patch.object(middleware, "_evaluate_policy", return_value=(False, denial_reasons)):
            result = await middleware.awrap_model_call(request, mock_handler)

        assert isinstance(result, ExtendedModelResponse)
        assert len(result.model_response.result) == 1
        denial_content = result.model_response.result[0].content
        assert POLICY_DENIAL_MESSAGE in denial_content
        assert "Banned word 'BMI' found in agent response" in denial_content
        assert "Retry the prompt :" in denial_content
        assert "BMI stands for Body Mass Index" in denial_content
        assert result.command is not None
        context = result.command.update["policy_violation_context"]
        assert context["checkpoint"] == "awrap_model_call"
        assert context["retry_available"] is True
        assert context["blocked_input"] == "BMI stands for Body Mass Index"

    @pytest.mark.asyncio
    async def test_awrap_model_call_skips_empty_ai_content(self):
        middleware = RegoTrajectoryMiddleware()
        request = ModelRequest(
            model=MagicMock(),
            tools=[],
            system_message=None,
            response_format=None,
            messages=[HumanMessage(content="Search")],
            tool_choice=None,
            state={"messages": [HumanMessage(content="Search")]},
            runtime=MagicMock(),
        )
        tool_call_response = ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_web",
                            "args": {"q": "bmi"},
                            "id": "1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )

        async def mock_handler(_req: ModelRequest) -> ModelResponse:
            return tool_call_response

        with patch.object(middleware, "_evaluate_policy") as mock_eval:
            result = await middleware.awrap_model_call(request, mock_handler)

        assert result is tool_call_response
        mock_eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_awrap_model_call_uses_hardcoded_response_when_configured(self):
        middleware = RegoTrajectoryMiddleware()
        request = ModelRequest(
            model=MagicMock(),
            tools=[],
            system_message=None,
            response_format=None,
            messages=[HumanMessage(content="Tell me about BMI")],
            tool_choice=None,
            state={"messages": [HumanMessage(content="Tell me about BMI")]},
            runtime=MagicMock(),
        )
        handler_called = False

        async def mock_handler(_req: ModelRequest) -> ModelResponse:
            nonlocal handler_called
            handler_called = True
            return ModelResponse(result=[AIMessage(content="Should not be used")])

        with patch(
            "deep_agent.src.infrastructure.compliance.settings"
        ) as mock_settings:
            mock_settings.COMPLIANCE_HARDCODED_MODEL_RESPONSE = (
                "BMI stands for Body Mass Index"
            )
            with patch.object(middleware, "_evaluate_policy", return_value=(True, [])):
                result = await middleware.awrap_model_call(request, mock_handler)

        assert handler_called is False
        assert result.result[0].content == "BMI stands for Body Mass Index"

    @pytest.mark.asyncio
    async def test_aafter_model_skips_when_handled_by_awrap_model_call(self):
        middleware = RegoTrajectoryMiddleware()
        state = {
            "messages": [
                HumanMessage(content="Tell me about BMI"),
                AIMessage(content=POLICY_DENIAL_MESSAGE),
            ],
            "policy_violation_context": {
                "checkpoint": "awrap_model_call",
                "denial_reasons": ["Banned word 'BMI' found in agent response"],
                "retry_available": True,
            },
        }

        with patch.object(middleware, "_evaluate_policy") as mock_eval:
            result = await middleware.aafter_model(state, runtime=None)

        assert result is None
        mock_eval.assert_not_called()


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


class TestRetryMechanism:
    """Tests for HITL retry mechanism when LLM responses violate policies."""

    @pytest.mark.asyncio
    async def test_aafter_model_offers_retry_on_first_violation(self):
        """First LLM violation should offer retry prompt."""
        middleware = RegoTrajectoryMiddleware(enable_retry=True)
        state = {
            "messages": [
                HumanMessage(content="Tell me about BMI"),
                AIMessage(content="BMI stands for Body Mass Index")
            ]
        }
        denial_reasons = ["Banned word 'BMI' found in agent response"]

        with patch.object(middleware, "_evaluate_policy", return_value=(False, denial_reasons)):
            result = await middleware.aafter_model(state, runtime=None)

        assert result["jump_to"] == "end"
        assert len(result["messages"]) == 2
        retry_content = result["messages"][1].content
        assert "Retry the prompt :" in retry_content
        assert "BMI stands for Body Mass Index" in retry_content
        assert "Banned word 'BMI' found in agent response" in retry_content
        # Should store violation context
        assert result["policy_violation_context"]["retry_available"] is True
        assert result["policy_violation_context"]["checkpoint"] == "aafter_model"
        assert result["policy_violation_context"]["denial_reasons"] == denial_reasons

    @pytest.mark.asyncio
    async def test_aafter_model_no_retry_when_disabled(self):
        """When retry is disabled, should show final denial immediately."""
        middleware = RegoTrajectoryMiddleware(enable_retry=False)
        state = {
            "messages": [
                HumanMessage(content="Tell me about BMI"),
                AIMessage(content="BMI stands for Body Mass Index")
            ]
        }
        denial_reasons = ["Banned word 'BMI' found in agent response"]

        with patch.object(middleware, "_evaluate_policy", return_value=(False, denial_reasons)):
            result = await middleware.aafter_model(state, runtime=None)

        assert result["jump_to"] == "end"
        # Should contain final denial, not retry prompt with blocked input
        denial_content = result["messages"][1].content
        assert POLICY_DENIAL_MESSAGE in denial_content
        assert "Retry the prompt :" not in denial_content
        assert result["policy_violation_context"]["retry_available"] is False

    @pytest.mark.asyncio
    async def test_aafter_model_no_retry_on_second_violation(self):
        """Second violation (after retry) should show final denial."""
        middleware = RegoTrajectoryMiddleware(enable_retry=True)
        state = {
            "messages": [
                HumanMessage(content="Tell me about BMI"),
                AIMessage(content="BMI stands for Body Mass Index again")
            ],
            "policy_violation_context": {
                "retry_available": False,  # Retry already used
                "checkpoint": "aafter_model",
                "denial_reasons": ["First violation"]
            }
        }
        denial_reasons = ["Banned word 'BMI' found in agent response"]

        with patch.object(middleware, "_evaluate_policy", return_value=(False, denial_reasons)):
            result = await middleware.aafter_model(state, runtime=None)

        assert result["jump_to"] == "end"
        # Should contain final denial, not retry prompt with blocked input
        denial_content = result["messages"][1].content
        assert POLICY_DENIAL_MESSAGE in denial_content
        assert "Retry the prompt :" not in denial_content
        assert result["policy_violation_context"]["retry_available"] is False

    @pytest.mark.asyncio
    async def test_abefore_model_detects_retry_keyword_yes(self):
        """Detect 'yes' as retry confirmation."""
        middleware = RegoTrajectoryMiddleware(enable_retry=True)
        state = {
            "messages": [
                HumanMessage(content="Tell me about health"),
                HumanMessage(content="yes")
            ],
            "policy_violation_context": {
                "retry_available": True,
                "checkpoint": "aafter_model",
                "denial_reasons": ["Previous violation"]
            }
        }

        with patch.object(middleware, "_evaluate_policy", return_value=(True, [])):
            result = await middleware.abefore_model(state, runtime=None)

        # Should inject SystemMessage with policy context
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], SystemMessage)
        assert "IMPORTANT POLICY CONTEXT" in result["messages"][0].content
        assert "Previous violation" in result["messages"][0].content
        # Should mark retry as used
        assert result["policy_violation_context"]["retry_available"] is False

    @pytest.mark.asyncio
    async def test_abefore_model_detects_retry_keyword_retry(self):
        """Detect 'retry' as retry confirmation."""
        middleware = RegoTrajectoryMiddleware(enable_retry=True)
        state = {
            "messages": [
                HumanMessage(content="Tell me about health"),
                HumanMessage(content="Please retry")
            ],
            "policy_violation_context": {
                "retry_available": True,
                "checkpoint": "aafter_model",
                "denial_reasons": ["Sensitive topic detected"]
            }
        }

        with patch.object(middleware, "_evaluate_policy", return_value=(True, [])):
            result = await middleware.abefore_model(state, runtime=None)

        assert "messages" in result
        assert isinstance(result["messages"][0], SystemMessage)
        assert "Sensitive topic detected" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_abefore_model_triggers_on_any_user_message(self):
        """Any user message after violation should trigger retry (no keywords needed)."""
        middleware = RegoTrajectoryMiddleware(enable_retry=True)
        state = {
            "messages": [
                HumanMessage(content="Tell me about health"),
                HumanMessage(content="What about nutrition?")
            ],
            "policy_violation_context": {
                "retry_available": True,
                "checkpoint": "aafter_model",
                "denial_reasons": ["Previous violation"]
            }
        }

        with patch.object(middleware, "_evaluate_policy", return_value=(True, [])):
            result = await middleware.abefore_model(state, runtime=None)

        # Should inject policy context for ANY user message after violation
        assert "messages" in result
        assert isinstance(result["messages"][0], SystemMessage)
        assert "Previous violation" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_abefore_model_no_retry_when_not_available(self):
        """Should not inject policy context when retry_available is False."""
        middleware = RegoTrajectoryMiddleware(enable_retry=True)
        state = {
            "messages": [
                HumanMessage(content="Tell me about health"),
                HumanMessage(content="yes")
            ],
            "policy_violation_context": {
                "retry_available": False,  # Retry already used
                "checkpoint": "aafter_model",
                "denial_reasons": ["Previous violation"]
            }
        }

        with patch.object(middleware, "_evaluate_policy", return_value=(True, [])):
            result = await middleware.abefore_model(state, runtime=None)

        # Should not inject policy context
        assert result is None
