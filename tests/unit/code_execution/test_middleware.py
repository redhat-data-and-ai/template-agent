"""Tests for CodeExecutionMiddleware."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from deep_agent.src.code_execution.config import CodeExecutionConfig


class TestToolInjection:
    async def test_awrap_model_call_injects_tool(self):
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        config = CodeExecutionConfig(enabled=True)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        request.tools = [MagicMock()]

        override_request = MagicMock()
        request.override = MagicMock(return_value=override_request)
        handler = AsyncMock(return_value=MagicMock())

        await mw.awrap_model_call(request, handler)

        request.override.assert_called_once()
        handler.assert_called_once_with(override_request)

    async def test_awrap_model_call_disabled(self):
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        config = CodeExecutionConfig(enabled=False)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        handler = AsyncMock(return_value=MagicMock())

        await mw.awrap_model_call(request, handler)
        handler.assert_called_once_with(request)
        request.override.assert_not_called()


class TestToolCallRouting:
    async def test_passthrough_non_execute_code(self):
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        config = CodeExecutionConfig(enabled=True)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        request.tool_call = {"name": "search_web", "args": {}, "id": "tc1"}
        handler = AsyncMock(return_value=MagicMock())

        await mw.awrap_tool_call(request, handler)
        handler.assert_called_once_with(request)

    async def test_invalid_language(self):
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        config = CodeExecutionConfig(enabled=True)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        request.tool_call = {
            "name": "execute_code",
            "args": {"code": "x=1", "language": "ruby"},
            "id": "tc1",
        }
        handler = AsyncMock()

        result = await mw.awrap_tool_call(request, handler)
        handler.assert_not_called()
        assert "Unsupported language" in result.content
        assert "ruby" in result.content

    async def test_code_too_long(self):
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        config = CodeExecutionConfig(enabled=True, max_code_length=100)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        request.tool_call = {
            "name": "execute_code",
            "args": {"code": "x" * 101, "language": "python"},
            "id": "tc1",
        }
        handler = AsyncMock()

        result = await mw.awrap_tool_call(request, handler)
        handler.assert_not_called()
        assert "exceeds maximum length" in result.content

    @patch("deep_agent.src.code_execution.middleware.CodeExecutionMetrics")
    @patch("deep_agent.src.code_execution.middleware.K8sJobRunner")
    async def test_successful_execution(self, mock_runner_cls, mock_metrics_cls):
        from deep_agent.src.code_execution.k8s_job_runner import ExecutionResult
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(
            return_value=ExecutionResult(
                stdout="42",
                stderr="",
                exit_code=0,
                duration_seconds=1.5,
                status="success",
                job_name="code-exec-abc",
                namespace="ap-test-agent",
            )
        )
        mock_runner.resolve_namespace = MagicMock(return_value="ap-default-agent")
        mock_runner_cls.return_value = mock_runner

        mock_metrics = MagicMock()
        mock_metrics_cls.return_value = mock_metrics

        config = CodeExecutionConfig(enabled=True)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        request.tool_call = {
            "name": "execute_code",
            "args": {"code": "print(42)", "language": "python"},
            "id": "tc1",
        }
        handler = AsyncMock()

        result = await mw.awrap_tool_call(request, handler)
        handler.assert_not_called()
        assert "42" in result.content
        assert "exit_code: 0" in result.content
        assert result.tool_call_id == "tc1"


class TestSyncPassthrough:
    def test_wrap_model_call_passes_through(self):
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        config = CodeExecutionConfig(enabled=True)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        handler = MagicMock(return_value=MagicMock())

        mw.wrap_model_call(request, handler)
        handler.assert_called_once_with(request)

    def test_wrap_tool_call_passes_through(self):
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        config = CodeExecutionConfig(enabled=True)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        handler = MagicMock(return_value=MagicMock())

        mw.wrap_tool_call(request, handler)
        handler.assert_called_once_with(request)
