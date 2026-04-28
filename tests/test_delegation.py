"""Tests for A2A delegation tool (Phase 3)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from template_agent.src.a2a.context import A2ARequestContext, a2a_request_ctx
from template_agent.src.a2a.delegation import (
    DelegateInput,
    build_a2a_delegation_tool,
    delegate_to_a2a_agent,
)
from template_agent.src.a2a.models import A2ATargetAgent
from template_agent.src.a2a.registry import A2AAgentRegistry


@pytest.fixture
def populated_registry(monkeypatch):
    """Inject a pre-populated registry singleton."""
    import template_agent.src.a2a.registry as reg_mod

    registry = A2AAgentRegistry.__new__(A2AAgentRegistry)
    registry._agents = {
        "echo": A2ATargetAgent(
            agent_id="echo",
            base_url="http://echo:8082",
            description="Echo agent",
            skills=["echo-skill"],
        )
    }
    registry._client = AsyncMock(spec=httpx.AsyncClient)
    monkeypatch.setattr(reg_mod, "_registry", registry)
    return registry


def _a2a_response(artifacts=None, state="completed", message=None):
    """Build an A2A JSON-RPC response body."""
    result = {"status": {"state": state}}
    if message:
        result["status"]["message"] = message
    result["artifacts"] = artifacts or []
    return {"jsonrpc": "2.0", "id": "1", "result": result}


def _mock_httpx_response(status_code=200, json_data=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


class TestBuildDelegationTool:
    def test_returns_none_when_no_agents(self, monkeypatch):
        import template_agent.src.a2a.registry as reg_mod

        empty = A2AAgentRegistry.__new__(A2AAgentRegistry)
        empty._agents = {}
        empty._client = AsyncMock(spec=httpx.AsyncClient)
        monkeypatch.setattr(reg_mod, "_registry", empty)
        tool = build_a2a_delegation_tool()
        assert tool is None

    def test_returns_tool_when_agents_present(self, populated_registry):
        tool = build_a2a_delegation_tool()
        assert tool is not None
        assert tool.name == "delegate_to_a2a_agent"
        assert "echo" in tool.description


class TestDelegateToA2AAgent:
    async def test_unknown_agent(self, populated_registry):
        result = await delegate_to_a2a_agent("unknown", "hello")
        assert "not registered" in result

    async def test_successful_delegation(self, populated_registry):
        response_body = _a2a_response(
            artifacts=[{"parts": [{"kind": "text", "text": "echoed: hello"}]}]
        )
        mock_resp = _mock_httpx_response(200, response_body)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "template_agent.src.a2a.delegation.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await delegate_to_a2a_agent("echo", "hello", access_token="tok123")
        assert "echoed: hello" in result

    async def test_downstream_failure(self, populated_registry):
        response_body = _a2a_response(state="failed", message="boom")
        mock_resp = _mock_httpx_response(200, response_body)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "template_agent.src.a2a.delegation.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await delegate_to_a2a_agent("echo", "hello")
        assert "failed" in result.lower()

    async def test_http_error(self, populated_registry):
        mock_resp = _mock_httpx_response(500)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "template_agent.src.a2a.delegation.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await delegate_to_a2a_agent("echo", "hello")
        assert "HTTP 500" in result

    async def test_headers_forwarded(self, populated_registry):
        """Verify auth and identity headers are forwarded to downstream."""
        response_body = _a2a_response()
        mock_resp = _mock_httpx_response(200, response_body)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        ctx = A2ARequestContext(
            access_token="parent-token",
            calling_agent_id="parent-agent",
            correlation_id="corr-123",
        )
        token = a2a_request_ctx.set(ctx)
        try:
            with patch(
                "template_agent.src.a2a.delegation.httpx.AsyncClient",
                return_value=mock_client,
            ):
                await delegate_to_a2a_agent("echo", "hello")
        finally:
            a2a_request_ctx.reset(token)

        call_args = mock_client.post.call_args
        headers = call_args.kwargs.get("headers", {})
        assert headers["Authorization"] == "Bearer parent-token"
        assert headers["X-Calling-Agent-ID"] == "template-agent"
        assert headers["X-Correlation-ID"] == "corr-123"
        assert "traceparent" not in headers

    async def test_message_result_kind(self, populated_registry):
        """Result with 'kind': 'message' extracts text from parts."""
        response_body = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "kind": "message",
                "parts": [{"kind": "text", "text": "message response"}],
                "status": {"state": "completed"},
            },
        }
        mock_resp = _mock_httpx_response(200, response_body)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "template_agent.src.a2a.delegation.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await delegate_to_a2a_agent("echo", "hello")
        assert "message response" in result

    async def test_no_text_artifacts_returned(self, populated_registry):
        """Returns appropriate message when no text artifacts in response."""
        response_body = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "artifacts": [],
                "status": {"state": "completed"},
            },
        }
        mock_resp = _mock_httpx_response(200, response_body)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "template_agent.src.a2a.delegation.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await delegate_to_a2a_agent("echo", "hello")
        assert "no text artifacts" in result.lower()

    async def test_connection_error(self, populated_registry):
        """Connection error returns appropriate error message."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "template_agent.src.a2a.delegation.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await delegate_to_a2a_agent("echo", "hello")
        assert "failed to reach" in result.lower()

    async def test_thread_id_in_metadata(self, populated_registry):
        """Thread ID is included in message metadata when provided."""
        response_body = _a2a_response(
            artifacts=[{"parts": [{"kind": "text", "text": "ok"}]}]
        )
        mock_resp = _mock_httpx_response(200, response_body)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "template_agent.src.a2a.delegation.httpx.AsyncClient",
            return_value=mock_client,
        ):
            await delegate_to_a2a_agent(
                "echo", "hello", thread_id="thread-123", user_id="user-456"
            )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json", {})
        metadata = payload.get("params", {}).get("message", {}).get("metadata", {})
        assert metadata.get("thread_id") == "thread-123"
        assert metadata.get("user_id") == "user-456"

    async def test_no_thread_id_omitted_from_metadata(self, populated_registry):
        """Thread ID is omitted from metadata when not provided."""
        response_body = _a2a_response(
            artifacts=[{"parts": [{"kind": "text", "text": "ok"}]}]
        )
        mock_resp = _mock_httpx_response(200, response_body)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "template_agent.src.a2a.delegation.httpx.AsyncClient",
            return_value=mock_client,
        ):
            await delegate_to_a2a_agent("echo", "hello")

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json", {})
        metadata = payload.get("params", {}).get("message", {}).get("metadata", {})
        assert "thread_id" not in metadata

    async def test_no_correlation_id_in_context(self, populated_registry):
        """When no correlation ID in context, header is not sent."""
        response_body = _a2a_response(
            artifacts=[{"parts": [{"kind": "text", "text": "ok"}]}]
        )
        mock_resp = _mock_httpx_response(200, response_body)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        ctx = A2ARequestContext(access_token="token")
        token = a2a_request_ctx.set(ctx)
        try:
            with patch(
                "template_agent.src.a2a.delegation.httpx.AsyncClient",
                return_value=mock_client,
            ):
                await delegate_to_a2a_agent("echo", "hello")
        finally:
            a2a_request_ctx.reset(token)

        call_args = mock_client.post.call_args
        headers = call_args.kwargs.get("headers", {})
        assert "X-Correlation-ID" not in headers


class TestDelegateInput:
    """Tests for DelegateInput schema."""

    def test_valid_input(self):
        """Valid input creates DelegateInput instance."""
        inp = DelegateInput(agent_id="echo", message="hello")
        assert inp.agent_id == "echo"
        assert inp.message == "hello"

    def test_input_validation(self):
        """DelegateInput validates required fields."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DelegateInput(agent_id="echo")
