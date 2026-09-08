"""Unit tests for deep_agent.src.opa.service — OPA policy evaluation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deep_agent.src.opa.service import (
    OpaResult,
    _error_result,
    _parse_result,
    _query,
    _query_sync,
    _serialize_message,
    evaluate_message,
    evaluate_message_sync,
    evaluate_trajectory,
)

OPA_URL = "http://opa:8181/v1/data/agent/authz"
OPA_TIMEOUT = 5.0

_PATCH_URL = "deep_agent.src.opa.service.get_opa_url"
_PATCH_TIMEOUT = "deep_agent.src.opa.service.get_opa_timeout"
_PATCH_FAIL_OPEN = "deep_agent.src.opa.service.get_opa_fail_open"


# ── OpaResult dataclass ─────────────────────────────────────────────


class TestOpaResult:
    def test_allowed_with_empty_reasons(self):
        result = OpaResult(allowed=True)
        assert result.allowed is True
        assert result.denial_reasons == []

    def test_denied_with_reasons(self):
        reasons = ["policy-a violated", "policy-b violated"]
        result = OpaResult(allowed=False, denial_reasons=reasons)
        assert result.allowed is False
        assert result.denial_reasons == reasons

    def test_default_denial_reasons_is_empty_list(self):
        r1 = OpaResult(allowed=True)
        r2 = OpaResult(allowed=True)
        assert r1.denial_reasons is not r2.denial_reasons


# ── _parse_result ────────────────────────────────────────────────────


class TestParseResult:
    def test_empty_deny_reasons_means_allowed(self):
        data = {"result": {"deny_reasons": []}}
        result = _parse_result(data)
        assert result.allowed is True
        assert result.denial_reasons == []

    def test_deny_reasons_present_means_denied(self):
        data = {"result": {"deny_reasons": ["blocked by policy X"]}}
        result = _parse_result(data)
        assert result.allowed is False
        assert result.denial_reasons == ["blocked by policy X"]

    def test_multiple_deny_reasons(self):
        reasons = ["reason-a", "reason-b", "reason-c"]
        data = {"result": {"deny_reasons": reasons}}
        result = _parse_result(data)
        assert result.allowed is False
        assert result.denial_reasons == reasons

    def test_missing_result_key_denied(self):
        data = {"something_else": 42}
        result = _parse_result(data)
        assert result.allowed is False
        assert "missing expected decision keys" in result.denial_reasons[0]

    def test_result_not_a_dict_denied(self):
        data = {"result": "unexpected-string"}
        result = _parse_result(data)
        assert result.allowed is False
        assert "missing expected decision keys" in result.denial_reasons[0]

    def test_result_is_list_denied(self):
        data = {"result": [1, 2, 3]}
        result = _parse_result(data)
        assert result.allowed is False

    def test_result_is_none_denied(self):
        data = {"result": None}
        result = _parse_result(data)
        assert result.allowed is False

    def test_missing_deny_reasons_in_result_denied(self):
        data = {"result": {"allow": True}}
        result = _parse_result(data)
        assert result.allowed is False
        assert "missing expected decision keys" in result.denial_reasons[0]

    def test_deny_reasons_not_a_list_denied(self):
        data = {"result": {"deny_reasons": "not-a-list"}}
        result = _parse_result(data)
        assert result.allowed is False
        assert "missing or invalid type" in result.denial_reasons[0]

    def test_deny_reasons_is_none_denied(self):
        data = {"result": {"deny_reasons": None}}
        result = _parse_result(data)
        assert result.allowed is False
        assert "missing or invalid type" in result.denial_reasons[0]

    def test_deny_reasons_is_dict_denied(self):
        data = {"result": {"deny_reasons": {"a": 1}}}
        result = _parse_result(data)
        assert result.allowed is False

    def test_empty_data_denied(self):
        result = _parse_result({})
        assert result.allowed is False


# ── _serialize_message ───────────────────────────────────────────────


class TestSerializeMessage:
    def test_serializes_human_message(self):
        msg = HumanMessage(content="Hello, agent!")
        serialized = _serialize_message(msg)
        assert serialized == {"type": "human", "content": "Hello, agent!"}

    def test_serializes_ai_message(self):
        msg = AIMessage(content="I can help with that.")
        serialized = _serialize_message(msg)
        assert serialized == {"type": "ai", "content": "I can help with that."}

    def test_serializes_empty_content(self):
        msg = HumanMessage(content="")
        serialized = _serialize_message(msg)
        assert serialized == {"type": "human", "content": ""}


# ── _error_result ────────────────────────────────────────────────────


class TestErrorResult:
    def test_fail_open_true_allows(self):
        result = _error_result("OPA unreachable", fail_open=True)
        assert result.allowed is True
        assert "allowed by default" in result.denial_reasons[0]
        assert "OPA unreachable" in result.denial_reasons[0]

    def test_fail_open_false_denies(self):
        result = _error_result("OPA unreachable", fail_open=False)
        assert result.allowed is False
        assert "denied by default" in result.denial_reasons[0]
        assert "OPA unreachable" in result.denial_reasons[0]

    def test_reason_preserved_in_message(self):
        result = _error_result("connection refused", fail_open=True)
        assert "connection refused" in result.denial_reasons[0]


# ── _query (async) ───────────────────────────────────────────────────


def _mock_response(json_data, status_code=200):
    """Create a mock httpx.Response with the given JSON body."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


def _mock_http_error_response(status_code=403):
    """Create a mock response that raises HTTPStatusError on raise_for_status."""
    request = httpx.Request("POST", OPA_URL)
    response = httpx.Response(status_code, request=request)
    return response


class TestQuery:
    @pytest.mark.asyncio
    @patch(_PATCH_FAIL_OPEN, return_value=True)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    async def test_successful_call_allowed(
        self, mock_url, mock_timeout, mock_fail_open
    ):
        opa_response = {"result": {"deny_reasons": []}}
        mock_resp = _mock_response(opa_response)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "deep_agent.src.opa.service.httpx.AsyncClient", return_value=mock_client
        ):
            result = await _query({"current_intent": {"action": "llm_response"}})

        assert result.allowed is True
        assert result.denial_reasons == []
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    @patch(_PATCH_FAIL_OPEN, return_value=True)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    async def test_successful_call_denied(self, mock_url, mock_timeout, mock_fail_open):
        opa_response = {"result": {"deny_reasons": ["policy violation"]}}
        mock_resp = _mock_response(opa_response)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "deep_agent.src.opa.service.httpx.AsyncClient", return_value=mock_client
        ):
            result = await _query({"current_intent": {"action": "llm_response"}})

        assert result.allowed is False
        assert "policy violation" in result.denial_reasons

    @pytest.mark.asyncio
    @patch(_PATCH_FAIL_OPEN, return_value=True)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    async def test_timeout_fail_open(self, mock_url, mock_timeout, mock_fail_open):
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "deep_agent.src.opa.service.httpx.AsyncClient", return_value=mock_client
        ):
            result = await _query({"current_intent": {"action": "llm_response"}})

        assert result.allowed is True
        assert "timed out" in result.denial_reasons[0]

    @pytest.mark.asyncio
    @patch(_PATCH_FAIL_OPEN, return_value=False)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    async def test_timeout_fail_closed(self, mock_url, mock_timeout, mock_fail_open):
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "deep_agent.src.opa.service.httpx.AsyncClient", return_value=mock_client
        ):
            result = await _query({"current_intent": {"action": "llm_response"}})

        assert result.allowed is False
        assert "timed out" in result.denial_reasons[0]

    @pytest.mark.asyncio
    @patch(_PATCH_FAIL_OPEN, return_value=True)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    async def test_http_status_error_fail_open(
        self, mock_url, mock_timeout, mock_fail_open
    ):
        error_response = _mock_http_error_response(status_code=500)
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Server Error", request=error_response.request, response=error_response
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "deep_agent.src.opa.service.httpx.AsyncClient", return_value=mock_client
        ):
            result = await _query({"current_intent": {"action": "llm_response"}})

        assert result.allowed is True
        assert "HTTP 500" in result.denial_reasons[0]

    @pytest.mark.asyncio
    @patch(_PATCH_FAIL_OPEN, return_value=False)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    async def test_http_status_error_fail_closed(
        self, mock_url, mock_timeout, mock_fail_open
    ):
        error_response = _mock_http_error_response(status_code=403)
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Forbidden", request=error_response.request, response=error_response
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "deep_agent.src.opa.service.httpx.AsyncClient", return_value=mock_client
        ):
            result = await _query({"current_intent": {"action": "llm_response"}})

        assert result.allowed is False
        assert "HTTP 403" in result.denial_reasons[0]

    @pytest.mark.asyncio
    @patch(_PATCH_FAIL_OPEN, return_value=True)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    async def test_connection_error_fail_open(
        self, mock_url, mock_timeout, mock_fail_open
    ):
        mock_client = AsyncMock()
        mock_client.post.side_effect = ConnectionError("refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "deep_agent.src.opa.service.httpx.AsyncClient", return_value=mock_client
        ):
            result = await _query({"current_intent": {"action": "llm_response"}})

        assert result.allowed is True
        assert "unreachable" in result.denial_reasons[0].lower()

    @pytest.mark.asyncio
    @patch(_PATCH_FAIL_OPEN, return_value=False)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    async def test_connection_error_fail_closed(
        self, mock_url, mock_timeout, mock_fail_open
    ):
        mock_client = AsyncMock()
        mock_client.post.side_effect = ConnectionError("refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "deep_agent.src.opa.service.httpx.AsyncClient", return_value=mock_client
        ):
            result = await _query({"current_intent": {"action": "llm_response"}})

        assert result.allowed is False

    @pytest.mark.asyncio
    @patch(_PATCH_FAIL_OPEN, return_value=True)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    async def test_posts_correct_payload(self, mock_url, mock_timeout, mock_fail_open):
        opa_response = {"result": {"deny_reasons": []}}
        mock_resp = _mock_response(opa_response)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        opa_input = {
            "current_intent": {"action": "llm_response", "agent_message": "hi"}
        }

        with patch(
            "deep_agent.src.opa.service.httpx.AsyncClient", return_value=mock_client
        ):
            await _query(opa_input)

        mock_client.post.assert_called_once_with(OPA_URL, json={"input": opa_input})


# ── _query_sync ──────────────────────────────────────────────────────


def _mock_sync_client(response=None, side_effect=None):
    """Create a mock httpx.Client context manager."""
    mock_client = MagicMock()
    if side_effect:
        mock_client.post.side_effect = side_effect
    else:
        mock_client.post.return_value = response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    return mock_client


class TestQuerySync:
    @patch(_PATCH_FAIL_OPEN, return_value=True)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    def test_successful_call_allowed(self, mock_url, mock_timeout, mock_fail_open):
        opa_response = {"result": {"deny_reasons": []}}
        mock_client = _mock_sync_client(response=_mock_response(opa_response))

        with patch("deep_agent.src.opa.service.httpx.Client", return_value=mock_client):
            result = _query_sync({"current_intent": {"action": "llm_response"}})

        assert result.allowed is True
        assert result.denial_reasons == []

    @patch(_PATCH_FAIL_OPEN, return_value=True)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    def test_successful_call_denied(self, mock_url, mock_timeout, mock_fail_open):
        opa_response = {"result": {"deny_reasons": ["not allowed"]}}
        mock_client = _mock_sync_client(response=_mock_response(opa_response))

        with patch("deep_agent.src.opa.service.httpx.Client", return_value=mock_client):
            result = _query_sync({"current_intent": {"action": "llm_response"}})

        assert result.allowed is False
        assert "not allowed" in result.denial_reasons

    @patch(_PATCH_FAIL_OPEN, return_value=True)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    def test_timeout_fail_open(self, mock_url, mock_timeout, mock_fail_open):
        mock_client = _mock_sync_client(side_effect=httpx.TimeoutException("timed out"))

        with patch("deep_agent.src.opa.service.httpx.Client", return_value=mock_client):
            result = _query_sync({"current_intent": {"action": "llm_response"}})

        assert result.allowed is True
        assert "timed out" in result.denial_reasons[0]

    @patch(_PATCH_FAIL_OPEN, return_value=False)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    def test_timeout_fail_closed(self, mock_url, mock_timeout, mock_fail_open):
        mock_client = _mock_sync_client(side_effect=httpx.TimeoutException("timed out"))

        with patch("deep_agent.src.opa.service.httpx.Client", return_value=mock_client):
            result = _query_sync({"current_intent": {"action": "llm_response"}})

        assert result.allowed is False

    @patch(_PATCH_FAIL_OPEN, return_value=True)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    def test_http_status_error(self, mock_url, mock_timeout, mock_fail_open):
        error_response = _mock_http_error_response(status_code=502)
        mock_client = _mock_sync_client(
            side_effect=httpx.HTTPStatusError(
                "Bad Gateway", request=error_response.request, response=error_response
            )
        )

        with patch("deep_agent.src.opa.service.httpx.Client", return_value=mock_client):
            result = _query_sync({"current_intent": {"action": "llm_response"}})

        assert result.allowed is True
        assert "HTTP 502" in result.denial_reasons[0]

    @patch(_PATCH_FAIL_OPEN, return_value=False)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    def test_connection_error_fail_closed(self, mock_url, mock_timeout, mock_fail_open):
        mock_client = _mock_sync_client(side_effect=ConnectionError("refused"))

        with patch("deep_agent.src.opa.service.httpx.Client", return_value=mock_client):
            result = _query_sync({"current_intent": {"action": "llm_response"}})

        assert result.allowed is False
        assert "unreachable" in result.denial_reasons[0].lower()

    @patch(_PATCH_FAIL_OPEN, return_value=True)
    @patch(_PATCH_TIMEOUT, return_value=OPA_TIMEOUT)
    @patch(_PATCH_URL, return_value=OPA_URL)
    def test_posts_correct_payload(self, mock_url, mock_timeout, mock_fail_open):
        opa_response = {"result": {"deny_reasons": []}}
        mock_client = _mock_sync_client(response=_mock_response(opa_response))
        opa_input = {"current_intent": {"action": "tool_response", "result": "ok"}}

        with patch("deep_agent.src.opa.service.httpx.Client", return_value=mock_client):
            _query_sync(opa_input)

        mock_client.post.assert_called_once_with(OPA_URL, json={"input": opa_input})


# ── evaluate_message (async) ────────────────────────────────────────


class TestEvaluateMessage:
    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.service._query")
    async def test_llm_response_with_agent_message(self, mock_query):
        mock_query.return_value = OpaResult(allowed=True)

        result = await evaluate_message("llm_response", agent_message="Hello world")

        assert result.allowed is True
        mock_query.assert_called_once()
        call_payload = mock_query.call_args[0][0]
        assert call_payload["current_intent"]["action"] == "llm_response"
        assert call_payload["current_intent"]["agent_message"] == "Hello world"

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.service._query")
    async def test_tool_response_with_result(self, mock_query):
        mock_query.return_value = OpaResult(allowed=True)

        result = await evaluate_message("tool_response", result="tool output")

        assert result.allowed is True
        call_payload = mock_query.call_args[0][0]
        assert call_payload["current_intent"]["action"] == "tool_response"
        assert call_payload["current_intent"]["result"] == "tool output"

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.service._query")
    async def test_opa_denies(self, mock_query):
        mock_query.return_value = OpaResult(allowed=False, denial_reasons=["blocked"])

        result = await evaluate_message("llm_response", agent_message="bad content")

        assert result.allowed is False
        assert result.denial_reasons == ["blocked"]

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.service._query")
    async def test_llm_response_without_agent_message(self, mock_query):
        mock_query.return_value = OpaResult(allowed=True)

        await evaluate_message("llm_response")

        call_payload = mock_query.call_args[0][0]
        assert "agent_message" not in call_payload["current_intent"]

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.service._query")
    async def test_tool_response_without_result(self, mock_query):
        mock_query.return_value = OpaResult(allowed=True)

        await evaluate_message("tool_response")

        call_payload = mock_query.call_args[0][0]
        assert "result" not in call_payload["current_intent"]


# ── evaluate_trajectory (async) ──────────────────────────────────────


class TestEvaluateTrajectory:
    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.service._query")
    async def test_with_messages(self, mock_query):
        mock_query.return_value = OpaResult(allowed=True)

        trajectory = [
            HumanMessage(content="What is the weather?"),
            AIMessage(content="It is sunny today."),
        ]

        result = await evaluate_trajectory(trajectory)

        assert result.allowed is True
        call_payload = mock_query.call_args[0][0]
        assert call_payload["current_intent"]["action"] == "trajectory_validation"
        assert len(call_payload["trajectory"]) == 2
        assert call_payload["trajectory"][0] == {
            "type": "human",
            "content": "What is the weather?",
        }
        assert call_payload["trajectory"][1] == {
            "type": "ai",
            "content": "It is sunny today.",
        }

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.service._query")
    async def test_empty_trajectory(self, mock_query):
        mock_query.return_value = OpaResult(allowed=True)

        result = await evaluate_trajectory([])

        assert result.allowed is True
        call_payload = mock_query.call_args[0][0]
        assert call_payload["trajectory"] == []

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.service._query")
    async def test_trajectory_denied(self, mock_query):
        mock_query.return_value = OpaResult(
            allowed=False, denial_reasons=["unsafe trajectory"]
        )

        result = await evaluate_trajectory([HumanMessage(content="hack the system")])

        assert result.allowed is False
        assert result.denial_reasons == ["unsafe trajectory"]


# ── evaluate_message_sync ────────────────────────────────────────────


class TestEvaluateMessageSync:
    @patch("deep_agent.src.opa.service._query_sync")
    def test_llm_response(self, mock_query_sync):
        mock_query_sync.return_value = OpaResult(allowed=True)

        result = evaluate_message_sync("llm_response", agent_message="agent says hi")

        assert result.allowed is True
        call_payload = mock_query_sync.call_args[0][0]
        assert call_payload["current_intent"]["action"] == "llm_response"
        assert call_payload["current_intent"]["agent_message"] == "agent says hi"

    @patch("deep_agent.src.opa.service._query_sync")
    def test_tool_response(self, mock_query_sync):
        mock_query_sync.return_value = OpaResult(allowed=True)

        result = evaluate_message_sync("tool_response", result="tool output data")

        assert result.allowed is True
        call_payload = mock_query_sync.call_args[0][0]
        assert call_payload["current_intent"]["action"] == "tool_response"
        assert call_payload["current_intent"]["result"] == "tool output data"

    @patch("deep_agent.src.opa.service._query_sync")
    def test_opa_denies_sync(self, mock_query_sync):
        mock_query_sync.return_value = OpaResult(
            allowed=False, denial_reasons=["sync denial"]
        )

        result = evaluate_message_sync("llm_response", agent_message="bad content")

        assert result.allowed is False
        assert result.denial_reasons == ["sync denial"]

    @patch("deep_agent.src.opa.service._query_sync")
    def test_llm_response_without_agent_message(self, mock_query_sync):
        mock_query_sync.return_value = OpaResult(allowed=True)

        evaluate_message_sync("llm_response")

        call_payload = mock_query_sync.call_args[0][0]
        assert "agent_message" not in call_payload["current_intent"]

    @patch("deep_agent.src.opa.service._query_sync")
    def test_tool_response_without_result(self, mock_query_sync):
        mock_query_sync.return_value = OpaResult(allowed=True)

        evaluate_message_sync("tool_response")

        call_payload = mock_query_sync.call_args[0][0]
        assert "result" not in call_payload["current_intent"]
