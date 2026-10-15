"""MCP HTTP surface: the thin bits around the OAuth handlers.

The one non-obvious behaviour here is `_http_exception_response`: a
dict `detail` has to bypass aegra's error handler, which assumes a
string and turns a 401 into a 500 on the way out.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse

import deep_agent.aegra.mcp_routes as mr


def _request(headers: dict | None = None) -> MagicMock:
    request = MagicMock()
    request.headers = headers or {}
    return request


class TestSsoBearerFromRequest:
    def test_extracts_the_token(self):
        assert (
            mr._sso_bearer_from_request(_request({"authorization": "Bearer abc123"}))
            == "abc123"
        )

    def test_none_without_a_header(self):
        assert mr._sso_bearer_from_request(_request()) is None

    def test_none_for_another_scheme(self):
        assert (
            mr._sso_bearer_from_request(_request({"authorization": "Basic x"})) is None
        )

    def test_none_for_an_empty_bearer(self):
        assert (
            mr._sso_bearer_from_request(_request({"authorization": "Bearer   "}))
            is None
        )


class TestHttpExceptionResponse:
    def test_dict_detail_becomes_a_json_response(self):
        # aegra's AgentProtocolError.message is a string, so a dict
        # detail must bypass that handler or the client sees 500
        # instead of 401.
        exc = HTTPException(status_code=401, detail={"error": "authorization_required"})
        response = mr._http_exception_response(exc)
        assert isinstance(response, JSONResponse)
        assert response.status_code == 401
        assert json.loads(response.body)["error"] == "authorization_required"

    def test_string_detail_is_left_to_the_normal_handler(self):
        assert mr._http_exception_response(HTTPException(404, "not found")) is None


class TestMcpConnect:
    @pytest.mark.asyncio
    async def test_403_when_dcr_is_disabled_for_a_dcr_server(self):
        config = MagicMock()
        config.get_mcp_servers.return_value = {"jira": {"auth_mode": "dcr"}}
        # Patch the one flag, not the whole settings object: other
        # modules read PYTHON_LOG_LEVEL off it at import time.
        with (
            patch("deep_agent.src.agent.config.agent_config", config),
            patch("deep_agent.src.settings.settings.MCP_DCR_ENABLED", False),
        ):
            response = await mr.mcp_connect("jira", _request())

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_delegates_to_the_handler_when_allowed(self):
        config = MagicMock()
        config.get_mcp_servers.return_value = {"jira": {"auth_mode": "oauth"}}
        # The route wraps whatever the handler returns, so it has
        # to be JSON-serialisable.
        handler = AsyncMock(return_value={"authorization_url": "https://x"})
        with (
            patch("deep_agent.src.agent.config.agent_config", config),
            patch("deep_agent.src.settings.settings.MCP_DCR_ENABLED", True),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.handle_mcp_connect", new=handler
            ),
            patch.object(
                mr, "_authenticated_user_id", new=AsyncMock(return_value="u1")
            ),
        ):
            await mr.mcp_connect("jira", _request())

        handler.assert_awaited_once()


class TestMcpConnections:
    @pytest.mark.asyncio
    async def test_returns_the_handler_payload(self):
        payload = {"connections": [{"mcp_name": "jira", "connected": True}]}
        with (
            patch.object(
                mr, "_authenticated_user_id", new=AsyncMock(return_value="u1")
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.handle_mcp_connections",
                new=AsyncMock(return_value=payload),
            ),
        ):
            response = await mr.mcp_connections(_request())

        assert json.loads(response.body)["connections"][0]["mcp_name"] == "jira"


class TestOAuthClientMetadata:
    @pytest.mark.asyncio
    async def test_describes_this_agent_as_an_oauth_client(self):
        # The document a CIMD-capable server fetches when it is handed
        # a URL as the client_id.
        response = await mr.oauth_client_metadata()
        body = json.loads(response.body)
        assert body["client_id"].endswith("/.well-known/oauth-client")
        assert "redirect_uris" in body
