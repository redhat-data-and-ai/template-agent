"""Unit tests for MCP OAuth handler edge cases."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.testclient import TestClient

from deep_agent.aegra.http_app import app
from deep_agent.aegra.mcp_oauth_handlers import (
    _callback_html,
    _register_dcr_client,
    handle_mcp_connect,
    handle_mcp_oauth_callback,
)


@pytest.mark.asyncio
class TestHandleMcpConnect:
    async def test_client_credentials_rejects_connect(self):
        server_cfg = {
            "enabled": True,
            "auth_mode": "oauth",
            "oauth": {
                "grant_type": "client_credentials",
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
            },
        }
        with patch(
            "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
            return_value=server_cfg,
        ):
            with pytest.raises(HTTPException) as exc:
                await handle_mcp_connect("user-1", "cc-mcp")
            assert exc.value.status_code == 400
            assert "client_credentials" in exc.value.detail
            assert "not required" in exc.value.detail

    async def test_authorization_code_proceeds_past_grant_type_check(self):
        server_cfg = {
            "enabled": True,
            "auth_mode": "oauth",
            "oauth": {
                "grant_type": "authorization_code",
                "authorization_endpoint": "https://auth.example.com/authorize",
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
            },
        }
        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value=server_cfg,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.cache_set",
                return_value=True,
            ),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_settings.agent_deployment_id = "test-agent"
            result = await handle_mcp_connect("user-1", "oauth-mcp")
            assert "authorize_url" in result
            assert "auth.example.com/authorize" in result["authorize_url"]

    async def test_non_oauth_auth_mode_rejects(self):
        server_cfg = {
            "enabled": True,
            "auth_mode": "sso",
        }
        with patch(
            "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
            return_value=server_cfg,
        ):
            with pytest.raises(HTTPException) as exc:
                await handle_mcp_connect("user-1", "sso-mcp")
            assert exc.value.status_code == 400
            assert "does not use OAuth" in exc.value.detail


@pytest.mark.asyncio
class TestRegisterDcrClient:
    async def test_uses_requested_scopes(self):
        oauth_cfg = {
            "registration_endpoint": "https://auth.example.com/register",
            "scopes": ["read", "write"],
        }
        server_cfg = {"enabled": True}

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "client_id": "dcr-cid",
            "client_secret": "dcr-secret",
        }

        mock_ctx = AsyncMock(post=AsyncMock(return_value=mock_response))
        mock_store = MagicMock()
        mock_store.upsert_client = AsyncMock()

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient",
            ) as mock_client_cls,
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.mcp_httpx_verify",
                return_value=True,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.McpTokenStore",
                return_value=mock_store,
            ),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            cid, secret = await _register_dcr_client(
                "test-agent", "dcr-mcp", oauth_cfg, server_cfg
            )

        assert cid == "dcr-cid"
        assert secret == "dcr-secret"
        post_kwargs = mock_ctx.post.call_args
        body = post_kwargs.kwargs.get("json") or post_kwargs[1].get("json", {})
        assert body["scope"] == "read write"


def _mock_request() -> Request:
    """Create a minimal mock Starlette Request for OAuth callback tests."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/mcp/oauth/callback",
        "query_string": b"",
        "headers": [],
    }
    return Request(scope)


@pytest.mark.asyncio
class TestHandleMcpOauthCallback:
    async def test_missing_code_returns_error(self):
        response = await handle_mcp_oauth_callback(
            code=None, state="some-state", request=_mock_request()
        )
        assert response.status_code == 400
        assert b"Missing" in response.body

    async def test_missing_state_returns_error(self):
        response = await handle_mcp_oauth_callback(
            code="some-code", state=None, request=_mock_request()
        )
        assert response.status_code == 400
        assert b"Missing" in response.body

    async def test_successful_callback_returns_connected_html(self):
        state_payload = json.dumps(
            {
                "user_id": "user-1",
                "mcp_name": "oauth-mcp",
                "code_verifier": "test-verifier",
            }
        )

        server_cfg = {
            "enabled": True,
            "auth_mode": "oauth",
            "oauth": {
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
            },
        }

        token_response = MagicMock()
        token_response.json.return_value = {
            "access_token": "new-access-token",
            "expires_in": 3600,
        }
        token_response.raise_for_status = MagicMock()

        mock_ctx = AsyncMock(post=AsyncMock(return_value=token_response))
        mock_store = MagicMock()
        mock_store.upsert_token = AsyncMock()

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.cache_get",
                return_value=state_payload,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value=server_cfg,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient",
            ) as mock_client_cls,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.mcp_httpx_verify",
                return_value=True,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.McpTokenStore",
                return_value=mock_store,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.resolve_oauth_client_secret",
                return_value="csecret",
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.get_mcp_credential_resolver",
            ) as mock_resolver,
            patch("deep_agent.aegra.mcp.invalidate_mcp_tool_cache"),
            patch("deep_agent.aegra.graph.invalidate_graph_cache"),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_settings.agent_deployment_id = "test-agent"
            mock_settings.database_uri = "sqlite:///test.db"
            mock_settings.ui_origin = "https://ui.example.com"

            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resolver.return_value.invalidate_cache = MagicMock()

            response = await handle_mcp_oauth_callback(
                code="auth-code", state="valid-state", request=_mock_request()
            )

        assert response.status_code == 200
        assert b"Connected" in response.body
        assert b"mcp_oauth_done" in response.body


class TestMcpOauthCallbackRoute:
    def test_callback_route_returns_error_without_params(self):
        client = TestClient(app)
        resp = client.get("/mcp/oauth/callback")
        assert resp.status_code == 400
        assert "Missing" in resp.text


class TestCallbackHtml:
    def test_includes_opener_origin_in_postmessage(self):
        html = _callback_html(
            mcp_name="test-mcp", opener_origin="https://ui.example.com"
        )
        assert '"https://ui.example.com"' in html
        assert "mcp_oauth_done" in html

    def test_skips_postmessage_when_no_origin(self):
        html = _callback_html(mcp_name="test-mcp", opener_origin=None)
        assert "postMessage" not in html
        assert "Connected" in html

    def test_error_html_returned(self):
        html = _callback_html(error="something broke")
        assert "something broke" in html
        assert "MCP OAuth Error" in html
