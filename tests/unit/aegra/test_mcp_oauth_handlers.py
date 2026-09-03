"""Unit tests for MCP OAuth handler edge cases."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.testclient import TestClient

from deep_agent.aegra.http_app import app
from deep_agent.aegra.mcp_oauth_handlers import (
    DcrClientManager,
    _callback_html,
    _is_invalid_client_error,
    _register_dcr_client,
    get_dcr_client_manager,
    handle_mcp_connect,
    handle_mcp_connections,
    handle_mcp_disconnect,
    handle_mcp_oauth_callback,
)
from deep_agent.aegra.mcp_token_store import McpOAuthClient, McpTokenStore


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

    async def test_ok_false_returns_error_with_message(self):
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
        token_response.json.return_value = {"ok": False, "error": "invalid_code"}
        token_response.raise_for_status = MagicMock()

        mock_ctx = AsyncMock(post=AsyncMock(return_value=token_response))

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
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.resolve_oauth_client_secret",
                return_value="csecret",
            ),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_settings.agent_deployment_id = "test-agent"
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response = await handle_mcp_oauth_callback(
                code="auth-code", state="valid-state", request=_mock_request()
            )

        assert response.status_code == 502
        assert b"invalid_code" in response.body

    async def test_authed_user_access_token_fallback(self):
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
            "ok": True,
            "authed_user": {
                "access_token": "xoxp-nested-token",
                "scope": "chat:write",
            },
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
        mock_store.upsert_token.assert_awaited_once()
        call_kwargs = mock_store.upsert_token.call_args[1]
        assert call_kwargs["access_token"] == "xoxp-nested-token"

    async def test_missing_access_token_everywhere_returns_error(self):
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
        token_response.json.return_value = {"token_type": "bearer", "expires_in": 3600}
        token_response.raise_for_status = MagicMock()

        mock_ctx = AsyncMock(post=AsyncMock(return_value=token_response))

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
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.resolve_oauth_client_secret",
                return_value="csecret",
            ),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_settings.agent_deployment_id = "test-agent"
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response = await handle_mcp_oauth_callback(
                code="auth-code", state="valid-state", request=_mock_request()
            )

        assert response.status_code == 502
        assert b"missing access_token" in response.body


class TestMcpOauthCallbackRoute:
    def test_callback_route_returns_error_without_params(self):
        client = TestClient(app)
        resp = client.get("/mcp/oauth/callback")
        assert resp.status_code == 400
        assert "Missing" in resp.text


_INTERACTIVE_SERVERS = {
    "alpha-oauth": {
        "enabled": True,
        "auth_mode": "oauth",
        "description": "Alpha tools",
        "oauth": {"grant_type": "authorization_code"},
    },
    "bravo-dcr": {
        "enabled": True,
        "auth_mode": "dcr",
        "description": "Bravo DCR",
        "oauth": {"grant_type": "authorization_code"},
    },
    "cc-mcp": {
        "enabled": True,
        "auth_mode": "oauth",
        "oauth": {"grant_type": "client_credentials"},
    },
    "off-oauth": {
        "enabled": False,
        "auth_mode": "oauth",
        "description": "Disabled",
    },
    "sso-mcp": {"enabled": True, "auth_mode": "sso"},
}


@pytest.mark.asyncio
class TestHandleMcpConnections:
    async def test_lists_interactive_oauth_and_dcr_with_status(self):
        resolver = MagicMock()
        resolver.has_valid_token = AsyncMock(side_effect=[True, False])

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.agent_config.get_mcp_servers",
                return_value=_INTERACTIVE_SERVERS,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.get_mcp_credential_resolver",
                return_value=resolver,
            ),
        ):
            mock_settings.MCP_DCR_ENABLED = True
            result = await handle_mcp_connections("user-1")

        assert result == {
            "connections": [
                {
                    "mcp_name": "alpha-oauth",
                    "auth_mode": "oauth",
                    "description": "Alpha tools",
                    "connected": True,
                },
                {
                    "mcp_name": "bravo-dcr",
                    "auth_mode": "dcr",
                    "description": "Bravo DCR",
                    "connected": False,
                },
            ]
        }

    async def test_omits_dcr_when_feature_disabled(self):
        resolver = MagicMock()
        resolver.has_valid_token = AsyncMock(return_value=False)

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.agent_config.get_mcp_servers",
                return_value=_INTERACTIVE_SERVERS,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.get_mcp_credential_resolver",
                return_value=resolver,
            ),
        ):
            mock_settings.MCP_DCR_ENABLED = False
            result = await handle_mcp_connections("user-1")

        names = [row["mcp_name"] for row in result["connections"]]
        assert names == ["alpha-oauth"]

    async def test_defaults_missing_description_to_empty_string(self):
        resolver = MagicMock()
        resolver.has_valid_token = AsyncMock(return_value=True)
        servers = {
            "plain": {
                "enabled": True,
                "auth_mode": "oauth",
                "oauth": {"grant_type": "authorization_code"},
            }
        }

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.agent_config.get_mcp_servers",
                return_value=servers,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.get_mcp_credential_resolver",
                return_value=resolver,
            ),
        ):
            mock_settings.MCP_DCR_ENABLED = True
            result = await handle_mcp_connections("user-1")

        assert result["connections"][0]["description"] == ""


@pytest.mark.asyncio
class TestHandleMcpDisconnect:
    async def test_clears_token_and_returns_disconnected(self):
        store = MagicMock()
        store.delete_token = AsyncMock(return_value=True)
        resolver = MagicMock()
        resolver.invalidate_cache = MagicMock()
        server_cfg = {
            "enabled": True,
            "auth_mode": "oauth",
            "oauth": {"grant_type": "authorization_code"},
        }

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value=server_cfg,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.McpTokenStore",
                return_value=store,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.get_mcp_credential_resolver",
                return_value=resolver,
            ),
            patch("deep_agent.aegra.mcp.invalidate_mcp_tool_cache") as mock_tools,
            patch("deep_agent.aegra.graph.invalidate_graph_cache") as mock_graph,
        ):
            mock_settings.database_uri = "postgresql://test"
            mock_settings.agent_deployment_id = "test-agent"
            result = await handle_mcp_disconnect("user-1", "oauth-mcp")

        assert result == {"mcp_name": "oauth-mcp", "connected": False}
        store.delete_token.assert_awaited_once_with("test-agent", "user-1", "oauth-mcp")
        resolver.invalidate_cache.assert_called_once_with("user-1", "oauth-mcp")
        mock_tools.assert_called_once_with(user_id="user-1")
        mock_graph.assert_called_once_with()

    async def test_disconnect_succeeds_when_graph_cache_invalidation_fails(self):
        store = MagicMock()
        store.delete_token = AsyncMock(return_value=True)
        resolver = MagicMock()
        resolver.invalidate_cache = MagicMock()
        server_cfg = {
            "enabled": True,
            "auth_mode": "oauth",
            "oauth": {"grant_type": "authorization_code"},
        }

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value=server_cfg,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.McpTokenStore",
                return_value=store,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.get_mcp_credential_resolver",
                return_value=resolver,
            ),
            patch("deep_agent.aegra.mcp.invalidate_mcp_tool_cache"),
            patch(
                "deep_agent.aegra.graph.invalidate_graph_cache",
                side_effect=RuntimeError("cache down"),
            ),
        ):
            mock_settings.database_uri = "postgresql://test"
            mock_settings.agent_deployment_id = "test-agent"
            result = await handle_mcp_disconnect("user-1", "oauth-mcp")

        assert result == {"mcp_name": "oauth-mcp", "connected": False}
        store.delete_token.assert_awaited_once_with("test-agent", "user-1", "oauth-mcp")
        resolver.invalidate_cache.assert_called_once_with("user-1", "oauth-mcp")

    async def test_rejects_dcr_when_feature_disabled(self):
        store = MagicMock()
        store.delete_token = AsyncMock()
        resolver = MagicMock()
        resolver.invalidate_cache = MagicMock()

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value={
                    "enabled": True,
                    "auth_mode": "dcr",
                    "oauth": {"grant_type": "authorization_code"},
                },
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.McpTokenStore",
                return_value=store,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.get_mcp_credential_resolver",
                return_value=resolver,
            ),
        ):
            mock_settings.MCP_DCR_ENABLED = False
            with pytest.raises(HTTPException) as exc:
                await handle_mcp_disconnect("user-1", "dcr-mcp")

        assert exc.value.status_code == 403
        assert "DCR is disabled" in str(exc.value.detail)
        store.delete_token.assert_not_awaited()
        resolver.invalidate_cache.assert_not_called()

    async def test_disconnects_dcr_when_feature_enabled(self):
        store = MagicMock()
        store.delete_token = AsyncMock(return_value=True)
        resolver = MagicMock()
        resolver.invalidate_cache = MagicMock()
        server_cfg = {
            "enabled": True,
            "auth_mode": "dcr",
            "oauth": {"grant_type": "authorization_code"},
        }

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value=server_cfg,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.McpTokenStore",
                return_value=store,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.get_mcp_credential_resolver",
                return_value=resolver,
            ),
            patch("deep_agent.aegra.mcp.invalidate_mcp_tool_cache"),
            patch("deep_agent.aegra.graph.invalidate_graph_cache"),
        ):
            mock_settings.MCP_DCR_ENABLED = True
            mock_settings.database_uri = "postgresql://test"
            mock_settings.agent_deployment_id = "test-agent"
            result = await handle_mcp_disconnect("user-1", "dcr-mcp")

        assert result == {"mcp_name": "dcr-mcp", "connected": False}
        store.delete_token.assert_awaited_once_with("test-agent", "user-1", "dcr-mcp")

    async def test_rejects_non_oauth_auth_mode(self):
        with patch(
            "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
            return_value={"enabled": True, "auth_mode": "sso"},
        ):
            with pytest.raises(HTTPException) as exc:
                await handle_mcp_disconnect("user-1", "sso-mcp")
        assert exc.value.status_code == 400
        assert "does not use OAuth" in exc.value.detail

    async def test_rejects_client_credentials(self):
        with patch(
            "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
            return_value={
                "enabled": True,
                "auth_mode": "oauth",
                "oauth": {"grant_type": "client_credentials"},
            },
        ):
            with pytest.raises(HTTPException) as exc:
                await handle_mcp_disconnect("user-1", "cc-mcp")
        assert exc.value.status_code == 400
        assert "client_credentials" in exc.value.detail


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
        result = _callback_html(error="something broke")
        assert "something broke" in result
        assert "MCP OAuth Error" in result

    def test_error_html_escapes_tags(self):
        result = _callback_html(error='<script>alert("xss")</script>')
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


_serialization_lock = asyncio.Lock()


@asynccontextmanager
async def _mock_distributed_lock_held(*args, **kwargs):
    """Stub distributed_lock that serializes via a real asyncio.Lock."""
    async with _serialization_lock:
        yield "held"


@asynccontextmanager
async def _mock_distributed_lock_timeout(*args, **kwargs):
    """Stub distributed_lock that always yields 'timeout'."""
    yield "timeout"


@asynccontextmanager
async def _mock_distributed_lock_no_redis(*args, **kwargs):
    """Stub distributed_lock that always yields 'no_redis'."""
    yield "no_redis"


@pytest.fixture
def dcr_store():
    """Return a mock McpTokenStore for DcrClientManager tests."""
    store = MagicMock(spec=McpTokenStore)
    store.get_client = AsyncMock()
    store.delete_client = AsyncMock(return_value=True)
    store.upsert_client = AsyncMock()
    return store


@pytest.fixture
def dcr_manager(dcr_store):
    """Return a DcrClientManager backed by the mock store."""
    return DcrClientManager(store=dcr_store)


_DCR_OAUTH_CFG = {
    "authorization_endpoint": "https://auth.example.com/authorize",
    "token_endpoint": "https://auth.example.com/token",
    "registration_endpoint": "https://auth.example.com/register",
}
_DCR_SERVER_CFG = {"enabled": True, "auth_mode": "dcr", "oauth": _DCR_OAUTH_CFG}


class TestIsInvalidClientError:
    """Tests for the _is_invalid_client_error helper."""

    def test_json_invalid_client(self):
        assert _is_invalid_client_error(400, '{"error": "invalid_client"}') is True

    def test_json_invalid_grant_not_matched(self):
        assert _is_invalid_client_error(400, '{"error": "invalid_grant"}') is False

    def test_bare_403_not_matched(self):
        assert _is_invalid_client_error(403, "Forbidden") is False

    def test_403_with_invalid_client_in_body(self):
        assert _is_invalid_client_error(403, '{"error": "invalid_client"}') is True

    def test_bare_401_not_matched(self):
        assert _is_invalid_client_error(401, "Unauthorized") is False

    def test_401_with_invalid_client_in_body(self):
        assert _is_invalid_client_error(401, '{"error": "invalid_client"}') is True

    def test_non_json_with_invalid_client_substring(self):
        assert _is_invalid_client_error(400, "error=invalid_client") is True

    def test_500_not_matched(self):
        assert _is_invalid_client_error(500, "Internal Server Error") is False


@pytest.mark.asyncio
class TestDcrClientManagerEnsureValidClient:
    async def test_returns_existing_valid_client(self, dcr_manager, dcr_store):
        dcr_store.get_client.return_value = McpOAuthClient(
            agent_name="agent", mcp_name="mcp", client_id="cid", client_secret="csec"
        )
        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.distributed_lock",
                _mock_distributed_lock_held,
            ),
            patch.object(
                DcrClientManager, "_validate_client", new=AsyncMock(return_value=True)
            ),
        ):
            cid, secret = await dcr_manager.ensure_valid_client(
                "agent", "mcp", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
            )

        assert cid == "cid"
        assert secret == "csec"
        dcr_store.delete_client.assert_not_awaited()

    async def test_reregisters_when_client_invalid(self, dcr_manager, dcr_store):
        dcr_store.get_client.return_value = McpOAuthClient(
            agent_name="agent", mcp_name="mcp", client_id="stale-cid"
        )
        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.distributed_lock",
                _mock_distributed_lock_held,
            ),
            patch.object(
                DcrClientManager, "_validate_client", new=AsyncMock(return_value=False)
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._register_dcr_client",
                new=AsyncMock(return_value=("new-cid", "new-sec")),
            ) as mock_register,
        ):
            cid, secret = await dcr_manager.ensure_valid_client(
                "agent", "mcp", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
            )

        assert cid == "new-cid"
        assert secret == "new-sec"
        dcr_store.delete_client.assert_awaited_once_with("agent", "mcp")
        mock_register.assert_awaited_once()

    async def test_registers_when_no_client_exists(self, dcr_manager, dcr_store):
        dcr_store.get_client.return_value = None
        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.distributed_lock",
                _mock_distributed_lock_held,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._register_dcr_client",
                new=AsyncMock(return_value=("fresh-cid", "fresh-sec")),
            ) as mock_register,
        ):
            cid, secret = await dcr_manager.ensure_valid_client(
                "agent", "mcp", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
            )

        assert cid == "fresh-cid"
        assert secret == "fresh-sec"
        dcr_store.delete_client.assert_not_awaited()
        mock_register.assert_awaited_once()

    async def test_lock_timeout_falls_back_to_stored_client(
        self, dcr_manager, dcr_store
    ):
        dcr_store.get_client.return_value = McpOAuthClient(
            agent_name="agent", mcp_name="mcp", client_id="cid", client_secret="csec"
        )
        with patch(
            "deep_agent.aegra.mcp_oauth_handlers.distributed_lock",
            _mock_distributed_lock_timeout,
        ):
            cid, secret = await dcr_manager.ensure_valid_client(
                "agent", "mcp", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
            )

        assert cid == "cid"
        assert secret == "csec"

    async def test_no_redis_falls_back_to_stored_client(self, dcr_manager, dcr_store):
        """When Redis is unavailable, fail closed — return stored client, no rotation."""
        dcr_store.get_client.return_value = McpOAuthClient(
            agent_name="agent", mcp_name="mcp", client_id="cid", client_secret="csec"
        )
        with patch(
            "deep_agent.aegra.mcp_oauth_handlers.distributed_lock",
            _mock_distributed_lock_no_redis,
        ):
            cid, secret = await dcr_manager.ensure_valid_client(
                "agent", "mcp", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
            )

        assert cid == "cid"
        assert secret == "csec"
        dcr_store.delete_client.assert_not_awaited()

    async def test_no_redis_raises_503_when_no_stored_client(
        self, dcr_manager, dcr_store
    ):
        """When Redis is unavailable and no stored client exists, raise 503."""
        dcr_store.get_client.return_value = None
        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.distributed_lock",
                _mock_distributed_lock_no_redis,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await dcr_manager.ensure_valid_client(
                "agent", "mcp", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
            )

        assert exc_info.value.status_code == 503

    async def test_serialized_registration_deduplicates(self, dcr_manager, dcr_store):
        """Second caller under the lock sees the client persisted by the first."""
        register_count = 0

        async def registering_register(*args, **kwargs):
            nonlocal register_count
            register_count += 1
            dcr_store.get_client.return_value = McpOAuthClient(
                agent_name="agent",
                mcp_name="mcp",
                client_id="cid",
                client_secret="sec",
            )
            await asyncio.sleep(0.01)
            return ("cid", "sec")

        dcr_store.get_client.return_value = None
        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.distributed_lock",
                _mock_distributed_lock_held,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._register_dcr_client",
                new=registering_register,
            ),
            patch.object(
                DcrClientManager, "_validate_client", new=AsyncMock(return_value=True)
            ),
        ):
            await asyncio.gather(
                dcr_manager.ensure_valid_client(
                    "agent", "mcp", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
                ),
                dcr_manager.ensure_valid_client(
                    "agent", "mcp", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
                ),
            )

        assert register_count == 1


@pytest.mark.asyncio
class TestDcrClientManagerHandleTokenEndpointError:
    async def test_invalid_client_json_triggers_reregistration(
        self, dcr_manager, dcr_store
    ):
        dcr_store.get_client.return_value = McpOAuthClient(
            agent_name="agent", mcp_name="mcp", client_id="old-cid"
        )
        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.distributed_lock",
                _mock_distributed_lock_held,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._register_dcr_client",
                new=AsyncMock(return_value=("new-cid", "new-sec")),
            ),
        ):
            result = await dcr_manager.handle_token_endpoint_error(
                "agent",
                "mcp",
                _DCR_OAUTH_CFG,
                _DCR_SERVER_CFG,
                400,
                '{"error": "invalid_client"}',
                rejected_client_id="old-cid",
            )

        assert result == ("new-cid", "new-sec")
        dcr_store.delete_client.assert_awaited_once_with("agent", "mcp")

    async def test_skips_rotation_when_another_worker_already_rotated(
        self, dcr_manager, dcr_store
    ):
        """If storage already holds a different client, skip re-registration."""
        dcr_store.get_client.return_value = McpOAuthClient(
            agent_name="agent",
            mcp_name="mcp",
            client_id="already-rotated",
            client_secret="sec-b",
        )
        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.distributed_lock",
                _mock_distributed_lock_held,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._register_dcr_client",
                new=AsyncMock(return_value=("should-not-be-called", "x")),
            ) as mock_register,
        ):
            result = await dcr_manager.handle_token_endpoint_error(
                "agent",
                "mcp",
                _DCR_OAUTH_CFG,
                _DCR_SERVER_CFG,
                400,
                '{"error": "invalid_client"}',
                rejected_client_id="old-cid",
            )

        assert result == ("already-rotated", "sec-b")
        dcr_store.delete_client.assert_not_awaited()
        mock_register.assert_not_awaited()

    async def test_bare_401_does_not_trigger_rotation(self, dcr_manager, dcr_store):
        result = await dcr_manager.handle_token_endpoint_error(
            "agent", "mcp", _DCR_OAUTH_CFG, _DCR_SERVER_CFG, 401, "Unauthorized"
        )

        assert result is None
        dcr_store.delete_client.assert_not_awaited()

    async def test_bare_403_does_not_trigger_rotation(self, dcr_manager, dcr_store):
        result = await dcr_manager.handle_token_endpoint_error(
            "agent", "mcp", _DCR_OAUTH_CFG, _DCR_SERVER_CFG, 403, "Forbidden"
        )

        assert result is None
        dcr_store.delete_client.assert_not_awaited()

    async def test_403_with_invalid_client_body_triggers_rotation(
        self, dcr_manager, dcr_store
    ):
        dcr_store.get_client.return_value = McpOAuthClient(
            agent_name="agent", mcp_name="mcp", client_id="old-cid"
        )
        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.distributed_lock",
                _mock_distributed_lock_held,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._register_dcr_client",
                new=AsyncMock(return_value=("new-cid", "new-sec")),
            ),
        ):
            result = await dcr_manager.handle_token_endpoint_error(
                "agent",
                "mcp",
                _DCR_OAUTH_CFG,
                _DCR_SERVER_CFG,
                403,
                '{"error": "invalid_client"}',
                rejected_client_id="old-cid",
            )

        assert result == ("new-cid", "new-sec")
        dcr_store.delete_client.assert_awaited_once()

    async def test_400_other_error_returns_none(self, dcr_manager, dcr_store):
        result = await dcr_manager.handle_token_endpoint_error(
            "agent",
            "mcp",
            _DCR_OAUTH_CFG,
            _DCR_SERVER_CFG,
            400,
            '{"error": "invalid_grant"}',
        )

        assert result is None
        dcr_store.delete_client.assert_not_awaited()

    async def test_500_returns_none(self, dcr_manager, dcr_store):
        result = await dcr_manager.handle_token_endpoint_error(
            "agent",
            "mcp",
            _DCR_OAUTH_CFG,
            _DCR_SERVER_CFG,
            500,
            "Internal Server Error",
        )

        assert result is None
        dcr_store.delete_client.assert_not_awaited()

    async def test_lock_timeout_returns_none(self, dcr_manager, dcr_store):
        with patch(
            "deep_agent.aegra.mcp_oauth_handlers.distributed_lock",
            _mock_distributed_lock_timeout,
        ):
            result = await dcr_manager.handle_token_endpoint_error(
                "agent",
                "mcp",
                _DCR_OAUTH_CFG,
                _DCR_SERVER_CFG,
                400,
                '{"error": "invalid_client"}',
            )

        assert result is None
        dcr_store.delete_client.assert_not_awaited()

    async def test_no_redis_returns_none(self, dcr_manager, dcr_store):
        """When Redis is unavailable, fail closed — do not rotate."""
        with patch(
            "deep_agent.aegra.mcp_oauth_handlers.distributed_lock",
            _mock_distributed_lock_no_redis,
        ):
            result = await dcr_manager.handle_token_endpoint_error(
                "agent",
                "mcp",
                _DCR_OAUTH_CFG,
                _DCR_SERVER_CFG,
                400,
                '{"error": "invalid_client"}',
            )

        assert result is None
        dcr_store.delete_client.assert_not_awaited()


@pytest.mark.asyncio
class TestDcrClientManagerValidateClient:
    async def test_returns_true_on_302_redirect(self):
        mock_response = MagicMock()
        mock_response.status_code = 302
        mock_response.text = ""

        with patch(
            "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient"
        ) as mock_client:
            mock_ctx = AsyncMock(get=AsyncMock(return_value=mock_response))
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await DcrClientManager._validate_client(
                "cid", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
            )

        assert result is True

    async def test_returns_false_on_invalid_client_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "error: invalid_client"

        with patch(
            "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient"
        ) as mock_client:
            mock_ctx = AsyncMock(get=AsyncMock(return_value=mock_response))
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await DcrClientManager._validate_client(
                "expired-cid", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
            )

        assert result is False

    async def test_returns_true_when_no_authorization_endpoint(self):
        result = await DcrClientManager._validate_client("cid", {}, _DCR_SERVER_CFG)
        assert result is True

    async def test_returns_false_on_400_invalid_request(self):
        """400 with 'invalid_request' (e.g. unregistered redirect_uri) should be invalid."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = '{"error":"invalid_request","error_description":"Unregistered redirect_uri"}'

        with patch(
            "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient"
        ) as mock_client:
            mock_ctx = AsyncMock(get=AsyncMock(return_value=mock_response))
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await DcrClientManager._validate_client(
                "bogus-uuid", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
            )

        assert result is False

    async def test_returns_false_on_401(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch(
            "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient"
        ) as mock_client:
            mock_ctx = AsyncMock(get=AsyncMock(return_value=mock_response))
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await DcrClientManager._validate_client(
                "bad-cid", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
            )

        assert result is False

    async def test_returns_false_on_500_server_error(self):
        """500 errors are treated as invalid client (e.g. Atlassian returns 500 for unknown client_id)."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch(
            "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient"
        ) as mock_client:
            mock_ctx = AsyncMock(get=AsyncMock(return_value=mock_response))
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await DcrClientManager._validate_client(
                "cid", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
            )

        assert result is False

    async def test_returns_true_on_502_gateway_error(self):
        """Non-500 5xx errors are transient — assume client is valid."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.text = "Bad Gateway"

        with patch(
            "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient"
        ) as mock_client:
            mock_ctx = AsyncMock(get=AsyncMock(return_value=mock_response))
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await DcrClientManager._validate_client(
                "cid", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
            )

        assert result is True

    async def test_returns_true_on_network_error(self):
        with patch(
            "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient"
        ) as mock_client:
            mock_ctx = AsyncMock(
                get=AsyncMock(side_effect=Exception("connection refused"))
            )
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await DcrClientManager._validate_client(
                "cid", _DCR_OAUTH_CFG, _DCR_SERVER_CFG
            )

        assert result is True


@pytest.mark.asyncio
class TestHandleMcpConnectWithDcrManager:
    async def test_dcr_connect_uses_manager(self):
        server_cfg = {
            "enabled": True,
            "auth_mode": "dcr",
            "oauth": {
                "authorization_endpoint": "https://auth.example.com/authorize",
                "token_endpoint": "https://auth.example.com/token",
                "registration_endpoint": "https://auth.example.com/register",
            },
        }

        mock_manager = MagicMock()
        mock_manager.ensure_valid_client = AsyncMock(
            return_value=("dcr-cid", "dcr-sec")
        )

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value=server_cfg,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.get_dcr_client_manager",
                return_value=mock_manager,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch("deep_agent.aegra.mcp_oauth_handlers.cache_set", return_value=True),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_settings.agent_deployment_id = "test-agent"
            mock_settings.MCP_DCR_ENABLED = True
            result = await handle_mcp_connect("user-1", "dcr-mcp")

        assert "authorize_url" in result
        assert "dcr-cid" in result["authorize_url"]
        mock_manager.ensure_valid_client.assert_awaited_once()


@pytest.mark.asyncio
class TestGetDcrClientManagerSingleton:
    async def test_returns_dcr_client_manager_instance(self):
        import deep_agent.aegra.mcp_oauth_handlers as mod

        mod._default_dcr_manager = None
        try:
            with patch.object(McpTokenStore, "__init__", return_value=None):
                mgr = get_dcr_client_manager()
            assert isinstance(mgr, DcrClientManager)
            assert get_dcr_client_manager() is mgr
        finally:
            mod._default_dcr_manager = None


@pytest.mark.asyncio
class TestHandleMcpCallbackWithDcrManager:
    async def test_callback_dcr_path_uses_manager(self):
        """The DCR branch in handle_mcp_oauth_callback uses the manager."""
        server_cfg = {
            "enabled": True,
            "auth_mode": "dcr",
            "oauth": {
                "authorization_endpoint": "https://auth.example.com/authorize",
                "token_endpoint": "https://auth.example.com/token",
                "registration_endpoint": "https://auth.example.com/register",
            },
        }

        state_data = json.dumps(
            {
                "user_id": "user-1",
                "mcp_name": "dcr-mcp",
                "code_verifier": "verifier123",
            }
        )

        mock_manager = MagicMock()
        mock_manager.ensure_valid_client = AsyncMock(
            return_value=("dcr-cid", "dcr-secret")
        )

        token_body = {
            "access_token": "at-abc",
            "refresh_token": "rt-abc",
            "expires_in": 3600,
        }

        mock_request = MagicMock(spec=Request)

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value=server_cfg,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.get_dcr_client_manager",
                return_value=mock_manager,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.cache_get",
                return_value=state_data,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient"
            ) as mock_httpx,
            patch.object(McpTokenStore, "__init__", return_value=None),
            patch.object(McpTokenStore, "upsert_token", new=AsyncMock()),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.get_mcp_credential_resolver"
            ) as mock_resolver,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.validate_granted_scopes",
                return_value=["read"],
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.parse_token_scopes",
                return_value=["read"],
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.requested_scopes",
                return_value=["read"],
            ),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_settings.agent_deployment_id = "test-agent"
            mock_settings.database_uri = "postgresql://localhost/test"
            mock_settings.MCP_DCR_ENABLED = True
            mock_settings.ui_origin = "https://ui.example.com"

            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = token_body

            mock_ctx = AsyncMock()
            mock_ctx.post = AsyncMock(return_value=mock_resp)
            mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_resolver.return_value.invalidate_cache = MagicMock()

            with patch(
                "deep_agent.aegra.mcp_oauth_handlers.invalidate_mcp_tool_cache",
                create=True,
            ):
                result = await handle_mcp_oauth_callback(
                    "auth-code", "state-123", mock_request
                )

        assert result.status_code == 200
        mock_manager.ensure_valid_client.assert_awaited_once()
