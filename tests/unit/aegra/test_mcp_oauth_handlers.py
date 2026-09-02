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
    handle_mcp_connections,
    handle_mcp_disconnect,
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
