"""Unit tests for deep_agent/aegra/mcp.py."""

from __future__ import annotations

import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import deep_agent.aegra.mcp as mcp_mod


def _make_jwt(payload: dict) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).rstrip(b"=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
    return f"{header.decode()}.{body.decode()}.sig"


class TestSetMcpAuthContext:
    def test_sets_context_vars(self):
        mcp_mod.set_mcp_auth_context("access-tok", "refresh-tok", "user-1")
        assert mcp_mod._current_access_token.get() == "access-tok"
        assert mcp_mod._current_refresh_token.get() == "refresh-tok"
        assert mcp_mod._current_user_id.get() == "user-1"

    def test_sets_none_values(self):
        mcp_mod.set_mcp_auth_context(None, None, None)
        assert mcp_mod._current_access_token.get() is None
        assert mcp_mod._current_refresh_token.get() is None
        assert mcp_mod._current_user_id.get() is None


class TestMcpHttpxVerify:
    def test_returns_true_by_default(self):
        assert mcp_mod.mcp_httpx_verify({}) is True

    def test_returns_true_when_ssl_verify_true(self):
        assert mcp_mod.mcp_httpx_verify({"ssl_verify": True}) is True

    def test_returns_false_when_ssl_verify_false_non_production(self):
        mock_settings = MagicMock()
        mock_settings.is_production = False
        with patch("deep_agent.src.settings.settings", mock_settings):
            assert mcp_mod.mcp_httpx_verify({"ssl_verify": False}) is False

    def test_enforces_true_in_production(self):
        mock_settings = MagicMock()
        mock_settings.is_production = True
        with patch("deep_agent.src.settings.settings", mock_settings):
            assert mcp_mod.mcp_httpx_verify({"ssl_verify": False}) is True


class TestJwtExp:
    def test_extracts_exp_from_valid_jwt(self):
        exp_ts = time.time() + 300
        token = _make_jwt({"exp": exp_ts, "sub": "user"})
        assert mcp_mod._jwt_exp(token) == exp_ts

    def test_returns_zero_on_invalid_token(self):
        assert mcp_mod._jwt_exp("not-a-jwt") == 0.0

    def test_returns_zero_when_no_exp(self):
        token = _make_jwt({"sub": "user"})
        assert mcp_mod._jwt_exp(token) == 0.0


class TestRefreshAccessToken:
    async def test_returns_original_when_still_valid(self):
        exp_ts = time.time() + 300
        token = _make_jwt({"exp": exp_ts})
        result = await mcp_mod.refresh_access_token(token, "refresh-tok")
        assert result == token

    async def test_returns_original_when_no_refresh_token(self):
        exp_ts = time.time() - 10
        token = _make_jwt({"exp": exp_ts})
        result = await mcp_mod.refresh_access_token(token, None)
        assert result == token

    async def test_returns_original_when_no_sso_config(self):
        exp_ts = time.time() - 10
        token = _make_jwt({"exp": exp_ts})
        with patch.object(mcp_mod, "_SSO_TOKEN_URL", ""):
            with patch.dict("os.environ", {"SSO_ISSUER_URL": "", "SSO_CLIENT_ID": ""}):
                result = await mcp_mod.refresh_access_token(token, "refresh-tok")
        assert result == token

    async def test_refreshes_when_near_expiry(self):
        exp_ts = time.time() - 10
        old_token = _make_jwt({"exp": exp_ts})
        new_exp = time.time() + 600
        new_token = _make_jwt({"exp": new_exp, "sub": "user"})

        with patch.object(mcp_mod, "_SSO_TOKEN_URL", "https://sso/token"):
            with patch.dict("os.environ", {"SSO_CLIENT_ID": "client"}):
                with patch(
                    "deep_agent.aegra.auth._oidc_refresh",
                    new=AsyncMock(return_value=(new_token, "new-rt")),
                ):
                    with patch(
                        "deep_agent.aegra.auth.EVAL_TOKEN_REFRESH_ENABLED", False
                    ):
                        result = await mcp_mod.refresh_access_token(
                            old_token, "refresh-tok"
                        )
        assert result == new_token

    async def test_returns_original_on_refresh_failure(self):
        exp_ts = time.time() - 10
        old_token = _make_jwt({"exp": exp_ts})

        with patch.object(mcp_mod, "_SSO_TOKEN_URL", "https://sso/token"):
            with patch.dict("os.environ", {"SSO_CLIENT_ID": "client"}):
                with patch(
                    "deep_agent.aegra.auth._oidc_refresh",
                    new=AsyncMock(side_effect=Exception("OIDC failed")),
                ):
                    result = await mcp_mod.refresh_access_token(
                        old_token, "refresh-tok"
                    )
        assert result == old_token


class TestFilterByNames:
    def test_returns_all_when_none(self):
        enabled = {"a": {"url": "..."}, "b": {"url": "..."}}
        assert mcp_mod._filter_by_names(enabled, None) == enabled

    def test_filters_to_requested(self):
        enabled = {"a": {"url": "1"}, "b": {"url": "2"}, "c": {"url": "3"}}
        result = mcp_mod._filter_by_names(enabled, ["a", "c"])
        assert set(result.keys()) == {"a", "c"}

    def test_warns_for_missing_names(self, caplog):
        enabled = {"a": {"url": "1"}}
        result = mcp_mod._filter_by_names(enabled, ["a", "missing"])
        assert "missing" in caplog.text
        assert set(result.keys()) == {"a"}

    def test_empty_list_returns_empty(self):
        enabled = {"a": {"url": "1"}}
        result = mcp_mod._filter_by_names(enabled, [])
        assert result == enabled


class TestInvalidateMcpToolCache:
    def test_clears_specific_user(self):
        mcp_mod._cached_tools["user1:server1"] = [MagicMock()]
        mcp_mod._cached_tools_ts["user1:server1"] = 1.0
        mcp_mod._cached_tools["user2:server1"] = [MagicMock()]
        mcp_mod._cached_tools_ts["user2:server1"] = 2.0

        mcp_mod.invalidate_mcp_tool_cache("user1")
        assert "user1:server1" not in mcp_mod._cached_tools
        assert "user2:server1" in mcp_mod._cached_tools

        mcp_mod._cached_tools.clear()
        mcp_mod._cached_tools_ts.clear()

    def test_clears_all_when_none(self):
        mcp_mod._cached_tools["user1:a"] = [MagicMock()]
        mcp_mod._cached_tools["user2:b"] = [MagicMock()]

        mcp_mod.invalidate_mcp_tool_cache(None)
        assert len(mcp_mod._cached_tools) == 0
        assert len(mcp_mod._cached_tools_ts) == 0


class TestResolveConnectionToken:
    async def test_returns_sso_token_for_sso_mode(self):
        result = await mcp_mod._resolve_connection_token(
            "server1", {"auth_mode": "sso"}, "my-sso-tok", "user1"
        )
        assert result == "my-sso-tok"

    async def test_returns_api_key_for_api_key_mode(self):
        with patch.dict("os.environ", {"MY_API_KEY": "secret123"}):
            result = await mcp_mod._resolve_connection_token(
                "server1",
                {"auth_mode": "api_key", "auth_env_var": "MY_API_KEY"},
                "sso-tok",
                "user1",
            )
        assert result == "secret123"

    async def test_returns_none_for_api_key_mode_missing_env(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("MISSING_KEY", None)
            result = await mcp_mod._resolve_connection_token(
                "server1",
                {"auth_mode": "api_key", "auth_env_var": "MISSING_KEY"},
                "sso-tok",
                "user1",
            )
        assert result is None

    async def test_returns_none_for_oauth_without_user_id(self):
        result = await mcp_mod._resolve_connection_token(
            "server1", {"auth_mode": "oauth"}, "sso-tok", None
        )
        assert result is None

    async def test_resolves_credential_for_oauth_with_user_id(self):
        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value="oauth-token")
        with patch(
            "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
            return_value=mock_resolver,
        ):
            result = await mcp_mod._resolve_connection_token(
                "server1", {"auth_mode": "dcr"}, "sso-tok", "user1"
            )
        assert result == "oauth-token"

    async def test_returns_none_on_needs_authorization(self):
        from deep_agent.aegra.mcp_auth import NeedsAuthorization

        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(
            side_effect=NeedsAuthorization("server1", "/connect")
        )
        with patch(
            "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
            return_value=mock_resolver,
        ):
            result = await mcp_mod._resolve_connection_token(
                "server1", {"auth_mode": "oauth"}, "sso-tok", "user1"
            )
        assert result is None


class TestIsNeedsAuthorization:
    def test_true_for_needs_authorization(self):
        from deep_agent.aegra.mcp_auth import NeedsAuthorization

        exc = NeedsAuthorization("srv", "/connect")
        assert mcp_mod._is_needs_authorization(exc) is True

    def test_false_for_other_exception(self):
        assert mcp_mod._is_needs_authorization(ValueError("nope")) is False

    def test_true_when_in_cause_chain(self):
        from deep_agent.aegra.mcp_auth import NeedsAuthorization

        inner = NeedsAuthorization("srv", "/connect")
        outer = RuntimeError("wrapper")
        outer.__cause__ = inner
        assert mcp_mod._is_needs_authorization(outer) is True

    def test_true_in_exception_group(self):
        from deep_agent.aegra.mcp_auth import NeedsAuthorization

        inner = NeedsAuthorization("srv", "/connect")
        group = ExceptionGroup("group", [inner])
        assert mcp_mod._is_needs_authorization(group) is True


class TestIsAuthError:
    def test_true_for_401_status(self):
        exc = Exception("fail")
        exc.response = MagicMock()
        exc.response.status_code = 401
        assert mcp_mod._is_auth_error(exc) is True

    def test_true_for_403_status(self):
        exc = Exception("fail")
        exc.response = MagicMock()
        exc.response.status_code = 403
        assert mcp_mod._is_auth_error(exc) is True

    def test_true_for_401_in_message(self):
        assert mcp_mod._is_auth_error(Exception("Got 401 error")) is True

    def test_true_for_unauthorized_in_message(self):
        assert mcp_mod._is_auth_error(Exception("Unauthorized access")) is True

    def test_false_for_unrelated_error(self):
        assert mcp_mod._is_auth_error(Exception("timeout")) is False

    def test_true_in_cause_chain(self):
        inner = Exception("401 Unauthorized")
        outer = RuntimeError("wrapper")
        outer.__cause__ = inner
        assert mcp_mod._is_auth_error(outer) is True


class TestIsConnectionError:
    def test_true_for_connect_error(self):
        assert mcp_mod._is_connection_error(Exception("ConnectError: refused")) is True

    def test_true_for_connection_refused(self):
        assert mcp_mod._is_connection_error(Exception("connection refused")) is True

    def test_true_for_connection_attempts_failed(self):
        assert (
            mcp_mod._is_connection_error(Exception("connection attempts failed"))
            is True
        )

    def test_false_for_unrelated_error(self):
        assert mcp_mod._is_connection_error(Exception("timeout")) is False

    def test_true_in_cause_chain(self):
        inner = Exception("connection refused")
        outer = RuntimeError("wrapper")
        outer.__cause__ = inner
        assert mcp_mod._is_connection_error(outer) is True


class TestBuildServerConfig:
    def test_basic_config(self):
        with patch("deep_agent.utils.pylogger._trace_id_var") as mock_trace:
            mock_trace.get.return_value = None
            config = mcp_mod._build_server_config(
                {"url": "http://mcp:8080", "auth": True},
                "my-token",
            )
        assert config["url"] == "http://mcp:8080"
        assert config["headers"]["Authorization"] == "Bearer my-token"

    def test_no_auth_header_when_no_token(self):
        with patch("deep_agent.utils.pylogger._trace_id_var") as mock_trace:
            mock_trace.get.return_value = None
            config = mcp_mod._build_server_config(
                {"url": "http://mcp:8080", "auth": True},
                None,
            )
        assert "Authorization" not in config["headers"]

    def test_no_auth_header_when_auth_false(self):
        with patch("deep_agent.utils.pylogger._trace_id_var") as mock_trace:
            mock_trace.get.return_value = None
            config = mcp_mod._build_server_config(
                {"url": "http://mcp:8080", "auth": False},
                "my-token",
            )
        assert "Authorization" not in config["headers"]

    def test_includes_trace_id(self):
        with patch("deep_agent.utils.pylogger._trace_id_var") as mock_trace:
            mock_trace.get.return_value = "trace-abc"
            config = mcp_mod._build_server_config(
                {"url": "http://mcp:8080", "auth": False},
                None,
            )
        assert config["headers"]["X-Trace-ID"] == "trace-abc"
