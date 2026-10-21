"""Unit tests for deep_agent/aegra/mcp.py."""

from __future__ import annotations

import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import logging

from types import SimpleNamespace

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
        mcp_mod._cached_connected_set["user1:server1"] = {"gitlab"}
        mcp_mod._cached_tools["user2:server1"] = [MagicMock()]
        mcp_mod._cached_tools_ts["user2:server1"] = 2.0
        mcp_mod._cached_connected_set["user2:server1"] = {"jira"}

        mcp_mod.invalidate_mcp_tool_cache("user1")
        assert "user1:server1" not in mcp_mod._cached_tools
        assert "user1:server1" not in mcp_mod._cached_connected_set
        assert "user2:server1" in mcp_mod._cached_tools
        assert "user2:server1" in mcp_mod._cached_connected_set

        mcp_mod._cached_tools.clear()
        mcp_mod._cached_tools_ts.clear()
        mcp_mod._cached_connected_set.clear()

    def test_clears_all_when_none(self):
        mcp_mod._cached_tools["user1:a"] = [MagicMock()]
        mcp_mod._cached_tools["user2:b"] = [MagicMock()]
        mcp_mod._cached_connected_set["user1:a"] = {"gitlab"}
        mcp_mod._cached_connected_set["user2:b"] = {"jira"}

        mcp_mod.invalidate_mcp_tool_cache(None)
        assert len(mcp_mod._cached_tools) == 0
        assert len(mcp_mod._cached_tools_ts) == 0
        assert len(mcp_mod._cached_connected_set) == 0


class TestGetMcpToolsCrossPodCache:
    """Tests for cross-pod cache invalidation via the Redis auth set."""

    def _seed_cache(self, user_id: str, server_key: str, tools: list, connected: set):
        cache_key = f"{user_id}:{server_key}"
        mcp_mod._cached_tools[cache_key] = tools
        mcp_mod._cached_tools_ts[cache_key] = time.time()
        mcp_mod._cached_connected_set[cache_key] = connected

    def _cleanup(self):
        mcp_mod._cached_tools.clear()
        mcp_mod._cached_tools_ts.clear()
        mcp_mod._cached_connected_set.clear()

    @pytest.mark.asyncio
    async def test_cache_hit_when_redis_set_matches(self):
        """If the Redis set matches the local snapshot, return cached tools."""
        fake_tool = MagicMock()
        fake_tool.name = "tool1"
        self._seed_cache("user1", "gitlab", [fake_tool], {"gitlab"})

        with (
            patch(
                "deep_agent.aegra.redis.cache_smembers",
                return_value={"gitlab"},
            ),
            patch("deep_agent.aegra.mcp.settings") as mock_settings,
        ):
            mock_settings.agent_deployment_id = "test-agent"
            result = await mcp_mod.get_mcp_tools(
                sso_token=None, server_names=["gitlab"], user_id="user1"
            )

        assert result == [fake_tool]
        self._cleanup()

    @pytest.mark.asyncio
    async def test_cache_evicted_when_redis_set_differs(self):
        """If the Redis set changed (server added), evict cache and reload."""
        stale_tool = MagicMock()
        stale_tool.name = "stale"
        self._seed_cache("user1", "gitlab", [stale_tool], {"gitlab"})

        fresh_tool = MagicMock()
        fresh_tool.name = "fresh"

        with (
            patch(
                "deep_agent.aegra.redis.cache_smembers",
                return_value={"gitlab", "jira"},
            ),
            patch("deep_agent.aegra.mcp.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "gitlab": {
                        "enabled": True,
                        "url": "http://mcp:8080",
                        "auth_mode": "sso",
                    },
                },
            ),
            patch(
                "deep_agent.aegra.mcp._connect_single_server",
                new_callable=AsyncMock,
                return_value=[fresh_tool],
            ),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new_callable=AsyncMock,
                return_value="tok",
            ),
        ):
            mock_settings.agent_deployment_id = "test-agent"
            result = await mcp_mod.get_mcp_tools(
                sso_token="tok", server_names=["gitlab"], user_id="user1"
            )

        # Plus a resume placeholder per connected server.
        assert [t for t in result if not t.name.startswith("mcp__")] == [fresh_tool]
        assert stale_tool not in result
        self._cleanup()

    @pytest.mark.asyncio
    async def test_cache_evicted_when_server_removed(self):
        """If the Redis set shrinks (server disconnected), evict cache and reload."""
        stale_tool = MagicMock()
        stale_tool.name = "stale"
        self._seed_cache("user1", "gitlab", [stale_tool], {"gitlab", "jira"})

        fresh_tool = MagicMock()
        fresh_tool.name = "fresh"

        with (
            patch(
                "deep_agent.aegra.redis.cache_smembers",
                return_value={"gitlab"},
            ),
            patch("deep_agent.aegra.mcp.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "gitlab": {
                        "enabled": True,
                        "url": "http://mcp:8080",
                        "auth_mode": "sso",
                    },
                },
            ),
            patch(
                "deep_agent.aegra.mcp._connect_single_server",
                new_callable=AsyncMock,
                return_value=[fresh_tool],
            ),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new_callable=AsyncMock,
                return_value="tok",
            ),
        ):
            mock_settings.agent_deployment_id = "test-agent"
            result = await mcp_mod.get_mcp_tools(
                sso_token="tok", server_names=["gitlab"], user_id="user1"
            )

        # Plus a resume placeholder per connected server.
        assert [t for t in result if not t.name.startswith("mcp__")] == [fresh_tool]
        self._cleanup()

    @pytest.mark.asyncio
    async def test_set_order_does_not_matter(self):
        """{'gitlab', 'jira'} == {'jira', 'gitlab'} — no false invalidation."""
        fake_tool = MagicMock()
        fake_tool.name = "tool1"
        self._seed_cache("user1", "gitlab,jira", [fake_tool], {"jira", "gitlab"})

        with (
            patch(
                "deep_agent.aegra.redis.cache_smembers",
                return_value={"gitlab", "jira"},
            ),
            patch("deep_agent.aegra.mcp.settings") as mock_settings,
        ):
            mock_settings.agent_deployment_id = "test-agent"
            result = await mcp_mod.get_mcp_tools(
                sso_token=None, server_names=["gitlab", "jira"], user_id="user1"
            )

        assert result == [fake_tool]
        self._cleanup()

    @pytest.mark.asyncio
    async def test_snapshot_saved_after_fresh_load(self):
        """After a fresh tool load, the connected set snapshot is persisted."""
        fresh_tool = MagicMock()
        fresh_tool.name = "tool1"

        with (
            patch(
                "deep_agent.aegra.redis.cache_smembers",
                return_value={"gitlab"},
            ),
            patch("deep_agent.aegra.mcp.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "gitlab": {
                        "enabled": True,
                        "url": "http://mcp:8080",
                        "auth_mode": "sso",
                    },
                },
            ),
            patch(
                "deep_agent.aegra.mcp._connect_single_server",
                new_callable=AsyncMock,
                return_value=[fresh_tool],
            ),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new_callable=AsyncMock,
                return_value="tok",
            ),
        ):
            mock_settings.agent_deployment_id = "test-agent"
            await mcp_mod.get_mcp_tools(
                sso_token="tok", server_names=["gitlab"], user_id="user1"
            )

        cache_key = "user1:gitlab"
        assert cache_key in mcp_mod._cached_connected_set
        assert mcp_mod._cached_connected_set[cache_key] == {"gitlab"}
        self._cleanup()


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


class TestTokenInjectorInterceptorFallback:
    """Verify that the SSO-token fallback only applies to auth_mode='sso'."""

    def _make_interceptor(self, auth_mode, mcp_name="test-mcp"):
        return mcp_mod._TokenInjectorInterceptor(
            mcp_name=mcp_name,
            server_cfg={"auth_mode": auth_mode},
            server_key=mcp_name,
        )

    async def test_sso_mode_falls_back_to_sso_token_on_error(self):
        """auth_mode=sso: resolver failure falls back to the SSO token."""
        interceptor = self._make_interceptor("sso")
        handler = AsyncMock(return_value="response")
        request = MagicMock()
        request.override = MagicMock(return_value=request)

        mcp_mod._current_access_token.set("sso-tok")
        mcp_mod._current_refresh_token.set(None)
        mcp_mod._current_user_id.set("user-1")

        with patch(
            "deep_agent.aegra.mcp.refresh_access_token",
            new=AsyncMock(side_effect=RuntimeError("SSO refresh exploded")),
        ):
            result = await interceptor(request, handler)

        request.override.assert_called_once()
        auth_header = request.override.call_args[1]["headers"]["Authorization"]
        assert auth_header == "Bearer sso-tok"
        assert result == "response"

    async def test_oauth_mode_raises_on_resolver_failure(self):
        """auth_mode=oauth: resolver failure must NOT fall back to SSO token."""
        interceptor = self._make_interceptor("oauth")
        handler = AsyncMock(return_value="response")
        request = MagicMock()

        mcp_mod._current_access_token.set("sso-tok")
        mcp_mod._current_user_id.set("user-1")
        mcp_mod._mcp_tool_discovery.set(False)

        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(side_effect=RuntimeError("Vault timeout"))
        with patch(
            "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
            return_value=mock_resolver,
        ):
            with pytest.raises(
                RuntimeError, match=r"\[test-mcp\] oauth credential resolution failed"
            ) as exc_info:
                await interceptor(request, handler)
            assert exc_info.value.__cause__ is not None
            assert "Vault timeout" in str(exc_info.value.__cause__)

        handler.assert_not_called()

    async def test_dcr_mode_raises_on_resolver_failure(self):
        """auth_mode=dcr: resolver failure must NOT fall back to SSO token."""
        interceptor = self._make_interceptor("dcr")
        handler = AsyncMock(return_value="response")
        request = MagicMock()

        mcp_mod._current_access_token.set("sso-tok")
        mcp_mod._current_user_id.set("user-1")
        mcp_mod._mcp_tool_discovery.set(False)

        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(side_effect=ConnectionError("network down"))
        with patch(
            "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
            return_value=mock_resolver,
        ):
            with pytest.raises(
                RuntimeError, match=r"\[test-mcp\] dcr credential resolution failed"
            ) as exc_info:
                await interceptor(request, handler)
            assert isinstance(exc_info.value.__cause__, ConnectionError)

        handler.assert_not_called()

    async def test_oauth_needs_authorization_still_propagates(self):
        """NeedsAuthorization is re-raised for oauth/dcr (unchanged behavior)."""
        from deep_agent.aegra.mcp_auth import NeedsAuthorization

        interceptor = self._make_interceptor("oauth")
        handler = AsyncMock()
        request = MagicMock()

        mcp_mod._current_access_token.set("sso-tok")
        mcp_mod._current_user_id.set("user-1")
        mcp_mod._mcp_tool_discovery.set(False)

        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(
            side_effect=NeedsAuthorization("test-mcp", "/connect")
        )
        with patch(
            "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
            return_value=mock_resolver,
        ):
            with pytest.raises(NeedsAuthorization):
                await interceptor(request, handler)

        handler.assert_not_called()

    async def test_sso_needs_authorization_still_propagates(self):
        """NeedsAuthorization is re-raised for sso mode too (unchanged behavior)."""
        from deep_agent.aegra.mcp_auth import NeedsAuthorization

        interceptor = self._make_interceptor("sso")
        handler = AsyncMock()
        request = MagicMock()

        mcp_mod._current_access_token.set("sso-tok")
        mcp_mod._current_refresh_token.set(None)
        mcp_mod._current_user_id.set("user-1")

        with patch(
            "deep_agent.aegra.mcp.refresh_access_token",
            new=AsyncMock(side_effect=NeedsAuthorization("test-mcp", "/connect")),
        ):
            with pytest.raises(NeedsAuthorization):
                await interceptor(request, handler)

        handler.assert_not_called()

    async def test_api_key_mode_skips_token_injection(self):
        """auth_mode=api_key: handler called directly, no token injection."""
        interceptor = self._make_interceptor("api_key")
        handler = AsyncMock(return_value="response")
        request = MagicMock()

        mcp_mod._current_access_token.set("sso-tok-should-not-leak")

        result = await interceptor(request, handler)

        handler.assert_called_once_with(request)
        request.override.assert_not_called()
        assert result == "response"


class TestTokenRemaining:
    """Opaque tokens must not be mistaken for expired ones.

    `_jwt_exp` returns 0.0 both for a token it cannot parse and for one
    that declares `exp: 0`. Treating the first as expired makes every
    tool call refresh, adding an IdP round trip per call.
    """

    @staticmethod
    def _jwt(exp: float) -> str:
        import base64
        import json

        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        payload = (
            base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode())
            .rstrip(b"=")
            .decode()
        )
        return f"{header}.{payload}.sig"

    def test_opaque_token_is_treated_as_immeasurable(self):
        import math

        from deep_agent.aegra import mcp as mcp_mod

        assert mcp_mod._token_remaining("opaque-reference-token") == math.inf

    def test_jwt_without_exp_claim_is_immeasurable(self):
        import base64
        import json
        import math

        from deep_agent.aegra import mcp as mcp_mod

        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        payload = (
            base64.urlsafe_b64encode(json.dumps({"sub": "u1"}).encode())
            .rstrip(b"=")
            .decode()
        )
        assert mcp_mod._token_remaining(f"{header}.{payload}.sig") == math.inf

    def test_explicit_zero_exp_is_expired_not_immeasurable(self):
        """`exp: 0` is a real claim — expired at the epoch, so refresh it."""
        from deep_agent.aegra import mcp as mcp_mod

        assert mcp_mod._token_remaining(self._jwt(0)) < 0

    def test_future_expiry_reports_remaining_seconds(self):
        import time

        from deep_agent.aegra import mcp as mcp_mod

        remaining = mcp_mod._token_remaining(self._jwt(time.time() + 3600))
        assert 3500 < remaining <= 3600

    def test_past_expiry_is_negative(self):
        import time

        from deep_agent.aegra import mcp as mcp_mod

        assert mcp_mod._token_remaining(self._jwt(time.time() - 60)) < 0


class TestPerServerCircuitBreakers:
    """One flaky MCP server must not strip every other server's tools."""

    def test_breakers_are_distinct_per_server(self):
        from deep_agent.aegra import mcp as mcp_mod

        mcp_mod._mcp_breakers.clear()
        first = mcp_mod._get_mcp_breaker("server-a")
        second = mcp_mod._get_mcp_breaker("server-b")

        assert first is not second
        assert first is mcp_mod._get_mcp_breaker("server-a")

    def test_breaker_names_are_namespaced(self):
        from deep_agent.aegra import mcp as mcp_mod

        mcp_mod._mcp_breakers.clear()
        assert mcp_mod._get_mcp_breaker("server-a").name == "mcp-server:server-a"

    def test_unnamed_caller_gets_the_shared_breaker(self):
        """Back-compat: callers that name no server keep one shared circuit."""
        from deep_agent.aegra import mcp as mcp_mod

        mcp_mod._mcp_breakers.clear()
        assert (
            mcp_mod._get_mcp_breaker().name
            == f"mcp-server:{mcp_mod._SHARED_BREAKER_KEY}"
        )

    def test_tripping_one_breaker_leaves_others_closed(self):
        from deep_agent.aegra import mcp as mcp_mod
        from deep_agent.src.error_handling import CircuitBreaker

        mcp_mod._mcp_breakers.clear()
        # In-memory breakers, so the assertion does not depend on whether a
        # Redis backend happens to be reachable in this test session.
        tripped = CircuitBreaker(name="mcp-server:a", threshold=1)
        tripped.record_failure()
        mcp_mod._mcp_breakers["a"] = tripped
        mcp_mod._mcp_breakers["b"] = CircuitBreaker(name="mcp-server:b", threshold=1)

        assert mcp_mod._get_mcp_breaker("a").is_open is True
        assert mcp_mod._get_mcp_breaker("b").is_open is False


# ── tool budget ───────────────────────────────────────────────────────


class TestToolBudget:
    """Bound tools are a per-turn cost, so the count needs reporting.

    Every schema goes into the system prompt on every model call.
    This agent's count went from 193 to 322 unnoticed — roughly
    120,000 input tokens on a question that used none of them —
    because nothing said what it cost.
    """

    @staticmethod
    def _tools(*names):
        """Build tools the way the pipeline does — attributed by metadata.

        Only auth placeholders carry an `mcp__` prefix in the name;
        discovered tools keep the server's own naming and are
        attributed by `prepare_tools_for_model`.
        """
        out = []
        for n in names:
            parts = n.split("__")
            server = parts[1] if len(parts) >= 2 and parts[0] == "mcp" else None
            out.append(
                SimpleNamespace(
                    name=n, metadata={"mcp_server": server} if server else {}
                )
            )
        return out

    @staticmethod
    def _attributed(server, count, description="d"):
        """Discovered tools: server in metadata, no prefix in the name."""
        return [
            SimpleNamespace(
                name=f"{server}_tool_{i}",
                description=description,
                metadata={"mcp_server": server},
            )
            for i in range(count)
        ]

    def test_warns_past_the_budget(self, caplog):
        import deep_agent.aegra.mcp as mcp_mod

        tools = self._tools(*[f"mcp__atlan__t{i}" for i in range(200)])
        with caplog.at_level(logging.WARNING):
            mcp_mod._log_tool_budget(tools)

        assert any("Tool budget" in r.message for r in caplog.records)
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_stays_quiet_under_the_budget(self, caplog):
        import deep_agent.aegra.mcp as mcp_mod

        with caplog.at_level(logging.WARNING):
            mcp_mod._log_tool_budget(self._tools("mcp__atlan__a", "mcp__gitlab__b"))

        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_orders_servers_by_token_cost(self, caplog):
        """Fattest first: schemas differ by an order of magnitude.

        A count alone does not answer "what do I drop" — twenty
        terse tools can cost less than three verbose ones.
        """
        import deep_agent.aegra.mcp as mcp_mod

        fat = self._attributed("fat", 1, "x" * 4000)
        thin = self._attributed("thin", 10, "y")
        with caplog.at_level(logging.INFO):
            mcp_mod._log_tool_budget([*thin, *fat])

        logged = " ".join(r.message % r.args for r in caplog.records)
        assert logged.index("fat=") < logged.index("thin="), logged

    def test_breaks_the_count_down_by_server(self, caplog):
        """Which server to scope is the decision this has to inform."""
        import deep_agent.aegra.mcp as mcp_mod

        # Real names are `mcp__<server>__<tool>`. The first version
        # of this split on the leading segment and reported every
        # server as "mcp".
        tools = self._tools(
            "mcp__atlan__a",
            "mcp__atlan__b",
            "mcp__atlan__c",
            "mcp__gitlab__x",
            "write_todos",
        )
        with caplog.at_level(logging.INFO):
            mcp_mod._log_tool_budget(tools)

        logged = " ".join(r.message % r.args for r in caplog.records)
        assert "atlan=3t/" in logged
        assert "gitlab=1t/" in logged
        assert "(built-in)=1t/" in logged
        assert "mcp=" not in logged

    def test_auth_placeholders_count_against_their_server(self, caplog):
        """`mcp__<server>` with no tool segment is a connect prompt."""
        import deep_agent.aegra.mcp as mcp_mod

        with caplog.at_level(logging.INFO):
            mcp_mod._log_tool_budget(self._tools("mcp__atlan", "mcp__gitlab"))

        logged = " ".join(r.message % r.args for r in caplog.records)
        assert "atlan=1t/" in logged
        assert "gitlab=1t/" in logged

    def test_empty_tool_set_logs_nothing(self, caplog):
        import deep_agent.aegra.mcp as mcp_mod

        with caplog.at_level(logging.INFO):
            mcp_mod._log_tool_budget([])
        assert not caplog.records


class TestToolBudgetAttribution:
    """Discovered tools are attributed by metadata, not by name.

    Only the auth placeholders are named `mcp__<server>`. Reading
    the name instead reported all 319 discovered tools as built-ins,
    which is the opposite of the number the breakdown exists to give.
    """

    def test_metadata_wins_over_the_name(self, caplog):
        import deep_agent.aegra.mcp as mcp_mod

        tools = [
            SimpleNamespace(
                name="search_assets", description="d", metadata={"mcp_server": "atlan"}
            ),
            SimpleNamespace(
                name="list_issues", description="d", metadata={"mcp_server": "gitlab"}
            ),
        ]
        with caplog.at_level(logging.INFO):
            mcp_mod._log_tool_budget(tools)

        logged = " ".join(r.message % r.args for r in caplog.records)
        assert "atlan=1t/" in logged
        assert "gitlab=1t/" in logged
        assert "(built-in)" not in logged

    def test_unattributed_tools_are_built_ins(self, caplog):
        import deep_agent.aegra.mcp as mcp_mod

        with caplog.at_level(logging.INFO):
            mcp_mod._log_tool_budget(
                [SimpleNamespace(name="write_todos", description="d", metadata={})]
            )
        logged = " ".join(r.message % r.args for r in caplog.records)
        assert "(built-in)=1t/" in logged


class TestToolCacheEviction:
    """The per-user tool cache has to be bounded.

    Each entry is ~615,705 characters of schema for 319 tools,
    before object overhead. The TTL is consulted only when an entry
    is read, so a user who stops using the agent is never read again
    and — before this — never freed. Growth tracked unique users
    since pod start against a 2Gi limit already carrying ~1.2Gi.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        import deep_agent.aegra.mcp as mcp_mod

        mcp_mod._cached_tools.clear()
        mcp_mod._cached_tools_ts.clear()
        mcp_mod._cached_connected_set.clear()
        mcp_mod._refreshing.clear()
        yield
        mcp_mod._cached_tools.clear()
        mcp_mod._cached_tools_ts.clear()
        mcp_mod._cached_connected_set.clear()
        mcp_mod._refreshing.clear()

    def test_evicts_past_the_bound(self, monkeypatch):
        import deep_agent.aegra.mcp as mcp_mod

        monkeypatch.setattr(mcp_mod, "_MCP_TOOL_CACHE_MAX_USERS", 3)
        for i in range(5):
            mcp_mod._store_cached_tools(f"user{i}:", [f"t{i}"], set())

        assert len(mcp_mod._cached_tools) == 3
        assert "user0:" not in mcp_mod._cached_tools
        assert "user4:" in mcp_mod._cached_tools

    def test_eviction_clears_every_parallel_structure(self, monkeypatch):
        """An entry in one dict and not another is a stale auth set."""
        import deep_agent.aegra.mcp as mcp_mod

        monkeypatch.setattr(mcp_mod, "_MCP_TOOL_CACHE_MAX_USERS", 1)
        mcp_mod._store_cached_tools("gone:", ["t"], {"atlan"})
        mcp_mod._refreshing.add("gone:")
        mcp_mod._store_cached_tools("kept:", ["t"], {"gitlab"})

        assert "gone:" not in mcp_mod._cached_tools
        assert "gone:" not in mcp_mod._cached_tools_ts
        assert "gone:" not in mcp_mod._cached_connected_set
        assert "gone:" not in mcp_mod._refreshing

    def test_reading_an_entry_protects_it_from_eviction(self, monkeypatch):
        """Least recently *used*, not least recently written.

        A long-lived user whose list rarely refreshes is exactly the
        entry worth keeping, and insertion order would evict it
        first.
        """
        import deep_agent.aegra.mcp as mcp_mod

        monkeypatch.setattr(mcp_mod, "_MCP_TOOL_CACHE_MAX_USERS", 2)
        mcp_mod._store_cached_tools("old:", ["t"], set())
        mcp_mod._store_cached_tools("new:", ["t"], set())

        mcp_mod._touch_cached_tools("old:")
        mcp_mod._store_cached_tools("newest:", ["t"], set())

        assert "old:" in mcp_mod._cached_tools
        assert "new:" not in mcp_mod._cached_tools

    def test_forget_is_idempotent_on_a_missing_key(self):
        import deep_agent.aegra.mcp as mcp_mod

        mcp_mod._forget_cached_tools("never-existed:")  # must not raise

    def test_touching_a_missing_key_is_a_no_op(self):
        import deep_agent.aegra.mcp as mcp_mod

        mcp_mod._touch_cached_tools("never-existed:")
        assert "never-existed:" not in mcp_mod._cached_tools
