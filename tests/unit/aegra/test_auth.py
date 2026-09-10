"""Unit tests for aegra auth module."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest

from deep_agent.aegra.auth import (
    _build_dev_user,
    _decode_sub_unverified,
    _make_user,
    _resolve_jwks_uri,
    encrypt_user_id,
)


class TestEncryptUserId:
    def test_passthrough_when_disabled(self):
        with patch.dict(os.environ, {}, clear=False):
            with patch("deep_agent.aegra.auth.ENABLE_USER_ID_ENCRYPTION", False):
                assert encrypt_user_id("user123") == "user123"

    def test_passthrough_when_no_key(self):
        with patch("deep_agent.aegra.auth.ENABLE_USER_ID_ENCRYPTION", True):
            with patch("deep_agent.aegra.auth.USER_ID_ENCRYPTION_KEY", ""):
                assert encrypt_user_id("user123") == "user123"

    def test_deterministic_encryption(self):
        with patch("deep_agent.aegra.auth.ENABLE_USER_ID_ENCRYPTION", True):
            with patch(
                "deep_agent.aegra.auth.USER_ID_ENCRYPTION_KEY",
                "secret_key_32_bytes_hex",
            ):
                result1 = encrypt_user_id("user123")
                result2 = encrypt_user_id("user123")
                assert result1 == result2
                assert result1 != "user123"
                assert len(result1) == 16

    def test_different_users_different_hashes(self):
        with patch("deep_agent.aegra.auth.ENABLE_USER_ID_ENCRYPTION", True):
            with patch(
                "deep_agent.aegra.auth.USER_ID_ENCRYPTION_KEY",
                "secret_key_32_bytes_hex",
            ):
                r1 = encrypt_user_id("alice")
                r2 = encrypt_user_id("bob")
                assert r1 != r2


class TestBuildDevUser:
    def test_dev_user_structure(self):
        user = _build_dev_user()
        assert user["is_authenticated"] is True
        assert "identity" in user
        assert "display_name" in user
        assert "permissions" in user
        assert "admin" in user["permissions"]
        assert "email" in user

    def test_dev_user_identity(self):
        with patch("deep_agent.aegra.auth.DEV_USER_ID", "custom-dev"):
            user = _build_dev_user()
            assert user["identity"] == "custom-dev"

    def test_dev_user_ldap_role_is_none(self):
        """AUTH_ENABLED=false → ldap_role=None → full access (chat=Yes, eval=Yes)."""
        user = _build_dev_user()
        assert user["ldap_role"] is None


class TestCheckLdapAccess:
    """Tests the chat-access gate from the behavior matrix.

    _check_ldap_access enforces the Chat column:
        - None → pass (AUTH_ENABLED=false, full access)
        - 'denied' → PermissionError (private + no match → 403)
        - 'users' → pass (chat allowed)
        - 'owners/admins/builders' → pass (chat allowed)
    """

    def test_none_passes(self):
        from deep_agent.aegra.auth import _check_ldap_access

        _check_ldap_access(None)

    def test_denied_raises(self):
        from deep_agent.aegra.auth import _check_ldap_access

        with pytest.raises(PermissionError, match="not a member"):
            _check_ldap_access("denied")

    def test_users_passes(self):
        from deep_agent.aegra.auth import _check_ldap_access

        _check_ldap_access("users")

    def test_owners_passes(self):
        from deep_agent.aegra.auth import _check_ldap_access

        _check_ldap_access("owners")

    def test_admins_passes(self):
        from deep_agent.aegra.auth import _check_ldap_access

        _check_ldap_access("admins")

    def test_builders_passes(self):
        from deep_agent.aegra.auth import _check_ldap_access

        _check_ldap_access("builders")


class TestResolveLdapRole:
    @pytest.mark.asyncio
    async def test_extracts_preferred_username(self):
        from deep_agent.aegra.auth import _resolve_ldap_role

        payload = {"preferred_username": "alice", "sub": "uuid-1"}
        with patch(
            "deep_agent.src.ldap.service.resolve_user_role",
            new_callable=AsyncMock,
            return_value="owners",
        ) as mock_resolve:
            result = await _resolve_ldap_role(payload)
        assert result == "owners"
        mock_resolve.assert_called_once_with("alice")

    @pytest.mark.asyncio
    async def test_falls_back_to_sub(self):
        from deep_agent.aegra.auth import _resolve_ldap_role

        payload = {"sub": "bob"}
        with patch(
            "deep_agent.src.ldap.service.resolve_user_role",
            new_callable=AsyncMock,
            return_value="users",
        ) as mock_resolve:
            result = await _resolve_ldap_role(payload)
        assert result == "users"
        mock_resolve.assert_called_once_with("bob")

    @pytest.mark.asyncio
    async def test_empty_user_id_returns_denied(self):
        from deep_agent.aegra.auth import _resolve_ldap_role

        payload = {"sub": ""}
        result = await _resolve_ldap_role(payload)
        assert result == "denied"

    @pytest.mark.asyncio
    async def test_missing_claims_returns_denied(self):
        from deep_agent.aegra.auth import _resolve_ldap_role

        result = await _resolve_ldap_role({})
        assert result == "denied"


class TestResolveJwksUri:
    def test_explicit_jwks_uri(self):
        with patch(
            "deep_agent.aegra.auth.SSO_JWKS_URI", "https://sso.example.com/jwks"
        ):
            result = _resolve_jwks_uri()
            assert result == "https://sso.example.com/jwks"

    def test_cached_uri(self):
        with patch("deep_agent.aegra.auth.SSO_JWKS_URI", ""):
            with patch.dict(
                os.environ, {"_RESOLVED_JWKS_URI": "https://cached.example.com/jwks"}
            ):
                result = _resolve_jwks_uri()
                assert result == "https://cached.example.com/jwks"

    def test_missing_issuer_raises(self):
        with patch("deep_agent.aegra.auth.SSO_JWKS_URI", ""):
            with patch("deep_agent.aegra.auth.SSO_ISSUER_URL", ""):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("_RESOLVED_JWKS_URI", None)
                    with pytest.raises(RuntimeError, match="SSO_ISSUER_URL"):
                        _resolve_jwks_uri()


class TestDecodeSubUnverified:
    def test_valid_jwt_returns_sub(self):
        token = pyjwt.encode(
            {"sub": "user-42", "exp": 9999999999}, "secret", algorithm="HS256"
        )
        assert _decode_sub_unverified(token) == "user-42"

    def test_jwt_without_sub_returns_none(self):
        token = pyjwt.encode({"exp": 9999999999}, "secret", algorithm="HS256")
        assert _decode_sub_unverified(token) is None

    def test_malformed_token_returns_none(self):
        assert _decode_sub_unverified("not.a.jwt") is None

    def test_empty_string_returns_none(self):
        assert _decode_sub_unverified("") is None


class TestOidcRefresh:
    async def test_successful_refresh(self):
        from deep_agent.aegra.auth import _oidc_refresh

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
        }
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("deep_agent.aegra.auth.httpx.AsyncClient", return_value=mock_client):
            access, refresh = await _oidc_refresh("old-refresh")

        assert access == "new-access"
        assert refresh == "new-refresh"

    async def test_refresh_falls_back_to_original_rt(self):
        from deep_agent.aegra.auth import _oidc_refresh

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"access_token": "new-access"}
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("deep_agent.aegra.auth.httpx.AsyncClient", return_value=mock_client):
            access, refresh = await _oidc_refresh("original-rt")

        assert access == "new-access"
        assert refresh == "original-rt"

    async def test_http_error_propagates(self):
        import httpx

        from deep_agent.aegra.auth import _oidc_refresh

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=mock_resp
        )
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("deep_agent.aegra.auth.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                await _oidc_refresh("bad-rt")


class TestGetJwksClient:
    def test_caches_client(self):
        import deep_agent.aegra.auth as auth_mod

        auth_mod._jwks_client = None
        mock_client = MagicMock()
        with patch(
            "deep_agent.aegra.auth._resolve_jwks_uri",
            return_value="https://jwks.example.com",
        ):
            with patch(
                "deep_agent.aegra.auth.jwt.PyJWKClient", return_value=mock_client
            ) as mock_cls:
                first = auth_mod._get_jwks_client()
                second = auth_mod._get_jwks_client()

        assert first is second
        mock_cls.assert_called_once()
        auth_mod._jwks_client = None

    def test_raises_runtime_error_on_failure(self):
        import deep_agent.aegra.auth as auth_mod

        auth_mod._jwks_client = None
        with patch(
            "deep_agent.aegra.auth._resolve_jwks_uri", side_effect=Exception("no URI")
        ):
            with pytest.raises(RuntimeError, match="JWKS initialization failed"):
                auth_mod._get_jwks_client()
        auth_mod._jwks_client = None


class TestDecodeToken:
    def test_successful_decode(self):
        from deep_agent.aegra.auth import _decode_token

        mock_key = MagicMock()
        mock_key.key = "test-key"
        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_key
        expected_payload = {"sub": "u1", "exp": 9999999999, "iss": "https://sso"}

        with (
            patch(
                "deep_agent.aegra.auth._get_jwks_client", return_value=mock_jwks_client
            ),
            patch("deep_agent.aegra.auth.jwt.decode", return_value=expected_payload),
        ):
            result = _decode_token("some-jwt")

        assert result["sub"] == "u1"

    def test_audience_skipped_when_empty(self):
        from deep_agent.aegra.auth import _decode_token

        mock_key = MagicMock()
        mock_key.key = "test-key"
        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_key

        with (
            patch(
                "deep_agent.aegra.auth._get_jwks_client", return_value=mock_jwks_client
            ),
            patch("deep_agent.aegra.auth.SSO_JWT_AUDIENCE", ""),
            patch(
                "deep_agent.aegra.auth.jwt.decode", return_value={"sub": "u1"}
            ) as mock_decode,
        ):
            _decode_token("tok")

        call_kwargs = mock_decode.call_args[1]
        assert "audience" not in call_kwargs
        assert call_kwargs["options"]["verify_aud"] is False

    def test_audience_provided_when_set(self):
        from deep_agent.aegra.auth import _decode_token

        mock_key = MagicMock()
        mock_key.key = "test-key"
        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_key

        with (
            patch(
                "deep_agent.aegra.auth._get_jwks_client", return_value=mock_jwks_client
            ),
            patch("deep_agent.aegra.auth.SSO_JWT_AUDIENCE", "my-aud"),
            patch(
                "deep_agent.aegra.auth.jwt.decode", return_value={"sub": "u1"}
            ) as mock_decode,
        ):
            _decode_token("tok")

        call_kwargs = mock_decode.call_args[1]
        assert call_kwargs["audience"] == "my-aud"


class TestMakeUser:
    def test_builds_correct_user_dict(self):
        payload = {
            "sub": "user-1",
            "name": "Alice",
            "email": "alice@example.com",
            "realm_access": {"roles": ["admin", "user"]},
        }
        result = _make_user(payload, "access-tok", "refresh-tok")
        assert result["identity"] == "user-1"
        assert result["display_name"] == "Alice"
        assert result["email"] == "alice@example.com"
        assert result["permissions"] == ["admin", "user"]
        assert result["is_authenticated"] is True
        assert result["access_token"] == "access-tok"
        assert result["refresh_token"] == "refresh-tok"

    def test_handles_missing_optional_fields(self):
        payload = {"sub": "user-2"}
        result = _make_user(payload, "at", "rt")
        assert result["identity"] == "user-2"
        assert result["display_name"] == ""
        assert result["email"] == ""
        assert result["permissions"] == []

    def test_preferred_username_fallback(self):
        payload = {"sub": "user-3", "preferred_username": "jdoe"}
        result = _make_user(payload, "at", "rt")
        assert result["display_name"] == "jdoe"


class TestAuthenticate:
    async def test_dev_user_when_auth_disabled(self):
        from deep_agent.aegra.auth import authenticate

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", False),
            patch("deep_agent.aegra.auth.ENVIRONMENT", "development"),
        ):
            result = await authenticate({"authorization": ""})

        assert result["is_authenticated"] is True
        assert result["identity"] == "dev-user"

    async def test_raises_when_auth_disabled_in_production(self):
        from deep_agent.aegra.auth import authenticate

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", False),
            patch("deep_agent.aegra.auth.ENVIRONMENT", "production"),
        ):
            with pytest.raises(PermissionError, match="production"):
                await authenticate({"authorization": ""})

    async def test_raises_when_no_bearer_header(self):
        from deep_agent.aegra.auth import authenticate

        with patch("deep_agent.aegra.auth.ENABLE_AUTH", True):
            with pytest.raises(PermissionError, match="Missing"):
                await authenticate({"authorization": "Basic abc"})

    async def test_raises_when_auth_header_empty(self):
        from deep_agent.aegra.auth import authenticate

        with patch("deep_agent.aegra.auth.ENABLE_AUTH", True):
            with pytest.raises(PermissionError, match="Missing"):
                await authenticate({"authorization": ""})

    async def test_happy_path_valid_token(self):
        from deep_agent.aegra.auth import authenticate

        payload = {
            "sub": "user-99",
            "name": "Test User",
            "email": "test@example.com",
            "realm_access": {"roles": ["viewer"]},
        }
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("deep_agent.aegra.auth.EVAL_TOKEN_REFRESH_ENABLED", False),
            patch("deep_agent.aegra.auth._decode_token", return_value=payload),
            patch(
                "deep_agent.aegra.auth._resolve_ldap_role",
                new_callable=AsyncMock,
                return_value="users",
            ),
        ):
            result = await authenticate(
                {"authorization": "Bearer valid-token", "x-refresh-token": ""}
            )

        assert result["identity"] == "user-99"
        assert result["access_token"] == "valid-token"
        assert result["ldap_role"] == "users"

    async def test_expired_token_raises_when_refresh_disabled(self):
        from deep_agent.aegra.auth import authenticate

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("deep_agent.aegra.auth.EVAL_TOKEN_REFRESH_ENABLED", False),
            patch(
                "deep_agent.aegra.auth._decode_token",
                side_effect=pyjwt.ExpiredSignatureError("Token expired"),
            ),
        ):
            with pytest.raises(PermissionError, match="Token expired"):
                await authenticate(
                    {"authorization": "Bearer expired-token", "x-refresh-token": "rt"}
                )

    async def test_valid_token_updates_redis_when_eval_active(self):
        from deep_agent.aegra.auth import authenticate

        payload = {"sub": "user-7", "name": "Active Eval User"}

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("deep_agent.aegra.auth.EVAL_TOKEN_REFRESH_ENABLED", True),
            patch("deep_agent.aegra.auth._decode_token", return_value=payload),
            patch("deep_agent.aegra.redis.get_redis_client", return_value=MagicMock()),
            patch("deep_agent.aegra.redis.cache_get", return_value="1") as mock_get,
            patch("deep_agent.aegra.redis.cache_set") as mock_set,
            patch("deep_agent.aegra.mcp_crypto.encrypt_secret", return_value="enc-rt"),
            patch(
                "deep_agent.aegra.auth._resolve_ldap_role",
                new_callable=AsyncMock,
                return_value="owners",
            ),
        ):
            result = await authenticate(
                {"authorization": "Bearer valid-tok", "x-refresh-token": "my-rt"}
            )

        assert result["identity"] == "user-7"
        mock_get.assert_called_once_with("eval:active:user-7")
        set_keys = [c.args[0] for c in mock_set.call_args_list]
        assert any("eval:refresh:user-7" in k for k in set_keys)

    async def test_valid_token_skips_redis_when_no_refresh_token(self):
        from deep_agent.aegra.auth import authenticate

        payload = {"sub": "user-8"}

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("deep_agent.aegra.auth.EVAL_TOKEN_REFRESH_ENABLED", True),
            patch("deep_agent.aegra.auth._decode_token", return_value=payload),
            patch("deep_agent.aegra.redis.cache_set") as mock_set,
            patch(
                "deep_agent.aegra.auth._resolve_ldap_role",
                new_callable=AsyncMock,
                return_value="users",
            ),
        ):
            result = await authenticate(
                {"authorization": "Bearer tok", "x-refresh-token": ""}
            )

        assert result["identity"] == "user-8"
        mock_set.assert_not_called()

    async def test_expired_token_raises_when_redis_unavailable(self):
        from deep_agent.aegra.auth import authenticate

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("deep_agent.aegra.auth.EVAL_TOKEN_REFRESH_ENABLED", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                side_effect=pyjwt.ExpiredSignatureError(),
            ),
            patch("deep_agent.aegra.redis.get_redis_client", return_value=None),
        ):
            with pytest.raises(PermissionError, match="Token expired"):
                await authenticate(
                    {"authorization": "Bearer expired", "x-refresh-token": "rt"}
                )

    async def test_expired_token_raises_when_sub_unreadable(self):
        from deep_agent.aegra.auth import authenticate

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("deep_agent.aegra.auth.EVAL_TOKEN_REFRESH_ENABLED", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                side_effect=pyjwt.ExpiredSignatureError(),
            ),
            patch("deep_agent.aegra.redis.get_redis_client", return_value=MagicMock()),
            patch("deep_agent.aegra.auth._decode_sub_unverified", return_value=None),
        ):
            with pytest.raises(PermissionError, match="sub unreadable"):
                await authenticate(
                    {"authorization": "Bearer expired", "x-refresh-token": "rt"}
                )

    async def test_expired_token_raises_when_no_active_eval(self):
        from deep_agent.aegra.auth import authenticate

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("deep_agent.aegra.auth.EVAL_TOKEN_REFRESH_ENABLED", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                side_effect=pyjwt.ExpiredSignatureError(),
            ),
            patch("deep_agent.aegra.redis.get_redis_client", return_value=MagicMock()),
            patch(
                "deep_agent.aegra.auth._decode_sub_unverified", return_value="user-x"
            ),
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
        ):
            with pytest.raises(PermissionError, match="no active eval"):
                await authenticate(
                    {"authorization": "Bearer expired", "x-refresh-token": "rt"}
                )

    async def test_expired_token_cache_hit_returns_user(self):
        """Line 259: cached access token is valid — return user without OIDC."""
        from deep_agent.aegra.auth import authenticate

        refreshed_payload = {"sub": "user-cache", "name": "Cached"}

        def _cache_get(key: str) -> str | None:
            if key == "eval:active:user-cache":
                return "1"
            if key == "eval:access:user-cache":
                return "enc-access"
            if key == "eval:refresh:user-cache":
                return "enc-refresh"
            return None

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("deep_agent.aegra.auth.EVAL_TOKEN_REFRESH_ENABLED", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                side_effect=[pyjwt.ExpiredSignatureError(), refreshed_payload],
            ),
            patch("deep_agent.aegra.redis.get_redis_client", return_value=MagicMock()),
            patch(
                "deep_agent.aegra.auth._decode_sub_unverified",
                return_value="user-cache",
            ),
            patch("deep_agent.aegra.redis.cache_get", side_effect=_cache_get),
            patch(
                "deep_agent.aegra.mcp_crypto.decrypt_secret",
                side_effect=lambda v: f"dec-{v}",
            ),
            patch(
                "deep_agent.aegra.auth._resolve_ldap_role",
                new_callable=AsyncMock,
                return_value="owners",
            ),
        ):
            result = await authenticate(
                {"authorization": "Bearer expired", "x-refresh-token": "rt"}
            )

        assert result["identity"] == "user-cache"
        assert result["display_name"] == "Cached"

    async def test_expired_token_lock_winner_refreshes(self):
        """Line 299: lock holder refreshes via OIDC and returns user."""
        from contextlib import asynccontextmanager

        from deep_agent.aegra.auth import authenticate

        refreshed_payload = {"sub": "user-winner", "name": "Winner"}

        def _cache_get(key: str) -> str | None:
            if key == "eval:active:user-winner":
                return "1"
            if key == "eval:refresh:user-winner":
                return "enc-rt"
            return None

        @asynccontextmanager
        async def _fake_lock(*a, **kw):
            yield "held"

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("deep_agent.aegra.auth.EVAL_TOKEN_REFRESH_ENABLED", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                side_effect=[pyjwt.ExpiredSignatureError(), refreshed_payload],
            ),
            patch("deep_agent.aegra.redis.get_redis_client", return_value=MagicMock()),
            patch(
                "deep_agent.aegra.auth._decode_sub_unverified",
                return_value="user-winner",
            ),
            patch("deep_agent.aegra.redis.cache_get", side_effect=_cache_get),
            patch("deep_agent.aegra.redis.cache_set"),
            patch("deep_agent.aegra.redis.distributed_lock", _fake_lock),
            patch(
                "deep_agent.aegra.mcp_crypto.decrypt_secret", return_value="plain-rt"
            ),
            patch("deep_agent.aegra.mcp_crypto.encrypt_secret", return_value="enc-val"),
            patch(
                "deep_agent.aegra.auth._oidc_refresh",
                return_value=("new-access", "new-rt"),
            ),
            patch(
                "deep_agent.aegra.auth._resolve_ldap_role",
                new_callable=AsyncMock,
                return_value="owners",
            ),
        ):
            result = await authenticate(
                {"authorization": "Bearer expired", "x-refresh-token": "rt"}
            )

        assert result["identity"] == "user-winner"
        assert result["display_name"] == "Winner"

    async def test_expired_token_lock_loser_polls_cache(self):
        """Line 315: lock loser polls until winner writes cached token."""
        from contextlib import asynccontextmanager

        from deep_agent.aegra.auth import authenticate

        polled_payload = {"sub": "user-loser", "name": "Loser"}
        poll_count = 0

        def _cache_get(key: str) -> str | None:
            nonlocal poll_count
            if key == "eval:active:user-loser":
                return "1"
            if key == "eval:access:user-loser":
                poll_count += 1
                if poll_count >= 2:
                    return "enc-polled-access"
                return None
            if key == "eval:refresh:user-loser":
                return "enc-rt"
            return None

        @asynccontextmanager
        async def _fake_lock(*a, **kw):
            yield "timeout"

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("deep_agent.aegra.auth.EVAL_TOKEN_REFRESH_ENABLED", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                side_effect=[pyjwt.ExpiredSignatureError(), polled_payload],
            ),
            patch("deep_agent.aegra.redis.get_redis_client", return_value=MagicMock()),
            patch(
                "deep_agent.aegra.auth._decode_sub_unverified",
                return_value="user-loser",
            ),
            patch("deep_agent.aegra.redis.cache_get", side_effect=_cache_get),
            patch("deep_agent.aegra.redis.distributed_lock", _fake_lock),
            patch(
                "deep_agent.aegra.mcp_crypto.decrypt_secret",
                side_effect=lambda v: f"dec-{v}",
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch(
                "deep_agent.aegra.auth._resolve_ldap_role",
                new_callable=AsyncMock,
                return_value="owners",
            ),
        ):
            result = await authenticate(
                {"authorization": "Bearer expired", "x-refresh-token": "rt"}
            )

        assert result["identity"] == "user-loser"
        assert result["display_name"] == "Loser"
