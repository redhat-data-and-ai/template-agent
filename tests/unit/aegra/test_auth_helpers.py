"""Unit tests for shared authentication helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from deep_agent.aegra.auth_helpers import (
    authenticated_user_id,
    check_ldap_role,
    memory_user_id,
    rule_user_ids,
)


class TestAuthenticatedUserId:
    @pytest.mark.asyncio
    async def test_returns_dev_user_when_auth_disabled(self):
        request = MagicMock()
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", False),
            patch("deep_agent.aegra.auth.DEV_USER_ID", "dev-user"),
        ):
            result = await authenticated_user_id(request)
        assert result == "dev-user"

    @pytest.mark.asyncio
    async def test_returns_anonymous_when_no_bearer_token(self):
        request = MagicMock()
        request.headers = {"authorization": ""}
        with patch("deep_agent.aegra.auth.ENABLE_AUTH", True):
            result = await authenticated_user_id(request)
        assert result == "anonymous"

    @pytest.mark.asyncio
    async def test_raises_401_when_reject_anonymous_and_no_token(self):
        request = MagicMock()
        request.headers = {"authorization": ""}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            pytest.raises(HTTPException) as exc_info,
        ):
            await authenticated_user_id(request, reject_anonymous=True)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_extracts_sub_from_valid_jwt(self):
        request = MagicMock()
        request.headers = {"authorization": "Bearer valid-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": "user-123"},
            ),
        ):
            result = await authenticated_user_id(request)
        assert result == "user-123"

    @pytest.mark.asyncio
    async def test_invalid_token_propagates_exception(self):
        """An expired or malformed JWT causes _decode_token to raise; verify it propagates."""
        request = MagicMock()
        request.headers = {"authorization": "Bearer expired-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                side_effect=Exception("Token expired"),
            ),
            pytest.raises(Exception, match="Token expired"),
        ):
            await authenticated_user_id(request)


class TestMemoryUserId:
    @pytest.mark.asyncio
    async def test_auth_off_uses_x_user_id(self):
        request = MagicMock()
        request.headers = {"x-user-id": "johnwick"}
        with patch("deep_agent.aegra.auth.ENABLE_AUTH", False):
            assert await memory_user_id(request) == "johnwick"

    @pytest.mark.asyncio
    async def test_auth_off_falls_back_to_dev_user(self):
        request = MagicMock()
        request.headers = {}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", False),
            patch("deep_agent.aegra.auth.DEV_USER_ID", "dev-user"),
        ):
            assert await memory_user_id(request) == "dev-user"

    @pytest.mark.asyncio
    async def test_auth_on_requires_bearer(self):
        request = MagicMock()
        request.headers = {}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            pytest.raises(HTTPException) as exc_info,
        ):
            await memory_user_id(request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_prefers_preferred_username_over_sub(self):
        request = MagicMock()
        request.headers = {"authorization": "Bearer valid-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": "uuid-1", "preferred_username": "dpundir"},
            ),
        ):
            assert await memory_user_id(request) == "dpundir"

    @pytest.mark.asyncio
    async def test_raises_when_token_has_no_user_claims(self):
        request = MagicMock()
        request.headers = {"authorization": "Bearer valid-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("deep_agent.aegra.auth._decode_token", return_value={}),
            pytest.raises(HTTPException) as exc_info,
        ):
            await memory_user_id(request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_rejects_spoofed_x_user_id(self):
        request = MagicMock()
        request.headers = {
            "authorization": "Bearer valid-token",
            "x-user-id": "victim",
        }
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": "uuid-1", "preferred_username": "dpundir"},
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await memory_user_id(request)
        assert exc_info.value.status_code == 403


class TestCheckLdapRole:
    """Tests the Developer/Eval column from the behavior matrix."""

    @pytest.mark.asyncio
    async def test_auth_disabled_returns_dev_user(self):
        """AUTH_ENABLED=false → full access (eval=Yes)."""
        request = MagicMock()
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", False),
            patch("deep_agent.aegra.auth.DEV_USER_ID", "dev-user"),
        ):
            result = await check_ldap_role(request, developer_only=True)
        assert result == "dev-user"

    @pytest.mark.asyncio
    async def test_no_bearer_raises_401(self):
        request = MagicMock()
        request.headers = {"authorization": ""}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            pytest.raises(HTTPException) as exc_info,
        ):
            await check_ldap_role(request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_denied_role_raises_403(self):
        """Groups defined, private, no match → 'denied' → 403."""
        request = MagicMock()
        request.headers = {"authorization": "Bearer valid-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"preferred_username": "outsider", "sub": "uuid-1"},
            ),
            patch(
                "deep_agent.src.ldap.resolve_user_role",
                new_callable=AsyncMock,
                return_value="denied",
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await check_ldap_role(request)
        assert exc_info.value.status_code == 403
        assert "not a member" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_users_role_passes_when_not_developer_only(self):
        """Users role → check_ldap_role without developer_only=True passes."""
        request = MagicMock()
        request.headers = {"authorization": "Bearer valid-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"preferred_username": "alice", "sub": "uuid-1"},
            ),
            patch(
                "deep_agent.src.ldap.resolve_user_role",
                new_callable=AsyncMock,
                return_value="users",
            ),
        ):
            result = await check_ldap_role(request)
        assert result == "alice"

    @pytest.mark.asyncio
    async def test_users_role_rejected_when_developer_only(self):
        """Users role + developer_only=True → 403 (eval=No)."""
        request = MagicMock()
        request.headers = {"authorization": "Bearer valid-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"preferred_username": "alice", "sub": "uuid-1"},
            ),
            patch(
                "deep_agent.src.ldap.resolve_user_role",
                new_callable=AsyncMock,
                return_value="users",
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await check_ldap_role(request, developer_only=True)
        assert exc_info.value.status_code == 403
        assert "Developer access" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_owners_pass_developer_only(self):
        """Owners role + developer_only=True → passes (eval=Yes)."""
        request = MagicMock()
        request.headers = {"authorization": "Bearer valid-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"preferred_username": "admin-user", "sub": "uuid-1"},
            ),
            patch(
                "deep_agent.src.ldap.resolve_user_role",
                new_callable=AsyncMock,
                return_value="owners",
            ),
        ):
            result = await check_ldap_role(request, developer_only=True)
        assert result == "admin-user"

    @pytest.mark.asyncio
    async def test_admins_pass_developer_only(self):
        request = MagicMock()
        request.headers = {"authorization": "Bearer valid-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"preferred_username": "admin-user", "sub": "uuid-1"},
            ),
            patch(
                "deep_agent.src.ldap.resolve_user_role",
                new_callable=AsyncMock,
                return_value="admins",
            ),
        ):
            result = await check_ldap_role(request, developer_only=True)
        assert result == "admin-user"

    @pytest.mark.asyncio
    async def test_builders_pass_developer_only(self):
        request = MagicMock()
        request.headers = {"authorization": "Bearer valid-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"preferred_username": "builder-user", "sub": "uuid-1"},
            ),
            patch(
                "deep_agent.src.ldap.resolve_user_role",
                new_callable=AsyncMock,
                return_value="builders",
            ),
        ):
            result = await check_ldap_role(request, developer_only=True)
        assert result == "builder-user"

    @pytest.mark.asyncio
    async def test_missing_user_identity_raises_401(self):
        request = MagicMock()
        request.headers = {"authorization": "Bearer valid-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": ""},
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await check_ldap_role(request)
        assert exc_info.value.status_code == 401


class TestAuthenticatedUserIdDeveloperOnly:
    """Tests the developer_only parameter on authenticated_user_id."""

    @pytest.mark.asyncio
    async def test_developer_only_blocks_users_role(self):
        request = MagicMock()
        request.headers = {"authorization": "Bearer valid-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": "user-1", "preferred_username": "alice"},
            ),
            patch(
                "deep_agent.src.ldap.resolve_user_role",
                new_callable=AsyncMock,
                return_value="users",
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await authenticated_user_id(request, developer_only=True)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_developer_only_passes_owners(self):
        request = MagicMock()
        request.headers = {"authorization": "Bearer valid-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": "user-1", "preferred_username": "alice"},
            ),
            patch(
                "deep_agent.src.ldap.resolve_user_role",
                new_callable=AsyncMock,
                return_value="owners",
            ),
        ):
            result = await authenticated_user_id(request, developer_only=True)
        assert result == "user-1"

    @pytest.mark.asyncio
    async def test_developer_only_none_role_passes(self):
        """When resolve_user_role returns None (no groups), developer_only does not block."""
        request = MagicMock()
        request.headers = {"authorization": "Bearer valid-token"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": "user-1", "preferred_username": "alice"},
            ),
            patch(
                "deep_agent.src.ldap.resolve_user_role",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await authenticated_user_id(request, developer_only=True)
        assert result == "user-1"


class TestRuleUserIds:
    @pytest.mark.asyncio
    async def test_uses_username_not_jwt_sub(self):
        request = MagicMock()
        request.headers = {
            "authorization": "Bearer valid-token",
            "x-user-id": "dpundir",
        }
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": "uuid-1", "preferred_username": "dpundir"},
            ),
        ):
            assert await rule_user_ids(request) == ["dpundir"]

    @pytest.mark.asyncio
    async def test_does_not_duplicate_when_sub_is_username(self):
        request = MagicMock()
        request.headers = {
            "authorization": "Bearer valid-token",
            "x-user-id": "uuid-1",
        }
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": "uuid-1"},
            ),
        ):
            assert await rule_user_ids(request) == ["uuid-1"]
