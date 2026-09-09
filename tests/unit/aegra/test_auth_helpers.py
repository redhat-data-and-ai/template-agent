"""Unit tests for shared authentication helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from deep_agent.aegra.auth_helpers import (
    _normalize_roles,
    authenticated_user_id,
    check_group_access,
    memory_user_id,
    rule_user_ids,
)


class TestNormalizeRoles:
    def test_normal_payload(self):
        assert _normalize_roles({"realm_access": {"roles": ["dev", "user"]}}) == [
            "dev",
            "user",
        ]

    def test_realm_access_null(self):
        assert _normalize_roles({"realm_access": None}) == []

    def test_realm_access_missing(self):
        assert _normalize_roles({}) == []

    def test_roles_is_string_rejected(self):
        assert _normalize_roles({"realm_access": {"roles": "admin-user"}}) == []

    def test_roles_null(self):
        assert _normalize_roles({"realm_access": {"roles": None}}) == []

    def test_non_string_items_filtered(self):
        assert _normalize_roles({"realm_access": {"roles": ["dev", 123, "user"]}}) == [
            "dev",
            "user",
        ]


def _groups(*, developer: list[str] | None = None, user: list[str] | None = None):
    mock_settings = patch("deep_agent.aegra.auth_helpers.settings")
    settings = mock_settings.start()
    settings.DEVELOPER_GROUP = developer or []
    settings.USER_GROUP = user or []
    return mock_settings, settings


class TestCheckGroupAccess:
    def test_open_when_both_groups_empty(self):
        ctx, _ = _groups()
        try:
            check_group_access([])
            check_group_access(["some-role"])
            check_group_access(["some-role"], developer_only=True)
        finally:
            ctx.stop()

    def test_only_developer_group_allows_members(self):
        ctx, _ = _groups(developer=["lightspeed-developer"])
        try:
            check_group_access(["lightspeed-developer"])
            check_group_access(["lightspeed-developer"], developer_only=True)
        finally:
            ctx.stop()

    def test_only_developer_group_denies_others(self):
        ctx, _ = _groups(developer=["lightspeed-developer"])
        try:
            with pytest.raises(HTTPException) as exc:
                check_group_access(["anyone"])
            assert exc.value.status_code == 403
            with pytest.raises(HTTPException) as eval_exc:
                check_group_access(["anyone"], developer_only=True)
            assert eval_exc.value.status_code == 403
        finally:
            ctx.stop()

    def test_only_user_group_allows_members_non_eval(self):
        ctx, _ = _groups(user=["lightspeed-user"])
        try:
            check_group_access(["lightspeed-user"])
        finally:
            ctx.stop()

    def test_only_user_group_denies_eval(self):
        ctx, _ = _groups(user=["lightspeed-user"])
        try:
            with pytest.raises(HTTPException) as exc:
                check_group_access(["lightspeed-user"], developer_only=True)
            assert exc.value.status_code == 403
        finally:
            ctx.stop()

    def test_only_user_group_denies_others(self):
        ctx, _ = _groups(user=["lightspeed-user"])
        try:
            with pytest.raises(HTTPException) as exc:
                check_group_access(["anyone"])
            assert exc.value.status_code == 403
        finally:
            ctx.stop()

    def test_developer_in_developer_group_passes(self):
        ctx, _ = _groups(developer=["lightspeed-developer"], user=["lightspeed-user"])
        try:
            check_group_access(["lightspeed-developer"])
        finally:
            ctx.stop()

    def test_user_in_user_group_passes_non_eval(self):
        ctx, _ = _groups(developer=["lightspeed-developer"], user=["lightspeed-user"])
        try:
            check_group_access(["lightspeed-user"])
        finally:
            ctx.stop()

    def test_user_in_neither_group_denied(self):
        ctx, _ = _groups(developer=["lightspeed-developer"], user=["lightspeed-user"])
        try:
            with pytest.raises(HTTPException) as exc:
                check_group_access(["other-role"])
            assert exc.value.status_code == 403
        finally:
            ctx.stop()

    def test_developer_only_blocks_user_group(self):
        ctx, _ = _groups(developer=["lightspeed-developer"], user=["lightspeed-user"])
        try:
            with pytest.raises(HTTPException) as exc:
                check_group_access(["lightspeed-user"], developer_only=True)
            assert exc.value.status_code == 403
        finally:
            ctx.stop()

    def test_developer_only_passes_for_developer(self):
        ctx, _ = _groups(developer=["lightspeed-developer"], user=["lightspeed-user"])
        try:
            check_group_access(["lightspeed-developer"], developer_only=True)
        finally:
            ctx.stop()

    def test_multi_group_any_dev_match_passes(self):
        ctx, _ = _groups(developer=["dev1", "dev2"])
        try:
            check_group_access(["dev2"])
        finally:
            ctx.stop()

    def test_multi_group_any_user_match_passes(self):
        ctx, _ = _groups(user=["user1", "user2"])
        try:
            check_group_access(["user2"])
        finally:
            ctx.stop()


class TestAuthenticatedUserIdGroupCheck:
    @pytest.mark.asyncio
    async def test_group_check_skipped_when_auth_disabled(self):
        request = MagicMock()
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", False),
            patch("deep_agent.aegra.auth_helpers.settings") as mock_settings,
        ):
            mock_settings.DEVELOPER_GROUP = ["devs"]
            mock_settings.USER_GROUP = ["users"]
            result = await authenticated_user_id(request)
        assert result == "dev-user"

    @pytest.mark.asyncio
    async def test_group_check_called_with_token_roles(self):
        request = MagicMock()
        request.headers = {"authorization": "Bearer tok"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": "u1", "realm_access": {"roles": ["devs"]}},
            ),
            patch("deep_agent.aegra.auth_helpers.settings") as mock_settings,
        ):
            mock_settings.DEVELOPER_GROUP = ["devs"]
            mock_settings.USER_GROUP = ["users"]
            result = await authenticated_user_id(request)
        assert result == "u1"

    @pytest.mark.asyncio
    async def test_developer_only_param_raises_403_for_user_group(self):
        request = MagicMock()
        request.headers = {"authorization": "Bearer tok"}
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": "u1", "realm_access": {"roles": ["users"]}},
            ),
            patch("deep_agent.aegra.auth_helpers.settings") as mock_settings,
        ):
            mock_settings.DEVELOPER_GROUP = ["devs"]
            mock_settings.USER_GROUP = ["users"]
            with pytest.raises(HTTPException) as exc:
                await authenticated_user_id(request, developer_only=True)
        assert exc.value.status_code == 403


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
