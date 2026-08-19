"""Unit tests for shared authentication helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from deep_agent.aegra.auth_helpers import authenticated_user_id


class TestAuthenticatedUserId:
    @pytest.mark.asyncio
    async def test_returns_dev_user_when_auth_disabled(self):
        request = MagicMock()
        with patch("deep_agent.aegra.auth.ENABLE_AUTH", False):
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
