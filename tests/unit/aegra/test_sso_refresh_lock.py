"""Unit tests for SSO token refresh locking in mcp.py."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from deep_agent.aegra.mcp import _locked_sso_refresh


@asynccontextmanager
async def _timeout_lock(*_args, **_kwargs):
    yield "timeout"


@asynccontextmanager
async def _held_lock(*_args, **_kwargs):
    yield "held"


@asynccontextmanager
async def _no_redis_lock(*_args, **_kwargs):
    yield "no_redis"


def _fake_jwt_exp_future(token: str) -> float:
    return time.time() + 3600


def _fake_jwt_exp_soon(token: str) -> float:
    return time.time() + 5


@pytest.mark.asyncio
class TestLockedSsoRefresh:
    async def test_timeout_peer_already_refreshed(self):
        with (
            patch("deep_agent.aegra.redis.distributed_lock", _timeout_lock),
            patch("deep_agent.aegra.mcp._current_access_token") as mock_cv,
            patch("deep_agent.aegra.mcp._jwt_exp", side_effect=_fake_jwt_exp_future),
        ):
            mock_cv.get.return_value = "fresh-token"

            result = await _locked_sso_refresh(
                "old-token", "refresh-tok", 10.0, user_id="user-1"
            )

        assert result == "fresh-token"

    async def test_timeout_no_peer_refresh(self):
        with (
            patch("deep_agent.aegra.redis.distributed_lock", _timeout_lock),
            patch("deep_agent.aegra.mcp._current_access_token") as mock_cv,
        ):
            mock_cv.get.return_value = None

            result = await _locked_sso_refresh(
                "old-token", "refresh-tok", 10.0, user_id="user-1"
            )

        assert result == "old-token"

    async def test_timeout_peer_token_expiring_soon(self):
        with (
            patch("deep_agent.aegra.redis.distributed_lock", _timeout_lock),
            patch("deep_agent.aegra.mcp._current_access_token") as mock_cv,
            patch("deep_agent.aegra.mcp._jwt_exp", side_effect=_fake_jwt_exp_soon),
        ):
            mock_cv.get.return_value = "almost-expired"

            result = await _locked_sso_refresh(
                "old-token", "refresh-tok", 10.0, user_id="user-1"
            )

        assert result == "old-token"

    async def test_held_peer_already_refreshed(self):
        mock_do_refresh = AsyncMock(return_value="should-not-be-called")

        with (
            patch("deep_agent.aegra.redis.distributed_lock", _held_lock),
            patch("deep_agent.aegra.mcp._current_access_token") as mock_cv,
            patch("deep_agent.aegra.mcp._jwt_exp", side_effect=_fake_jwt_exp_future),
            patch("deep_agent.aegra.mcp._do_sso_refresh", mock_do_refresh),
        ):
            mock_cv.get.return_value = "fresh-token"

            result = await _locked_sso_refresh(
                "old-token", "refresh-tok", 10.0, user_id="user-1"
            )

        assert result == "fresh-token"
        mock_do_refresh.assert_not_called()

    async def test_held_no_peer_refresh_calls_do_sso_refresh(self):
        mock_do_refresh = AsyncMock(return_value="new-token")

        with (
            patch("deep_agent.aegra.redis.distributed_lock", _held_lock),
            patch("deep_agent.aegra.mcp._current_access_token") as mock_cv,
            patch("deep_agent.aegra.mcp._current_refresh_token") as mock_rt,
            patch("deep_agent.aegra.mcp._do_sso_refresh", mock_do_refresh),
        ):
            mock_cv.get.return_value = None
            mock_rt.get.return_value = "rotated-refresh"

            result = await _locked_sso_refresh(
                "old-token", "refresh-tok", 10.0, user_id="user-1"
            )

        assert result == "new-token"
        mock_do_refresh.assert_awaited_once_with("old-token", "rotated-refresh", 10.0)

    async def test_held_uses_latest_refresh_token(self):
        mock_do_refresh = AsyncMock(return_value="new-token")

        with (
            patch("deep_agent.aegra.redis.distributed_lock", _held_lock),
            patch("deep_agent.aegra.mcp._current_access_token") as mock_cv,
            patch("deep_agent.aegra.mcp._current_refresh_token") as mock_rt,
            patch("deep_agent.aegra.mcp._do_sso_refresh", mock_do_refresh),
        ):
            mock_cv.get.return_value = None
            mock_rt.get.return_value = None

            result = await _locked_sso_refresh(
                "old-token", "original-refresh", 10.0, user_id="user-1"
            )

        assert result == "new-token"
        mock_do_refresh.assert_awaited_once_with("old-token", "original-refresh", 10.0)

    async def test_no_redis_peer_already_refreshed(self):
        mock_do_refresh = AsyncMock(return_value="should-not-be-called")

        with (
            patch("deep_agent.aegra.redis.distributed_lock", _no_redis_lock),
            patch("deep_agent.aegra.mcp._current_access_token") as mock_cv,
            patch("deep_agent.aegra.mcp._jwt_exp", side_effect=_fake_jwt_exp_future),
            patch("deep_agent.aegra.mcp._do_sso_refresh", mock_do_refresh),
        ):
            mock_cv.get.return_value = "fresh-token"

            result = await _locked_sso_refresh(
                "old-token", "refresh-tok", 10.0, user_id="user-1"
            )

        assert result == "fresh-token"
        mock_do_refresh.assert_not_called()

    async def test_no_redis_uses_latest_refresh_token(self):
        mock_do_refresh = AsyncMock(return_value="new-token")

        with (
            patch("deep_agent.aegra.redis.distributed_lock", _no_redis_lock),
            patch("deep_agent.aegra.mcp._current_access_token") as mock_cv,
            patch("deep_agent.aegra.mcp._current_refresh_token") as mock_rt,
            patch("deep_agent.aegra.mcp._do_sso_refresh", mock_do_refresh),
        ):
            mock_cv.get.return_value = None
            mock_rt.get.return_value = "rotated-refresh"

            result = await _locked_sso_refresh(
                "old-token", "original-refresh", 10.0, user_id="user-1"
            )

        assert result == "new-token"
        mock_do_refresh.assert_awaited_once_with("old-token", "rotated-refresh", 10.0)

    async def test_user_id_used_in_lock_name(self):
        lock_names = []

        @asynccontextmanager
        async def _capture_lock(name, **_kwargs):
            lock_names.append(name)
            yield "held"

        mock_do_refresh = AsyncMock(return_value="new-token")

        with (
            patch("deep_agent.aegra.redis.distributed_lock", _capture_lock),
            patch("deep_agent.aegra.mcp._current_access_token") as mock_cv,
            patch("deep_agent.aegra.mcp._current_refresh_token") as mock_rt,
            patch("deep_agent.aegra.mcp._do_sso_refresh", mock_do_refresh),
        ):
            mock_cv.get.return_value = None
            mock_rt.get.return_value = None

            await _locked_sso_refresh("old-token", "refresh-tok", 10.0, user_id="alice")

        assert lock_names == ["sso:refresh:alice"]
