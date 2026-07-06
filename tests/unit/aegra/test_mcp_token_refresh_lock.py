"""Unit tests for MCP token refresh locking."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from deep_agent.aegra.mcp_auth import McpCredentialResolver
from deep_agent.aegra.mcp_token_store import McpOAuthToken


@asynccontextmanager
async def _held_lock(*_args, **_kwargs):
    yield "held"


@asynccontextmanager
async def _timeout_lock(*_args, **_kwargs):
    yield "timeout"


def _expired_token() -> McpOAuthToken:
    return McpOAuthToken(
        user_id="user-1",
        mcp_name="oauth-mcp",
        access_token="expired-access",
        refresh_token="refresh-me",
        expires_at=datetime.now(UTC) - timedelta(minutes=5),
    )


def _fresh_token() -> McpOAuthToken:
    return McpOAuthToken(
        user_id="user-1",
        mcp_name="oauth-mcp",
        access_token="fresh-access",
        refresh_token="refresh-me",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )


@pytest.mark.asyncio
class TestMcpTokenRefreshLock:
    async def test_skips_refresh_when_peer_refreshed_under_lock(self):
        store = AsyncMock()
        store.get_token = AsyncMock(side_effect=[_expired_token(), _fresh_token()])
        resolver = McpCredentialResolver(token_store=store)

        with (
            patch("deep_agent.aegra.mcp_auth.distributed_lock", _held_lock),
            patch.object(
                resolver,
                "_refresh_mcp_token",
                new=AsyncMock(return_value="should-not-run"),
            ) as refresh,
        ):
            token = await resolver.resolve(
                "user-1",
                "oauth-mcp",
                {
                    "auth_mode": "oauth",
                    "oauth": {"token_endpoint": "https://as.example.com/token"},
                },
            )

        assert token == "fresh-access"
        refresh.assert_not_called()
        assert store.get_token.await_count == 2

    async def test_waits_for_peer_refresh_on_lock_timeout(self):
        store = AsyncMock()
        store.get_token = AsyncMock(
            side_effect=[
                _expired_token(),
                _expired_token(),
                _fresh_token(),
            ]
        )
        resolver = McpCredentialResolver(token_store=store)

        with (
            patch("deep_agent.aegra.mcp_auth.distributed_lock", _timeout_lock),
            patch(
                "deep_agent.aegra.mcp_auth.asyncio.sleep",
                new=AsyncMock(),
            ),
            patch.object(
                resolver,
                "_refresh_mcp_token",
                new=AsyncMock(return_value="should-not-run"),
            ) as refresh,
        ):
            token = await resolver.resolve(
                "user-1",
                "oauth-mcp",
                {
                    "auth_mode": "oauth",
                    "oauth": {"token_endpoint": "https://as.example.com/token"},
                },
            )

        assert token == "fresh-access"
        refresh.assert_not_called()

    async def test_refreshes_once_when_lock_held(self):
        store = AsyncMock()
        store.get_token = AsyncMock(side_effect=[_expired_token(), _expired_token()])
        resolver = McpCredentialResolver(token_store=store)

        with (
            patch("deep_agent.aegra.mcp_auth.distributed_lock", _held_lock),
            patch.object(
                resolver,
                "_refresh_mcp_token",
                new=AsyncMock(return_value="new-access"),
            ) as refresh,
        ):
            token = await resolver.resolve(
                "user-1",
                "oauth-mcp",
                {
                    "auth_mode": "oauth",
                    "oauth": {
                        "token_endpoint": "https://as.example.com/token",
                        "client_id": "cid",
                    },
                },
            )

        assert token == "new-access"
        refresh.assert_awaited_once()
