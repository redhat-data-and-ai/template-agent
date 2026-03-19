"""Tests for cancel endpoint IDOR protection."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from template_agent.src.api import cancel_deep_research

_OWNER_PATCH = "template_agent.src.core.deep_research.plan_store.get_thread_owner"


@pytest.mark.asyncio
async def test_cancel_requires_user_id():
    """Should raise 403 when user_id is not provided."""
    with pytest.raises(HTTPException) as exc_info:
        await cancel_deep_research("thread-1", user_id=None)
    assert exc_info.value.status_code == 403
    assert "user_id is required" in exc_info.value.detail


@pytest.mark.asyncio
async def test_cancel_owner_mismatch():
    """Should raise 403 when user is not the thread owner."""
    with patch(
        _OWNER_PATCH,
        new_callable=AsyncMock,
        return_value="owner-a",
    ):
        with pytest.raises(HTTPException) as exc_info:
            await cancel_deep_research("thread-1", user_id="attacker")
        assert exc_info.value.status_code == 403
        assert "Not authorized" in exc_info.value.detail


@pytest.mark.asyncio
async def test_cancel_success_matching_owner():
    """Should succeed when user_id matches the thread owner."""
    with (
        patch(
            _OWNER_PATCH,
            new_callable=AsyncMock,
            return_value="owner-a",
        ),
        patch(
            "template_agent.src.api.get_cancel_store",
        ) as mock_store_fn,
    ):
        mock_store = AsyncMock()
        mock_store_fn.return_value = mock_store
        result = await cancel_deep_research("thread-1", user_id="owner-a")
        assert result["status"] == "cancelled"
        mock_store.request_cancel.assert_awaited_once_with("thread-1")


@pytest.mark.asyncio
async def test_cancel_success_no_owner_registered():
    """Should succeed when no owner is registered (new thread)."""
    with (
        patch(
            _OWNER_PATCH,
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "template_agent.src.api.get_cancel_store",
        ) as mock_store_fn,
    ):
        mock_store = AsyncMock()
        mock_store_fn.return_value = mock_store
        result = await cancel_deep_research("thread-1", user_id="any-user")
        assert result["status"] == "cancelled"
