"""Tests for deep research plan route endpoints (IDOR + error handling)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from template_agent.src.routes.deep_research_plan import (
    _verify_thread_ownership,
    delete_plan,
    get_plan,
    save_plan,
)
from template_agent.src.schema import DeepResearchPlanRequest


@pytest.mark.asyncio
async def test_verify_thread_ownership_no_user_id():
    """Should raise 403 when user_id is None."""
    with pytest.raises(HTTPException) as exc_info:
        await _verify_thread_ownership("thread-1", None)
    assert exc_info.value.status_code == 403
    assert "user_id is required" in exc_info.value.detail


_OWNER_PATCH = "template_agent.src.core.deep_research.plan_store.get_thread_owner"


@pytest.mark.asyncio
async def test_verify_thread_ownership_matching_owner():
    """Should pass when user_id matches the registered owner."""
    with patch(
        _OWNER_PATCH,
        new_callable=AsyncMock,
        return_value="user-a",
    ):
        await _verify_thread_ownership("thread-1", "user-a")


@pytest.mark.asyncio
async def test_verify_thread_ownership_no_owner_registered():
    """Should pass when no owner is registered yet (first access)."""
    with patch(
        _OWNER_PATCH,
        new_callable=AsyncMock,
        return_value=None,
    ):
        await _verify_thread_ownership("thread-1", "user-a")


@pytest.mark.asyncio
async def test_verify_thread_ownership_mismatch():
    """Should raise 403 when user_id does not match the registered owner."""
    with patch(
        _OWNER_PATCH,
        new_callable=AsyncMock,
        return_value="user-a",
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _verify_thread_ownership("thread-1", "user-b")
        assert exc_info.value.status_code == 403
        assert "Not authorized" in exc_info.value.detail


@pytest.mark.asyncio
async def test_save_plan_success():
    """Should save plan when ownership is valid."""
    req = DeepResearchPlanRequest(thread_id="t1", plan=["q1", "q2"], user_id="user-a")
    with (
        patch(
            "template_agent.src.routes.deep_research_plan._verify_thread_ownership",
            new_callable=AsyncMock,
        ),
        patch(
            "template_agent.src.core.deep_research.plan_store.set_plan",
            new_callable=AsyncMock,
        ),
        patch(
            "template_agent.src.core.deep_research.plan_store.register_thread_owner",
            new_callable=AsyncMock,
        ),
    ):
        resp = await save_plan(req)
        assert resp.thread_id == "t1"
        assert resp.plan_count == 2


@pytest.mark.asyncio
async def test_save_plan_idor_blocked():
    """Should raise 403 when ownership check fails."""
    req = DeepResearchPlanRequest(thread_id="t1", plan=["q1"], user_id="user-b")
    with patch(
        "template_agent.src.routes.deep_research_plan._verify_thread_ownership",
        new_callable=AsyncMock,
        side_effect=HTTPException(status_code=403, detail="Not authorized"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await save_plan(req)
        assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_save_plan_generic_error_message():
    """Should return generic error, not leak internal details."""
    req = DeepResearchPlanRequest(thread_id="t1", plan=["q1"], user_id="user-a")
    with (
        patch(
            "template_agent.src.routes.deep_research_plan._verify_thread_ownership",
            new_callable=AsyncMock,
        ),
        patch(
            "template_agent.src.core.deep_research.plan_store.set_plan",
            new_callable=AsyncMock,
            side_effect=RuntimeError("DB connection refused on port 5432"),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await save_plan(req)
        assert exc_info.value.status_code == 500
        assert "internal error" in exc_info.value.detail.lower()
        assert "5432" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_plan_success():
    """Should return plan when ownership is valid."""
    with (
        patch(
            "template_agent.src.routes.deep_research_plan._verify_thread_ownership",
            new_callable=AsyncMock,
        ),
        patch(
            "template_agent.src.core.deep_research.plan_store.get_plan",
            new_callable=AsyncMock,
            return_value=["q1", "q2"],
        ),
    ):
        result = await get_plan("t1", user_id="user-a")
        assert result["plan"] == ["q1", "q2"]
        assert result["plan_count"] == 2


@pytest.mark.asyncio
async def test_get_plan_not_found():
    """Should return 404 when plan is missing."""
    with (
        patch(
            "template_agent.src.routes.deep_research_plan._verify_thread_ownership",
            new_callable=AsyncMock,
        ),
        patch(
            "template_agent.src.core.deep_research.plan_store.get_plan",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await get_plan("t1", user_id="user-a")
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_plan_generic_error():
    """Should return generic error on unexpected exception."""
    with (
        patch(
            "template_agent.src.routes.deep_research_plan._verify_thread_ownership",
            new_callable=AsyncMock,
        ),
        patch(
            "template_agent.src.core.deep_research.plan_store.get_plan",
            new_callable=AsyncMock,
            side_effect=RuntimeError("secret leak"),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await get_plan("t1", user_id="user-a")
        assert exc_info.value.status_code == 500
        assert "secret" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_delete_plan_success():
    """Should delete plan when ownership is valid."""
    with (
        patch(
            "template_agent.src.routes.deep_research_plan._verify_thread_ownership",
            new_callable=AsyncMock,
        ),
        patch(
            "template_agent.src.core.deep_research.plan_store.clear_plan",
            new_callable=AsyncMock,
        ),
    ):
        result = await delete_plan("t1", user_id="user-a")
        assert result["status"] == "success"


@pytest.mark.asyncio
async def test_delete_plan_generic_error():
    """Should return generic error on unexpected exception."""
    with (
        patch(
            "template_agent.src.routes.deep_research_plan._verify_thread_ownership",
            new_callable=AsyncMock,
        ),
        patch(
            "template_agent.src.core.deep_research.plan_store.clear_plan",
            new_callable=AsyncMock,
            side_effect=RuntimeError("disk full"),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await delete_plan("t1", user_id="user-a")
        assert exc_info.value.status_code == 500
        assert "disk" not in exc_info.value.detail
