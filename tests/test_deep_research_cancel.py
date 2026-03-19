"""Tests for the deep research cancel module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import template_agent.src.core.deep_research.cancel as cancel_mod
from template_agent.src.core.deep_research.cancel import (
    CancelStore,
    _CANCEL_NAMESPACE,
    get_cancel_store,
)


@pytest.fixture(autouse=True)
def reset_cancel_store():
    """Reset the singleton _cancel_store before and after each test."""
    cancel_mod._cancel_store = None
    yield
    cancel_mod._cancel_store = None


# -----------------------------------------------------------------------
# Original L1-only tests (no backing store configured)
# -----------------------------------------------------------------------


class TestCancelStore:
    """Test cases for CancelStore (L1-only, backward-compatible)."""

    @pytest.mark.asyncio
    async def test_request_cancel_marks_thread(self):
        """Test that request_cancel adds thread_id to cancelled set."""
        store = CancelStore()
        thread_id = "thread-123"

        await store.request_cancel(thread_id)

        assert await store.is_cancelled(thread_id) is True

    @pytest.mark.asyncio
    async def test_is_cancelled_returns_false_for_unknown(self):
        """Test that is_cancelled returns False for unknown thread."""
        store = CancelStore()

        result = await store.is_cancelled("unknown-thread")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_cancelled_returns_true_after_request(self):
        """Test that is_cancelled returns True after request_cancel."""
        store = CancelStore()
        thread_id = "thread-456"

        await store.request_cancel(thread_id)
        result = await store.is_cancelled(thread_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_clear_removes_cancelled_thread(self):
        """Test that clear removes thread from cancelled set."""
        store = CancelStore()
        thread_id = "thread-789"

        await store.request_cancel(thread_id)
        await store.clear(thread_id)

        assert await store.is_cancelled(thread_id) is False

    @pytest.mark.asyncio
    async def test_clear_idempotent_for_unknown_thread(self):
        """Test that clear on unknown thread does not raise."""
        store = CancelStore()

        await store.clear("never-cancelled-thread")

        assert await store.is_cancelled("never-cancelled-thread") is False

    @pytest.mark.asyncio
    async def test_multiple_threads_independent(self):
        """Test that cancelling one thread does not affect others."""
        store = CancelStore()
        thread_a = "thread-a"
        thread_b = "thread-b"

        await store.request_cancel(thread_a)

        assert await store.is_cancelled(thread_a) is True
        assert await store.is_cancelled(thread_b) is False


class TestGetCancelStore:
    """Test cases for get_cancel_store singleton."""

    def test_get_cancel_store_returns_singleton(self):
        """Test that get_cancel_store returns the same instance on repeated calls."""
        store1 = get_cancel_store()
        store2 = get_cancel_store()

        assert store1 is store2

    def test_get_cancel_store_creates_instance_on_first_call(self):
        """Test that get_cancel_store creates CancelStore on first call."""
        store = get_cancel_store()

        assert isinstance(store, CancelStore)


# -----------------------------------------------------------------------
# Helpers for building a mock BaseStore
# -----------------------------------------------------------------------


def _make_mock_store() -> MagicMock:
    """Return a mock that mimics ``BaseStore`` async methods."""
    mock = MagicMock()
    mock.aput = AsyncMock()
    mock.aget = AsyncMock(return_value=None)
    mock.adelete = AsyncMock()
    return mock


def _make_item(value: dict[str, Any]) -> MagicMock:
    """Return a mock ``Item`` with the given value dict."""
    item = MagicMock()
    item.value = value
    return item


# -----------------------------------------------------------------------
# Two-tier (L1 + L2) tests
# -----------------------------------------------------------------------


class TestCancelStoreWithBackingStore:
    """Tests for the L2 backing-store integration."""

    @pytest.mark.asyncio
    async def test_configure_backing_store_sets_store(self):
        """configure_backing_store attaches the store reference."""
        cs = CancelStore()
        mock_store = _make_mock_store()

        cs.configure_backing_store(mock_store)

        assert cs._store is mock_store

    @pytest.mark.asyncio
    async def test_configure_backing_store_none_reverts_to_l1(self):
        """Passing None removes the backing store."""
        cs = CancelStore()
        cs.configure_backing_store(_make_mock_store())
        cs.configure_backing_store(None)

        assert cs._store is None

    @pytest.mark.asyncio
    async def test_request_cancel_writes_to_both_tiers(self):
        """request_cancel writes to L1 (in-memory) and L2 (store)."""
        cs = CancelStore()
        mock_store = _make_mock_store()
        cs.configure_backing_store(mock_store)

        await cs.request_cancel("t1")

        assert "t1" in cs._cancelled
        mock_store.aput.assert_awaited_once_with(
            _CANCEL_NAMESPACE, "t1", {"cancelled": True}, index=False
        )

    @pytest.mark.asyncio
    async def test_is_cancelled_l1_hit_skips_store(self):
        """When L1 has the thread, no L2 lookup is performed."""
        cs = CancelStore()
        mock_store = _make_mock_store()
        cs.configure_backing_store(mock_store)
        cs._cancelled.add("t1")

        result = await cs.is_cancelled("t1")

        assert result is True
        mock_store.aget.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_is_cancelled_l1_miss_l2_hit_backfills(self):
        """On L1 miss, L2 is checked; a hit back-fills L1."""
        cs = CancelStore()
        mock_store = _make_mock_store()
        mock_store.aget = AsyncMock(return_value=_make_item({"cancelled": True}))
        cs.configure_backing_store(mock_store)

        result = await cs.is_cancelled("t2")

        assert result is True
        assert "t2" in cs._cancelled
        mock_store.aget.assert_awaited_once_with(_CANCEL_NAMESPACE, "t2")

    @pytest.mark.asyncio
    async def test_is_cancelled_l1_miss_l2_miss(self):
        """On L1 miss and L2 miss, returns False without back-filling."""
        cs = CancelStore()
        mock_store = _make_mock_store()
        mock_store.aget = AsyncMock(return_value=None)
        cs.configure_backing_store(mock_store)

        result = await cs.is_cancelled("t3")

        assert result is False
        assert "t3" not in cs._cancelled

    @pytest.mark.asyncio
    async def test_clear_deletes_from_both_tiers(self):
        """clear() removes from L1 and calls adelete on L2."""
        cs = CancelStore()
        mock_store = _make_mock_store()
        cs.configure_backing_store(mock_store)
        cs._cancelled.add("t1")

        await cs.clear("t1")

        assert "t1" not in cs._cancelled
        mock_store.adelete.assert_awaited_once_with(_CANCEL_NAMESPACE, "t1")


# -----------------------------------------------------------------------
# Graceful degradation: store failures must not break cancellation
# -----------------------------------------------------------------------


class TestCancelStoreGracefulDegradation:
    """Store exceptions must be caught -- L1 always works."""

    @pytest.mark.asyncio
    async def test_request_cancel_survives_store_put_failure(self):
        """L1 is updated even when L2 aput raises."""
        cs = CancelStore()
        mock_store = _make_mock_store()
        mock_store.aput = AsyncMock(side_effect=RuntimeError("db down"))
        cs.configure_backing_store(mock_store)

        await cs.request_cancel("t1")

        assert await cs.is_cancelled("t1") is True

    @pytest.mark.asyncio
    async def test_is_cancelled_survives_store_get_failure(self):
        """L2 failure returns False (no crash)."""
        cs = CancelStore()
        mock_store = _make_mock_store()
        mock_store.aget = AsyncMock(side_effect=RuntimeError("db down"))
        cs.configure_backing_store(mock_store)

        result = await cs.is_cancelled("t1")

        assert result is False

    @pytest.mark.asyncio
    async def test_clear_survives_store_delete_failure(self):
        """L1 is cleared even when L2 adelete raises."""
        cs = CancelStore()
        mock_store = _make_mock_store()
        mock_store.adelete = AsyncMock(side_effect=RuntimeError("db down"))
        cs.configure_backing_store(mock_store)
        cs._cancelled.add("t1")

        await cs.clear("t1")

        assert "t1" not in cs._cancelled

    @pytest.mark.asyncio
    async def test_no_store_configured_is_pure_l1(self):
        """Without a backing store the behaviour is identical to the old code."""
        cs = CancelStore()

        await cs.request_cancel("t1")
        assert await cs.is_cancelled("t1") is True

        await cs.clear("t1")
        assert await cs.is_cancelled("t1") is False
