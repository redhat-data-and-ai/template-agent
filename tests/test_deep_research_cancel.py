"""Tests for the deep research cancel module."""

import pytest

import template_agent.src.core.deep_research.cancel as cancel_mod
from template_agent.src.core.deep_research.cancel import (
    CancelStore,
    get_cancel_store,
)


@pytest.fixture(autouse=True)
def reset_cancel_store():
    """Reset the singleton _cancel_store before and after each test."""
    cancel_mod._cancel_store = None
    yield
    cancel_mod._cancel_store = None


class TestCancelStore:
    """Test cases for CancelStore."""

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
