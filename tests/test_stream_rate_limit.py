"""Tests for per-user rate limiting in the stream route."""

from __future__ import annotations

import asyncio

import pytest

from template_agent.src.routes.stream import (
    _get_user_semaphore,
    _MAX_CONCURRENT_DEEP_RESEARCH_PER_USER,
    _user_semaphores,
)


class TestPerUserSemaphore:
    """Verify per-user semaphore creation and limits."""

    def setup_method(self):
        """Clear module-level state between tests."""
        _user_semaphores.clear()

    def test_creates_semaphore_on_first_access(self):
        """Should create a new semaphore for an unknown user."""
        sem = _get_user_semaphore("user-1")
        assert isinstance(sem, asyncio.Semaphore)

    def test_returns_same_semaphore_for_same_user(self):
        """Should return the same semaphore object for the same user."""
        sem1 = _get_user_semaphore("user-1")
        sem2 = _get_user_semaphore("user-1")
        assert sem1 is sem2

    def test_different_users_get_different_semaphores(self):
        """Should return different semaphore objects for different users."""
        sem1 = _get_user_semaphore("user-1")
        sem2 = _get_user_semaphore("user-2")
        assert sem1 is not sem2

    @pytest.mark.asyncio
    async def test_semaphore_has_correct_limit(self):
        """Should allow up to _MAX_CONCURRENT requests."""
        sem = _get_user_semaphore("user-1")
        acquired = 0
        for _ in range(_MAX_CONCURRENT_DEEP_RESEARCH_PER_USER):
            assert not sem.locked()
            await sem.acquire()
            acquired += 1
        assert sem.locked()
        assert acquired == _MAX_CONCURRENT_DEEP_RESEARCH_PER_USER
        for _ in range(acquired):
            sem.release()
