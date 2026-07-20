"""Unit tests for HeadlessRuntime and HeadlessUser."""

from __future__ import annotations

from deep_agent.src.triggers.runtime import HeadlessRuntime, HeadlessUser


class TestHeadlessUser:
    """Test HeadlessUser data attributes."""

    def test_default_identity(self):
        user = HeadlessUser()
        assert user.identity == "headless-worker"

    def test_custom_identity(self):
        user = HeadlessUser(identity="batch-processor-42")
        assert user.identity == "batch-processor-42"

    def test_access_token_is_none(self):
        user = HeadlessUser()
        assert user.access_token is None

    def test_refresh_token_is_none(self):
        user = HeadlessUser()
        assert user.refresh_token is None


class TestHeadlessRuntime:
    """Test HeadlessRuntime creates a user with expected attributes."""

    def test_creates_user_with_default_identity(self):
        runtime = HeadlessRuntime()
        assert runtime.user.identity == "headless-worker"

    def test_creates_user_with_custom_identity(self):
        runtime = HeadlessRuntime(identity="nightly-report-agent")
        assert runtime.user.identity == "nightly-report-agent"

    def test_user_has_expected_attributes(self):
        runtime = HeadlessRuntime(identity="test-agent")
        user = runtime.user

        assert user.identity == "test-agent"
        assert user.access_token is None
        assert user.refresh_token is None
