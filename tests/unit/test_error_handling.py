"""Unit tests for error_handling module.

Tests cover:
- classify_error: all 4 classification branches
- with_fallback: sync, async, selective exception catching
- CircuitBreaker (in-memory): full closed→open→half-open→closed lifecycle
- CircuitBreaker (Redis-backed): mocked Redis hash operations
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.error_handling import (
    CircuitBreaker,
    classify_error,
    create_circuit_breaker,
    with_fallback,
)
from deep_agent.src.exceptions import (
    AuthenticationError,
    ConfigurationError,
    LLMError,
    MCPError,
    RateLimitError,
    SubAgentError,
)

# ───────────────────────────────────────────────────────────────────
# classify_error
# ───────────────────────────────────────────────────────────────────


class TestClassifyError:
    """Tests for classify_error — 4 branches."""

    def test_rate_limit_error(self):
        result = classify_error(RateLimitError("quota exceeded"))
        assert result["recoverable"] is True
        assert result["error_type"] == "rate_limit"
        assert "rate limit" in result["message"].lower()

    def test_transient_error(self):
        result = classify_error(LLMError("model unavailable"))
        assert result["recoverable"] is True
        assert result["error_type"] == "transient"
        assert "unavailable" in result["message"].lower()

    def test_transient_mcp_error(self):
        result = classify_error(MCPError("connection refused"))
        assert result["recoverable"] is True
        assert result["error_type"] == "transient"

    def test_app_exception_non_transient(self):
        result = classify_error(SubAgentError("build failed"))
        assert result["recoverable"] is False
        assert result["error_type"] == "E_006"

    def test_app_exception_config_error(self):
        result = classify_error(ConfigurationError("missing key"))
        assert result["recoverable"] is False
        assert result["message"] == "Configuration Initialization Failed"

    def test_app_exception_auth_error(self):
        result = classify_error(AuthenticationError("bad token"))
        assert result["recoverable"] is False
        assert result["error_type"] == "E_010"

    def test_unknown_exception(self):
        result = classify_error(RuntimeError("something unexpected"))
        assert result["recoverable"] is False
        assert result["error_type"] == "unknown"
        assert result["message"] == "Internal server error"

    def test_base_exception_treated_as_unknown(self):
        result = classify_error(TypeError("bad type"))
        assert result["error_type"] == "unknown"

    def test_rate_limit_before_transient(self):
        """RateLimitError IS a TransientError, but classify_error checks it first."""
        result = classify_error(RateLimitError("429"))
        assert result["error_type"] == "rate_limit"
        assert result["recoverable"] is True


# ───────────────────────────────────────────────────────────────────
# with_fallback
# ───────────────────────────────────────────────────────────────────


class TestWithFallback:
    """Tests for with_fallback decorator."""

    def test_sync_returns_normal_result(self):
        @with_fallback("default")
        def good() -> str:
            return "real"

        assert good() == "real"

    def test_sync_returns_fallback_on_exception(self):
        @with_fallback("default")
        def bad() -> str:
            raise ValueError("boom")

        assert bad() == "default"

    def test_sync_selective_catch(self):
        """Only catches specified exception types."""

        @with_fallback("default", on=(ValueError,))
        def bad() -> str:
            raise TypeError("wrong type")

        with pytest.raises(TypeError, match="wrong type"):
            bad()

    def test_sync_selective_catch_matches(self):
        @with_fallback("default", on=(ValueError,))
        def bad() -> str:
            raise ValueError("expected")

        assert bad() == "default"

    def test_async_returns_normal_result(self):
        @with_fallback("default")
        async def good() -> str:
            return "real"

        assert asyncio.run(good()) == "real"

    def test_async_returns_fallback_on_exception(self):
        @with_fallback("default")
        async def bad() -> str:
            raise RuntimeError("async boom")

        assert asyncio.run(bad()) == "default"

    def test_async_selective_catch(self):
        @with_fallback("default", on=(ValueError,))
        async def bad() -> str:
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            asyncio.run(bad())

    def test_fallback_with_none_value(self):
        @with_fallback(None)
        def bad() -> str | None:
            raise ValueError("boom")

        assert bad() is None

    def test_fallback_with_list_value(self):
        @with_fallback([])
        def bad() -> list[str]:
            raise ValueError("boom")

        assert bad() == []

    def test_preserves_function_name(self):
        @with_fallback("x")
        def my_function() -> str:
            return "y"

        assert my_function.__name__ == "my_function"

    def test_async_preserves_function_name(self):
        @with_fallback("x")
        async def my_async_fn() -> str:
            return "y"

        assert my_async_fn.__name__ == "my_async_fn"


# ───────────────────────────────────────────────────────────────────
# CircuitBreaker — in-memory
# ───────────────────────────────────────────────────────────────────


class TestCircuitBreakerInMemory:
    """Tests for CircuitBreaker with in-memory backend (no Redis)."""

    def test_starts_closed(self):
        cb = CircuitBreaker("test", threshold=3, reset_timeout=10.0)
        assert cb.state == "closed"
        assert cb.is_open is False

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("test", threshold=3, reset_timeout=10.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"
        assert cb.is_open is False

    def test_opens_at_threshold(self):
        cb = CircuitBreaker("test", threshold=3, reset_timeout=10.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        assert cb.is_open is True

    def test_success_resets_failures(self):
        cb = CircuitBreaker("test", threshold=3, reset_timeout=10.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "closed"

    def test_success_closes_open_circuit(self):
        cb = CircuitBreaker("test", threshold=2, reset_timeout=10.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        cb.record_success()
        assert cb.state == "closed"

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker("test", threshold=2, reset_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True

        time.sleep(0.06)
        assert cb.is_open is False
        assert cb.state == "half-open"

    def test_full_lifecycle(self):
        """closed → open → half-open → closed (after success)."""
        cb = CircuitBreaker("test", threshold=2, reset_timeout=0.05)

        assert cb.state == "closed"

        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        assert cb.is_open is True

        time.sleep(0.06)
        assert cb.is_open is False
        assert cb.state == "half-open"

        cb.record_success()
        assert cb.state == "closed"
        assert cb.is_open is False

    def test_half_open_reopens_on_failure(self):
        """half-open → open if probe fails."""
        cb = CircuitBreaker("test", threshold=2, reset_timeout=0.05)
        cb.record_failure()
        cb.record_failure()

        time.sleep(0.06)
        assert cb.state == "half-open"

        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

    def test_threshold_one(self):
        cb = CircuitBreaker("test", threshold=1, reset_timeout=10.0)
        cb.record_failure()
        assert cb.state == "open"

    def test_default_parameters(self):
        cb = CircuitBreaker("defaults")
        assert cb.threshold == 5
        assert cb.reset_timeout == 60.0
        assert cb.name == "defaults"


# ───────────────────────────────────────────────────────────────────
# CircuitBreaker — Redis-backed (mocked)
# ───────────────────────────────────────────────────────────────────


class TestCircuitBreakerRedis:
    """Tests for CircuitBreaker with Redis backend (mocked)."""

    def _make_redis_mock(self) -> MagicMock:
        """Create a mock Redis client that behaves like a real hash store."""
        store: dict[str, dict[str, str]] = {}

        mock = MagicMock()

        def hgetall(key: str) -> dict[str, str]:
            return store.get(key, {})

        def hset(key: str, mapping: dict[str, str]) -> int:
            if key not in store:
                store[key] = {}
            store[key].update({k: str(v) for k, v in mapping.items()})
            return len(mapping)

        def delete(key: str) -> int:
            return 1 if store.pop(key, None) is not None else 0

        mock.hgetall = MagicMock(side_effect=hgetall)
        mock.hset = MagicMock(side_effect=hset)
        mock.delete = MagicMock(side_effect=delete)
        mock.expire = MagicMock(return_value=True)
        mock.ping = MagicMock(return_value=True)
        mock._store = store
        return mock

    def test_starts_closed(self):
        mock_redis = self._make_redis_mock()
        cb = CircuitBreaker("test", threshold=3, redis_client=mock_redis)
        assert cb.state == "closed"
        assert cb.is_open is False

    def test_opens_at_threshold(self):
        mock_redis = self._make_redis_mock()
        cb = CircuitBreaker("test", threshold=3, redis_client=mock_redis)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

    def test_success_resets(self):
        mock_redis = self._make_redis_mock()
        cb = CircuitBreaker("test", threshold=3, redis_client=mock_redis)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == "closed"

    def test_redis_state_shared_across_instances(self):
        """Two CircuitBreaker instances sharing the same Redis see the same state."""
        mock_redis = self._make_redis_mock()
        cb1 = CircuitBreaker("shared", threshold=2, redis_client=mock_redis)
        cb2 = CircuitBreaker("shared", threshold=2, redis_client=mock_redis)

        cb1.record_failure()
        cb1.record_failure()
        assert cb2.state == "open"

    def test_redis_half_open_after_timeout(self):
        mock_redis = self._make_redis_mock()
        cb = CircuitBreaker(
            "test", threshold=2, reset_timeout=0.05, redis_client=mock_redis
        )
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True

        time.sleep(0.06)
        assert cb.is_open is False
        assert cb.state == "half-open"

    def test_redis_full_lifecycle(self):
        mock_redis = self._make_redis_mock()
        cb = CircuitBreaker(
            "test", threshold=2, reset_timeout=0.05, redis_client=mock_redis
        )

        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        time.sleep(0.06)
        assert cb.state == "half-open"

        cb.record_success()
        assert cb.state == "closed"

    def test_redis_error_falls_back_to_closed(self):
        """If Redis raises, circuit breaker should degrade to closed (allow requests)."""
        mock_redis = MagicMock()
        mock_redis.hgetall = MagicMock(side_effect=ConnectionError("Redis down"))
        mock_redis.hset = MagicMock(side_effect=ConnectionError("Redis down"))
        mock_redis.delete = MagicMock(side_effect=ConnectionError("Redis down"))

        cb = CircuitBreaker("test", threshold=2, redis_client=mock_redis)
        cb.record_failure()
        assert cb.is_open is False

    def test_expire_called_on_write(self):
        """Every write should refresh the key TTL."""
        mock_redis = self._make_redis_mock()
        cb = CircuitBreaker(
            "test", threshold=3, reset_timeout=60.0, redis_client=mock_redis
        )
        cb.record_failure()
        mock_redis.expire.assert_called_once_with(cb._redis_key, cb._key_ttl)

    def test_ttl_minimum_300s(self):
        """TTL should be at least 300s even for tiny reset_timeout."""
        mock_redis = self._make_redis_mock()
        cb = CircuitBreaker(
            "test", threshold=2, reset_timeout=1.0, redis_client=mock_redis
        )
        assert cb._key_ttl == 300

    def test_ttl_scales_with_reset_timeout(self):
        """TTL = 3 * reset_timeout when that exceeds 300s."""
        mock_redis = self._make_redis_mock()
        cb = CircuitBreaker(
            "test", threshold=2, reset_timeout=200.0, redis_client=mock_redis
        )
        assert cb._key_ttl == 600


# ───────────────────────────────────────────────────────────────────
# create_circuit_breaker factory
# ───────────────────────────────────────────────────────────────────


class TestCreateCircuitBreaker:
    """Tests for the factory function."""

    def test_creates_in_memory_when_no_redis(self):
        with patch("deep_agent.src.error_handling.get_redis_client", return_value=None):
            cb = create_circuit_breaker("test", threshold=3)
            assert cb._redis is None

    def test_creates_redis_backed_when_available(self):
        mock_redis = MagicMock()
        with patch(
            "deep_agent.src.error_handling.get_redis_client",
            return_value=mock_redis,
        ):
            cb = create_circuit_breaker("test", threshold=3)
            assert cb._redis is mock_redis

    def test_explicit_redis_client_overrides_auto_detect(self):
        mock_redis = MagicMock()
        cb = create_circuit_breaker("test", threshold=3, redis_client=mock_redis)
        assert cb._redis is mock_redis
