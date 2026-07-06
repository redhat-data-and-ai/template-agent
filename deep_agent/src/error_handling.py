"""Centralized error handling: retry decorators, circuit breaker, fallback patterns.

This module provides production-grade error handling utilities built on tenacity.
It separates *how* we handle errors (retry, circuit break, degrade) from *what*
errors look like (exceptions.py).

Usage:
    from deep_agent.src.error_handling import llm_retry, mcp_retry, create_circuit_breaker

    @llm_retry
    def create_model(name: str) -> ChatModel: ...

    breaker = create_circuit_breaker("mcp-server", threshold=3)
    if breaker.is_open:
        return fallback()
"""

import asyncio
import logging as _logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from deep_agent.src.exceptions import (
    AppException,
    LLMError,
    MCPError,
    RateLimitError,
    TransientError,
)
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Retry callbacks (shared across decorators)
# ---------------------------------------------------------------------------


def _log_retry(retry_state: RetryCallState) -> None:
    """Log retry attempts with structured context."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retry %d/%d for '%s': %s",
        retry_state.attempt_number,
        retry_state.retry_object.stop.max_attempt_number,
        retry_state.fn.__name__ if retry_state.fn else "unknown",
        exc,
    )


# ---------------------------------------------------------------------------
# Retry decorators
# ---------------------------------------------------------------------------

llm_retry = retry(
    retry=retry_if_exception_type((LLMError, RateLimitError, ConnectionError, OSError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=_log_retry,
    reraise=True,
)

mcp_retry = retry(
    retry=retry_if_exception_type((MCPError, ConnectionError, TimeoutError, OSError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    before_sleep=_log_retry,
    reraise=True,
)

subagent_retry = retry(
    retry=retry_if_exception_type((TransientError, ConnectionError, OSError)),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=_log_retry,
    reraise=True,
)

try:
    from pymongo.errors import (
        AutoReconnect,
        ConnectionFailure,
        NetworkTimeout,
        NotPrimaryError,
        ServerSelectionTimeoutError,
    )

    _MONGO_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
        AutoReconnect,
        ConnectionFailure,
        NetworkTimeout,
        NotPrimaryError,
        ServerSelectionTimeoutError,
        ConnectionError,
        TimeoutError,
        OSError,
    )
except ImportError:  # pragma: no cover - pymongo optional at import time
    _MONGO_TRANSIENT_ERRORS = (ConnectionError, TimeoutError, OSError)

mongo_retry = retry(
    retry=retry_if_exception_type(_MONGO_TRANSIENT_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, min=0.1, max=2),
    before_sleep=_log_retry,
    reraise=True,
)


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

_REDIS_KEY_PREFIX = "aegra:circuit:"


class CircuitBreaker:
    """Circuit breaker for external service calls with optional Redis persistence.

    Tracks consecutive failures. After ``threshold`` failures, the circuit
    opens and remains open for ``reset_timeout`` seconds, during which
    calls should be skipped (or use a fallback).

    When ``redis_client`` is provided, state is stored in a Redis hash,
    enabling multi-replica awareness. When Redis is unavailable or not
    provided, state is kept in-memory (single-process only).

    Redis errors never propagate — the breaker degrades to "closed"
    (allow all requests) if Redis is unreachable.

    Args:
        name: Human-readable name (also used as Redis key suffix).
        threshold: Consecutive failures before opening.
        reset_timeout: Seconds to wait before allowing a probe (half-open).
        redis_client: Optional Redis client. Pass explicitly or use
            ``create_circuit_breaker()`` for auto-detection.
    """

    def __init__(
        self,
        name: str,
        threshold: int = 5,
        reset_timeout: float = 60.0,
        redis_client: Any = None,
    ) -> None:
        """Initialize circuit breaker with name, threshold, and optional Redis backing."""
        self.name = name
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self._redis: Any = redis_client
        self._redis_key: str = f"{_REDIS_KEY_PREFIX}{name}"
        self._key_ttl: int = max(int(reset_timeout * 3), 300)

        # In-memory fallback state
        self._mem_failure_count: int = 0
        self._mem_last_failure_time: float = 0.0
        self._mem_state: str = "closed"

    # ── State reading ─────────────────────────────────────────────

    def _read_state(self) -> tuple[int, str, float]:
        """Read (failure_count, state, last_failure_time) from backend."""
        if self._redis is not None:
            try:
                data: dict[str, str] = self._redis.hgetall(self._redis_key)
                if not data:
                    return 0, "closed", 0.0
                return (
                    int(data.get("failures", "0")),
                    data.get("state", "closed"),
                    float(data.get("last_failure_ts", "0")),
                )
            except Exception:
                logger.debug(
                    "Circuit '%s' Redis read failed — falling back to closed",
                    self.name,
                )
                return 0, "closed", 0.0
        return self._mem_failure_count, self._mem_state, self._mem_last_failure_time

    def _write_state(self, failures: int, state: str, last_failure_ts: float) -> None:
        """Write state to backend."""
        if self._redis is not None:
            try:
                self._redis.hset(
                    self._redis_key,
                    mapping={
                        "failures": str(failures),
                        "state": state,
                        "last_failure_ts": str(last_failure_ts),
                    },
                )
                self._redis.expire(self._redis_key, self._key_ttl)
                return
            except Exception:
                logger.debug(
                    "Circuit '%s' Redis write failed — using in-memory",
                    self.name,
                )
        self._mem_failure_count = failures
        self._mem_state = state
        self._mem_last_failure_time = last_failure_ts

    def _clear_state(self) -> None:
        """Clear all state (reset to closed)."""
        if self._redis is not None:
            try:
                self._redis.delete(self._redis_key)
                return
            except Exception:
                logger.debug("Circuit '%s' Redis delete failed", self.name)
        self._mem_failure_count = 0
        self._mem_state = "closed"
        self._mem_last_failure_time = 0.0

    # ── Public API ────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        """True when the circuit is open (calls should be skipped)."""
        failures, state, last_ts = self._read_state()
        if state == "open":
            if time.monotonic() - last_ts >= self.reset_timeout:
                self._write_state(failures, "half-open", last_ts)
                logger.info(
                    "Circuit '%s' half-open — allowing probe request", self.name
                )
                return False
            return True
        return False

    @property
    def state(self) -> str:
        """Current circuit state: closed, open, or half-open."""
        _ = self.is_open
        _, state, _ = self._read_state()
        return state

    def record_success(self) -> None:
        """Record a successful call. Resets failure count and closes circuit."""
        failures, state, _ = self._read_state()
        if failures > 0 or state != "closed":
            logger.info(
                "Circuit '%s' reset after success (was %s, %d failures)",
                self.name,
                state,
                failures,
            )
        self._clear_state()

    def record_failure(self) -> None:
        """Record a failed call. Opens circuit if threshold exceeded."""
        failures, _, _ = self._read_state()
        failures += 1
        now = time.monotonic()

        new_state = "open" if failures >= self.threshold else "closed"
        self._write_state(failures, new_state, now)

        if new_state == "open":
            logger.warning(
                "Circuit '%s' OPEN after %d consecutive failures (cooldown: %.0fs)",
                self.name,
                failures,
                self.reset_timeout,
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_redis_client() -> Any:
    """Import and return Redis client from aegra.redis (None if unavailable)."""
    try:
        from deep_agent.aegra.redis import get_redis_client as _get

        return _get()
    except Exception:
        return None


def create_circuit_breaker(
    name: str,
    threshold: int = 5,
    reset_timeout: float = 60.0,
    redis_client: Any = None,
) -> CircuitBreaker:
    """Create a CircuitBreaker with auto-detected Redis backend.

    If ``redis_client`` is not provided, attempts to obtain one from
    ``aegra.redis.get_redis_client()``. Falls back to in-memory if
    Redis is unavailable.

    Args:
        name: Circuit name (used as Redis key suffix).
        threshold: Failures before opening.
        reset_timeout: Seconds before half-open probe.
        redis_client: Explicit Redis client (overrides auto-detect).

    Returns:
        Configured CircuitBreaker instance.
    """
    if redis_client is None:
        redis_client = get_redis_client()

    if redis_client is not None:
        logger.info("Circuit '%s' using Redis-backed state", name)
    else:
        logger.info("Circuit '%s' using in-memory state (single-replica)", name)

    return CircuitBreaker(
        name=name,
        threshold=threshold,
        reset_timeout=reset_timeout,
        redis_client=redis_client,
    )


# ---------------------------------------------------------------------------
# Graceful degradation helpers
# ---------------------------------------------------------------------------


def classify_error(exc: Exception) -> dict[str, Any]:
    """Classify an exception into a structured error response for the API.

    Returns a dict suitable for yielding as a stream error event.

    Args:
        exc: The exception to classify.

    Returns:
        Structured error dict with type, message, recoverable flag, and error_type.
    """
    if isinstance(exc, RateLimitError):
        return {
            "message": "Rate limit exceeded — please wait and try again",
            "recoverable": True,
            "error_type": "rate_limit",
        }
    if isinstance(exc, TransientError):
        return {
            "message": f"Service temporarily unavailable: {exc.message}",
            "recoverable": True,
            "error_type": "transient",
        }
    if isinstance(exc, AppException):
        return {
            "message": exc.message,
            "recoverable": False,
            "error_type": exc.code,
        }
    return {
        "message": "Internal server error",
        "recoverable": False,
        "error_type": "unknown",
    }


def with_fallback(
    fallback_value: Any,
    *,
    on: tuple[type[Exception], ...] = (Exception,),
    log_level: int = _logging.WARNING,
) -> Callable[[F], F]:
    """Decorator that returns a fallback value instead of raising.

    Use for non-critical paths where a degraded response is better than
    a failure. The original exception is logged.

    Args:
        fallback_value: Value to return when the wrapped function raises.
        on: Tuple of exception types to catch.
        log_level: Logging level for the caught exception.
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except on as exc:
                logger.log(
                    log_level,
                    "Fallback for '%s': %s (returning %r)",
                    fn.__name__,
                    exc,
                    fallback_value,
                )
                return fallback_value

        @wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except on as exc:
                logger.log(
                    log_level,
                    "Fallback for '%s': %s (returning %r)",
                    fn.__name__,
                    exc,
                    fallback_value,
                )
                return fallback_value

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore[return-value]
        return wrapper  # type: ignore[return-value]

    return decorator
