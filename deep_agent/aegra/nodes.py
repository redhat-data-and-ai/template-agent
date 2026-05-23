"""Error-handling node wrappers for graph execution.

Provides decorator-style wrappers that add retry logic, error capture,
and structured logging around graph node functions. These are used by
the graph builder to make the agent resilient in production.

The deepagents library handles its own internal node execution. These
wrappers sit at the aegra integration boundary, catching errors that
escape the deepagents graph and recording them in platform metadata.
"""

import asyncio
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

MAX_NODE_RETRIES: int = 2
RETRY_DELAY_SECONDS: float = 1.0


def _log_node_retry(retry_state: RetryCallState) -> None:
    """Log node retry attempts."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retry %d/%d for node '%s': %s",
        retry_state.attempt_number,
        retry_state.retry_object.stop.max_attempt_number,
        retry_state.fn.__name__ if retry_state.fn else "unknown",
        exc,
    )


def with_error_handling(node_name: str) -> Callable[..., Any]:
    """Decorator that adds structured error handling to a graph node.

    Catches exceptions, logs them with the node name for traceability,
    and re-raises after recording the failure. Used during graph
    construction to wrap custom nodes added around the deepagents core.

    Args:
        node_name: Human-readable name for log messages.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception:
                logger.exception("Node '%s' failed", node_name)
                raise

        @wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except Exception:
                logger.exception("Node '%s' failed", node_name)
                raise

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return wrapper

    return decorator


def with_retry(
    max_retries: int = MAX_NODE_RETRIES,
    delay: float = RETRY_DELAY_SECONDS,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> Callable[..., Any]:
    """Decorator that retries a node function on failure using tenacity.

    Supports both sync and async functions with exponential backoff.
    Intended for nodes that call external services (MCP tools, LLM APIs)
    where transient failures are expected.

    Args:
        max_retries: Maximum number of retry attempts.
        delay: Base delay in seconds (multiplied exponentially).
        retry_on: Tuple of exception types to retry on.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tenacity_retry = retry(
            retry=retry_if_exception_type(retry_on),
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential(multiplier=delay, min=delay, max=delay * 10),
            before_sleep=_log_node_retry,
            reraise=True,
        )

        if asyncio.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                @tenacity_retry
                async def _inner() -> Any:
                    return await fn(*args, **kwargs)

                return await _inner()

            return async_wrapper
        else:
            wrapped: Callable[..., Any] = tenacity_retry(fn)
            return wrapped

    return decorator


def timed_node(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that logs execution duration of a node function.

    Supports both sync and async functions.
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info("Node '%s' completed in %.2fs", fn.__name__, elapsed)
            return result
        except Exception:
            elapsed = time.perf_counter() - start
            logger.error("Node '%s' failed after %.2fs", fn.__name__, elapsed)
            raise

    @wraps(fn)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = await fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info("Node '%s' completed in %.2fs", fn.__name__, elapsed)
            return result
        except Exception:
            elapsed = time.perf_counter() - start
            logger.error("Node '%s' failed after %.2fs", fn.__name__, elapsed)
            raise

    if asyncio.iscoroutinefunction(fn):
        return async_wrapper
    return wrapper
