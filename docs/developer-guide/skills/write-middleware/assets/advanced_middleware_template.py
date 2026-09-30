"""Production middleware template with logging, timing, and error handling.

Demonstrates the patterns used by AuditMiddleware and OPAMiddleware in the
codebase. Customize the hook bodies for your use case -- delete any hooks
you don't need.

Registration: add to config/agent/runtime/agent.yaml under middleware.extra:
  extra:
    - "your_module.path:InstrumentedMiddleware"

Important: the constructor must accept zero arguments because
_import_middleware() calls the class with no args. Read configuration
from environment variables or settings inside __init__.
"""

import time
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


def _model_name(request: ModelRequest) -> str:
    """Extract a human-readable model name from the request."""
    model = request.model
    if isinstance(model, str):
        return model
    name: str | None = getattr(model, "model_name", None) or getattr(
        model, "model", None
    )
    return name or "unknown"


def _get_thread_id() -> str | None:
    """Extract thread_id from LangGraph config, or None if unavailable."""
    try:
        from langgraph.config import get_config

        config: dict[str, Any] = get_config() or {}
        result: str | None = config.get("configurable", {}).get("thread_id")
        return result
    except Exception:
        return None


class InstrumentedMiddleware(AgentMiddleware):
    """Production middleware template with logging and timing.

    Wraps both model calls and tool calls with:
    - Structured logging (start/end events with context)
    - Latency measurement via time.monotonic()
    - Error logging that always re-raises (never swallow pipeline errors)
    """

    # ── Model call wrapping ──────────────────────────────────────────────

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse:
        """Async wrapper for LLM calls with timing and logging."""
        thread_id = _get_thread_id() or "unknown"
        model = _model_name(request)
        msg_count = len(request.messages)

        logger.info(
            "model_call_start thread_id=%s model=%s message_count=%d",
            thread_id,
            model,
            msg_count,
        )
        started = time.monotonic()

        try:
            response = await handler(request)
        except Exception:
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            logger.exception(
                "model_call_error thread_id=%s model=%s elapsed_ms=%.2f",
                thread_id,
                model,
                elapsed_ms,
            )
            raise  # Always re-raise -- don't break the pipeline

        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        logger.info(
            "model_call_end thread_id=%s model=%s elapsed_ms=%.2f",
            thread_id,
            model,
            elapsed_ms,
        )
        return response

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse:
        """Sync wrapper for LLM calls (used by synchronous subagent invocations)."""
        thread_id = _get_thread_id() or "unknown"
        model = _model_name(request)

        logger.info("model_call_start thread_id=%s model=%s", thread_id, model)
        started = time.monotonic()

        try:
            response = handler(request)
        except Exception:
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            logger.exception(
                "model_call_error thread_id=%s model=%s elapsed_ms=%.2f",
                thread_id,
                model,
                elapsed_ms,
            )
            raise

        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        logger.info(
            "model_call_end thread_id=%s model=%s elapsed_ms=%.2f",
            thread_id,
            model,
            elapsed_ms,
        )
        return response

    # ── Tool call wrapping ───────────────────────────────────────────────

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Any,
    ) -> ToolMessage | Command[Any]:
        """Async wrapper for tool calls with timing and logging."""
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "unknown")
        thread_id = _get_thread_id() or "unknown"

        logger.info(
            "tool_call_start thread_id=%s tool=%s",
            thread_id,
            tool_name,
        )
        started = time.monotonic()

        try:
            result = await handler(request)
        except Exception:
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            logger.exception(
                "tool_call_error thread_id=%s tool=%s elapsed_ms=%.2f",
                thread_id,
                tool_name,
                elapsed_ms,
            )
            raise

        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        logger.info(
            "tool_call_end thread_id=%s tool=%s elapsed_ms=%.2f",
            thread_id,
            tool_name,
            elapsed_ms,
        )
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Any,
    ) -> ToolMessage | Command[Any]:
        """Sync wrapper for tool calls (used by synchronous subagent invocations)."""
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "unknown")
        thread_id = _get_thread_id() or "unknown"

        logger.info("tool_call_start thread_id=%s tool=%s", thread_id, tool_name)
        started = time.monotonic()

        try:
            result = handler(request)
        except Exception:
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            logger.exception(
                "tool_call_error thread_id=%s tool=%s elapsed_ms=%.2f",
                thread_id,
                tool_name,
                elapsed_ms,
            )
            raise

        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        logger.info(
            "tool_call_end thread_id=%s tool=%s elapsed_ms=%.2f",
            thread_id,
            tool_name,
            elapsed_ms,
        )
        return result
