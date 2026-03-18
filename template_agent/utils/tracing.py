"""Token usage tracking, LLM invocation utilities, and Langfuse tracing.

Provides thread-safe token tracking, cost estimation, a convenience
wrapper for invoking LLMs with automatic usage recording, and centralized
Langfuse tracing infrastructure (client, AgentTracer, StreamTracer).
"""

import asyncio
import logging
import time as _time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional
from uuid import uuid4

from template_agent.src.settings import settings

_MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4@20250514": {"input": 3.0, "output": 15.0},
}

DEFAULT_INPUT_COST_PER_MILLION = 1.25
DEFAULT_OUTPUT_COST_PER_MILLION = 10.0

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Langfuse client & handler initialization (guarded by credentials check)
# ---------------------------------------------------------------------------

client = None
langfuse_handler = None

_has_credentials = bool(
    settings.LANGFUSE_PUBLIC_KEY
    and settings.LANGFUSE_SECRET_KEY
    and settings.LANGFUSE_BASE_URL
)

if _has_credentials:
    try:
        from langfuse import Langfuse
        from langfuse.callback import CallbackHandler

        client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_BASE_URL,
            environment=settings.LANGFUSE_TRACING_ENVIRONMENT,
        )
        langfuse_handler = CallbackHandler(
            trace_name="template-agent",
            environment=settings.LANGFUSE_TRACING_ENVIRONMENT,
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_BASE_URL,
        )
    except Exception as exc:
        logger.warning("Failed to initialize Langfuse: %s", exc)


# ---------------------------------------------------------------------------
# AgentTracer -- wraps a Langfuse trace for per-request tracing
# ---------------------------------------------------------------------------


class AgentTracer:
    """Per-request wrapper around a Langfuse trace.

    Provides convenience helpers for creating child spans, generations,
    events, and scores. All methods are safe to call even when Langfuse
    is not configured -- they simply no-op.
    """

    def __init__(
        self,
        name: str,
        trace_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the tracer with a name and optional trace ID."""
        self._trace = None
        if client is not None:
            self._trace = client.trace(
                id=trace_id or str(uuid4()),
                name=name,
                **kwargs,
            )

    def update(self, **kwargs: Any) -> None:
        """Update the trace with additional metadata or attributes."""
        if self._trace:
            self._trace.update(**kwargs)

    def span(self, name: str, **kwargs: Any) -> Any:
        """Create a child span under this trace."""
        if self._trace:
            return self._trace.span(name=name, **kwargs)
        return None

    def event(self, name: str, **kwargs: Any) -> None:
        """Record an event on the trace."""
        if self._trace:
            self._trace.event(name=name, **kwargs)

    def generation(self, name: str, **kwargs: Any) -> Any:
        """Create a generation span for LLM calls."""
        if self._trace:
            return self._trace.generation(name=name, **kwargs)
        return None

    def score(self, name: str, value: float, **kwargs: Any) -> None:
        """Record a score on the trace."""
        if self._trace:
            self._trace.score(name=name, value=value, **kwargs)

    @property
    def trace_id(self) -> str | None:
        """Return the Langfuse trace ID, or None if not configured."""
        return self._trace.id if self._trace else None


# ---------------------------------------------------------------------------
# StreamTracer -- child span that tracks a streaming response lifecycle
# ---------------------------------------------------------------------------


class StreamTracer:
    """Tracks a streaming response as a child span under a parent tracer."""

    def __init__(
        self,
        parent_tracer: AgentTracer | None = None,
        name: str = "stream",
    ) -> None:
        """Initialize the stream tracer with an optional parent tracer."""
        self._span = None
        self._parent = parent_tracer
        self._message_count = 0
        if parent_tracer:
            self._span = parent_tracer.span(name=name)

    def track_message(self, content: str, role: str = "assistant") -> None:
        """Record a streamed message event on the span."""
        self._message_count += 1
        if self._span and role == "assistant" and content:
            self._span.event(name="stream_message", output=content)

    def track_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Update the span with input and output token counts."""
        if self._span:
            self._span.update(
                metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )

    def track_error(self, error: Exception) -> None:
        """Record an error on the span."""
        if self._span:
            self._span.update(level="ERROR", status_message=str(error))

    def end_stream(self, duration_ms: float | None = None) -> None:
        """End the stream span with message count and optional duration."""
        if self._span:
            metadata: Dict[str, Any] = {"message_count": self._message_count}
            if duration_ms is not None:
                metadata["duration_ms"] = round(duration_ms, 2)
            self._span.end(metadata=metadata)


# ---------------------------------------------------------------------------
# _record_langfuse_generation -- records a single LLM call as a generation
# ---------------------------------------------------------------------------


def _record_langfuse_generation(
    root_tracer: Any,
    response: Any,
    phase: str,
    model: Any,
    start_time: float,
) -> None:
    """Record an LLM call as a Langfuse generation under *root_tracer*."""
    if root_tracer is None:
        return
    try:
        model_name = (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or "unknown"
        )
        input_tokens, output_tokens = extract_usage_from_response(response)
        duration_ms = (_time.time() - start_time) * 1000

        gen = root_tracer.generation(
            name=f"llm.{phase}",
            model=str(model_name),
            usage={"input": input_tokens, "output": output_tokens},
            metadata={
                "phase": phase,
                "duration_ms": round(duration_ms, 2),
            },
        )
        if gen is not None:
            gen.end()
    except Exception as exc:
        logger.debug("Failed to record Langfuse generation: %s", exc)


@dataclass
class TokenUsage:
    """Aggregated token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


@dataclass
class TokenUsageTracker:
    """Thread-safe tracker for aggregating LLM token usage.

    Tracks both cumulative totals and per-phase breakdowns for detailed
    reporting. Useful for monitoring costs across multi-step pipelines.

    Example:
        tracker = TokenUsageTracker(model_name="gemini-2.5-flash")
        tracker.track("planning", input_tokens=500, output_tokens=200)
        tracker.track("research", input_tokens=1000, output_tokens=400)
        summary = tracker.get_summary()
    """

    model_name: str = ""
    _lock: Lock = field(default_factory=Lock, repr=False)
    _total: TokenUsage = field(default_factory=TokenUsage)
    _per_phase: Dict[str, TokenUsage] = field(default_factory=dict)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on model-specific pricing."""
        pricing = _MODEL_PRICING.get(self.model_name)
        input_rate = pricing["input"] if pricing else DEFAULT_INPUT_COST_PER_MILLION
        output_rate = pricing["output"] if pricing else DEFAULT_OUTPUT_COST_PER_MILLION
        return (input_tokens / 1_000_000) * input_rate + (
            output_tokens / 1_000_000
        ) * output_rate

    def track(self, phase: str, input_tokens: int, output_tokens: int) -> None:
        """Track token usage for an LLM call.

        Args:
            phase: The pipeline phase (e.g., "planning", "research", "synthesis").
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens generated.
        """
        if input_tokens < 0 or output_tokens < 0:
            return

        total_tokens = input_tokens + output_tokens
        cost = self._calculate_cost(input_tokens, output_tokens)

        with self._lock:
            self._total.input_tokens += input_tokens
            self._total.output_tokens += output_tokens
            self._total.total_tokens += total_tokens
            self._total.llm_calls += 1
            self._total.estimated_cost_usd += cost

            if phase not in self._per_phase:
                self._per_phase[phase] = TokenUsage()

            phase_usage = self._per_phase[phase]
            phase_usage.input_tokens += input_tokens
            phase_usage.output_tokens += output_tokens
            phase_usage.total_tokens += total_tokens
            phase_usage.llm_calls += 1
            phase_usage.estimated_cost_usd += cost

    def get_total(self) -> TokenUsage:
        """Get cumulative token usage across all phases."""
        with self._lock:
            return TokenUsage(
                input_tokens=self._total.input_tokens,
                output_tokens=self._total.output_tokens,
                total_tokens=self._total.total_tokens,
                llm_calls=self._total.llm_calls,
                estimated_cost_usd=self._total.estimated_cost_usd,
            )

    def get_per_phase(self) -> Dict[str, TokenUsage]:
        """Get token usage breakdown by phase."""
        with self._lock:
            return {
                phase: TokenUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                    llm_calls=usage.llm_calls,
                    estimated_cost_usd=usage.estimated_cost_usd,
                )
                for phase, usage in self._per_phase.items()
            }

    def get_summary(self) -> Dict[str, Any]:
        """Get complete usage summary for streaming to UI."""
        with self._lock:
            return {
                "total": self._total.to_dict(),
                "per_phase": {
                    phase: usage.to_dict() for phase, usage in self._per_phase.items()
                },
            }

    def reset(self) -> None:
        """Reset all tracked usage."""
        with self._lock:
            self._total = TokenUsage()
            self._per_phase = {}

    @property
    def estimated_cost(self) -> float:
        """Estimated cost in USD for sentinel and budget checks."""
        with self._lock:
            return self._total.estimated_cost_usd

    def persist_to_log(self, thread_id: str, user_id: str | None = None) -> None:
        """Log token usage totals as structured log for auditing."""
        try:
            total = self.get_total()
            logger.info(
                "token_usage_persist",
                extra={
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "input_tokens": total.input_tokens,
                    "output_tokens": total.output_tokens,
                    "total_tokens": total.total_tokens,
                    "llm_calls": total.llm_calls,
                    "estimated_cost_usd": round(total.estimated_cost_usd, 6),
                },
            )
        except Exception as e:
            logger.warning("Failed to persist token usage: %s", e)

    def flush_to_langfuse(self, root_tracer: Any) -> None:
        """Update the root trace with token totals and per-phase cost scores."""
        if root_tracer is None:
            return
        try:
            total = self.get_total()
            root_tracer.update(
                metadata={
                    "total_input_tokens": total.input_tokens,
                    "total_output_tokens": total.output_tokens,
                    "total_tokens": total.total_tokens,
                    "llm_calls": total.llm_calls,
                    "estimated_cost_usd": round(total.estimated_cost_usd, 6),
                }
            )
            for phase_name, usage in self.get_per_phase().items():
                root_tracer.score(
                    name=f"cost.{phase_name}",
                    value=round(usage.estimated_cost_usd, 6),
                )
        except Exception as e:
            logger.warning("Failed to flush token usage to Langfuse: %s", e)


def extract_usage_from_response(response: Any) -> tuple[int, int]:
    """Extract input and output token counts from an LLM response.

    Works with various LLM response formats including LangChain AIMessage,
    and responses with usage_metadata or response_metadata attributes.

    Returns:
        Tuple of (input_tokens, output_tokens).
    """
    input_tokens = 0
    output_tokens = 0

    if not response:
        return input_tokens, output_tokens

    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata and isinstance(usage_metadata, dict):
        input_tokens = (
            usage_metadata.get("input_tokens")
            or usage_metadata.get("prompt_tokens")
            or 0
        )
        output_tokens = (
            usage_metadata.get("output_tokens")
            or usage_metadata.get("completion_tokens")
            or 0
        )

    response_metadata = getattr(response, "response_metadata", None)
    if response_metadata and isinstance(response_metadata, dict):
        usage = response_metadata.get("usage", {})
        if usage:
            input_tokens = input_tokens or usage.get("input_tokens", 0)
            output_tokens = output_tokens or usage.get("output_tokens", 0)

    return input_tokens, output_tokens


async def tracked_invoke(
    model: Any,
    messages: Any,
    tracker: Optional[TokenUsageTracker],
    phase: str,
    timeout_seconds: Optional[float] = None,
    llm_semaphore: Optional[asyncio.Semaphore] = None,
    timeout: Optional[float] = None,
    callbacks: Optional[list] = None,
    root_tracer: Any = None,
    **kwargs: Any,
) -> Any:
    """Invoke an LLM and track token usage.

    Convenience wrapper that calls the model and automatically records
    token usage. Supports per-call timeouts, concurrency limiting, and
    Langfuse generation recording via *root_tracer*.

    Args:
        model: The language model to invoke.
        messages: The messages to send to the model.
        tracker: Optional token tracker to record usage.
        phase: The pipeline phase name for tracking.
        timeout_seconds: Per-call timeout in seconds.
        llm_semaphore: Optional semaphore to limit concurrent LLM calls.
        callbacks: Optional list of LangChain callbacks.
        root_tracer: Optional AgentTracer for Langfuse generation recording.

    Returns:
        The model response.

    Raises:
        ValueError: If messages is empty.
        asyncio.TimeoutError: If the call exceeds timeout_seconds.
    """
    if timeout is not None and timeout_seconds is None:
        timeout_seconds = timeout

    if not messages:
        raise ValueError(f"Cannot invoke LLM with empty messages in phase '{phase}'.")

    start_time = _time.time()

    invoke_kwargs: Dict[str, Any] = {}
    if callbacks:
        invoke_kwargs["config"] = {"callbacks": callbacks}

    async def _invoke() -> Any:
        if llm_semaphore is not None:
            async with llm_semaphore:
                return await model.ainvoke(messages, **invoke_kwargs)
        return await model.ainvoke(messages, **invoke_kwargs)

    has_timeout = timeout_seconds is not None and timeout_seconds > 0
    if has_timeout:
        response = await asyncio.wait_for(_invoke(), timeout=timeout_seconds)
    else:
        response = await _invoke()

    if tracker is not None:
        input_tokens, output_tokens = extract_usage_from_response(response)
        if input_tokens > 0 or output_tokens > 0:
            tracker.track(phase, input_tokens, output_tokens)

    _record_langfuse_generation(root_tracer, response, phase, model, start_time)

    return response
