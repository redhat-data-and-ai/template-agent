"""Token usage tracking and LLM invocation utilities.

Provides thread-safe token tracking, cost estimation, and a convenience
wrapper for invoking LLMs with automatic usage recording.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional

_MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4@20250514": {"input": 3.0, "output": 15.0},
}

DEFAULT_INPUT_COST_PER_MILLION = 1.25
DEFAULT_OUTPUT_COST_PER_MILLION = 10.0

logger = logging.getLogger(__name__)


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
    **kwargs: Any,
) -> Any:
    """Invoke an LLM and track token usage.

    Convenience wrapper that calls the model and automatically records
    token usage. Supports per-call timeouts and concurrency limiting.

    Args:
        model: The language model to invoke.
        messages: The messages to send to the model.
        tracker: Optional token tracker to record usage.
        phase: The pipeline phase name for tracking.
        timeout_seconds: Per-call timeout in seconds.
        llm_semaphore: Optional semaphore to limit concurrent LLM calls.
        callbacks: Optional list of LangChain callbacks (e.g. Langfuse handler).

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

    return response
