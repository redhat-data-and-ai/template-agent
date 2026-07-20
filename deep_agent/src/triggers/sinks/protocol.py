"""Protocols and data classes for output sinks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from deep_agent.src.triggers.sources.protocol import TriggerEvent


@dataclass
class TriggerResult:
    """Result of processing a trigger event through the agent graph."""

    event: TriggerEvent
    output: Any
    duration_ms: float
    success: bool
    error: str | None = None


@runtime_checkable
class OutputSink(Protocol):
    """Async output sink that receives trigger results."""

    async def emit(self, result: TriggerResult) -> None:
        """Emit a trigger result to the sink."""
        ...

    async def close(self) -> None:
        """Close the sink and release resources."""
        ...
