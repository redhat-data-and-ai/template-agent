"""Protocols and data classes for trigger sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class TriggerEvent:
    """An event produced by a trigger source."""

    name: str
    payload: dict[str, Any]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@runtime_checkable
class TriggerSource(Protocol):
    """Async trigger source that yields events."""

    async def start(self) -> None:
        """Start the trigger source."""
        ...

    async def stop(self) -> None:
        """Stop the trigger source."""
        ...

    def __aiter__(self) -> AsyncIterator[TriggerEvent]:
        """Return the async iterator."""
        ...

    async def __anext__(self) -> TriggerEvent:
        """Return the next trigger event."""
        ...
