"""Stdout output sink — writes results as JSON to stdout."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from deep_agent.src.triggers.sinks.protocol import OutputSink, TriggerResult


def _default_serializer(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "content"):
        return {"type": type(obj).__name__, "content": obj.content}
    return str(obj)


class StdoutSink(OutputSink):
    """Writes TriggerResult as JSON to stdout."""

    async def emit(self, result: TriggerResult) -> None:
        """Write the trigger result as JSON to stdout."""
        data = asdict(result)
        line = json.dumps(data, default=_default_serializer)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    async def close(self) -> None:
        """No-op close for stdout sink."""
        pass
