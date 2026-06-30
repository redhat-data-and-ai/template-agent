"""File output sink — appends results as JSONL."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from deep_agent.src.triggers.sinks.protocol import OutputSink, TriggerResult
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


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


class FileSink(OutputSink):
    """Appends TriggerResult as JSONL to a file."""

    def __init__(self, path: str) -> None:
        """Initialize the file sink with the output path."""
        self._path = Path(path)
        self._handle: IO[str] | None = None

    def _ensure_handle(self) -> IO[str]:
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        return self._handle

    async def emit(self, result: TriggerResult) -> None:
        """Append the trigger result as a JSON line to the file."""
        try:
            data = asdict(result)
            line = json.dumps(data, default=_default_serializer)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._write_sync, line)
        except Exception:
            logger.exception("Failed to write to file sink: %s", self._path)

    def _write_sync(self, line: str) -> None:
        """Synchronous write executed in a thread pool to avoid blocking the event loop."""
        handle = self._ensure_handle()
        handle.write(line + "\n")
        handle.flush()

    async def close(self) -> None:
        """Close the file handle."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
