"""Redis output sink — publishes results to a Redis Stream."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

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


class RedisSink(OutputSink):
    """Publishes TriggerResult to a Redis Stream via XADD."""

    def __init__(
        self,
        stream: str,
        redis_url: str = "redis://redis:6379/0",
    ) -> None:
        """Initialize the Redis sink with stream name and connection URL."""
        self._stream = stream
        self._redis_url = redis_url
        self._client: Any = None

    async def _ensure_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    async def emit(self, result: TriggerResult) -> None:
        """Publish the trigger result to the Redis stream."""
        try:
            client = await self._ensure_client()
            data = asdict(result)
            payload = json.dumps(data, default=_default_serializer)
            await client.xadd(self._stream, {"result": payload})
        except Exception:
            logger.exception("Failed to publish to Redis stream: %s", self._stream)

    async def close(self) -> None:
        """Close the Redis client connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
