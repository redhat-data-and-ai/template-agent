"""Queue trigger source — abstract consumer protocol + Redis Streams.

Defines a ``QueueConsumer`` protocol with ``consume``/``ack``/``close``
methods, a concrete ``RedisStreamsConsumer`` implementation using
``redis.asyncio``, and a ``QueueTriggerSource`` adapter that wraps any
``QueueConsumer`` into the ``TriggerSource`` async-iterator interface.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from deep_agent.src.triggers.config import QueueTriggerConfig
from deep_agent.src.triggers.sources.protocol import TriggerEvent
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Default Redis URL (matches the project convention in aegra/redis.py).
_DEFAULT_REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

# Maximum backoff delay for reconnection (seconds).
_MAX_BACKOFF = 60.0

# Number of messages to fetch per XREADGROUP call.
_READ_COUNT = 10

# Block timeout for XREADGROUP (milliseconds).
_BLOCK_MS = 5_000


# ------------------------------------------------------------------
# Queue abstractions
# ------------------------------------------------------------------


@dataclass
class QueueMessage:
    """A single message consumed from a queue backend."""

    id: str
    data: dict[str, Any]


@runtime_checkable
class QueueConsumer(Protocol):
    """Protocol for consuming messages from a queue backend."""

    def consume(self) -> AsyncIterator[QueueMessage]:
        """Consume messages from the queue backend."""
        ...

    async def ack(self, message: QueueMessage) -> None:
        """Acknowledge a consumed message."""
        ...

    async def close(self) -> None:
        """Close the consumer and release resources."""
        ...


# ------------------------------------------------------------------
# Redis Streams implementation
# ------------------------------------------------------------------


class RedisStreamsConsumer:
    """Consumes messages from a Redis Stream using consumer groups.

    Handles:
    - Automatic consumer group creation (``XGROUP CREATE ... MKSTREAM``).
    - Blocking reads via ``XREADGROUP``.
    - Message acknowledgment via ``XACK``.
    - Reconnection with exponential backoff on connection failures.
    """

    def __init__(
        self,
        stream: str,
        consumer_group: str,
        consumer_name: str,
        redis_url: str = _DEFAULT_REDIS_URL,
        block_ms: int = _BLOCK_MS,
        read_count: int = _READ_COUNT,
    ) -> None:
        """Initialize the Redis Streams consumer with connection and group settings."""
        self._stream = stream
        self._group = consumer_group
        self._consumer = consumer_name
        self._redis_url = redis_url
        self._block_ms = block_ms
        self._read_count = read_count
        self._client: Any = None
        self._running = True

    async def _ensure_client(self) -> Any:
        """Lazily create the Redis client and consumer group."""
        if self._client is not None:
            return self._client

        import redis.asyncio as aioredis

        self._client = aioredis.from_url(
            self._redis_url,
            decode_responses=True,
        )

        # Create the consumer group if it does not already exist.
        try:
            await self._client.xgroup_create(
                self._stream, self._group, id="0", mkstream=True
            )
            logger.info(
                "redis consumer group created",
                stream=self._stream,
                group=self._group,
            )
        except Exception as exc:
            # BUSYGROUP means the group already exists — safe to ignore.
            if "BUSYGROUP" not in str(exc):
                raise

        return self._client

    async def consume(self) -> AsyncIterator[QueueMessage]:
        """Yield messages from the stream, reconnecting on failure."""
        backoff = 1.0

        while self._running:
            try:
                client = await self._ensure_client()
                results = await client.xreadgroup(
                    self._group,
                    self._consumer,
                    {self._stream: ">"},
                    count=self._read_count,
                    block=self._block_ms,
                )
                # Reset backoff on successful read (even if no messages).
                backoff = 1.0

                if not results:
                    continue

                for _stream_name, messages in results:
                    for msg_id, data in messages:
                        yield QueueMessage(id=msg_id, data=data)

            except asyncio.CancelledError:
                return
            except Exception:
                logger.error(
                    "redis streams consumer error, reconnecting",
                    backoff_seconds=backoff,
                    stream=self._stream,
                    exc_info=True,
                )
                # Tear down the broken client so _ensure_client rebuilds it.
                await self._close_client()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

    async def ack(self, message: QueueMessage) -> None:
        """Acknowledge a consumed message."""
        client = await self._ensure_client()
        await client.xack(self._stream, self._group, message.id)

    async def close(self) -> None:
        """Stop consuming and close the Redis connection."""
        self._running = False
        await self._close_client()

    async def _close_client(self) -> None:
        """Close the underlying Redis connection if open."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                logger.debug("redis client close error", exc_info=True)
            self._client = None


# ------------------------------------------------------------------
# TriggerSource adapter
# ------------------------------------------------------------------


class QueueTriggerSource:
    """Adapts a ``QueueConsumer`` into the ``TriggerSource`` protocol.

    Messages consumed from the queue are placed on an internal event
    queue with the original message and consumer reference in metadata,
    allowing downstream middleware to acknowledge after processing.
    """

    def __init__(
        self,
        config: QueueTriggerConfig,
        redis_url: str = _DEFAULT_REDIS_URL,
    ) -> None:
        """Initialize the queue trigger source with configuration and Redis URL."""
        self._config = config
        self._redis_url = redis_url
        self._queue: asyncio.Queue[TriggerEvent] = asyncio.Queue()
        self._consumer: RedisStreamsConsumer | QueueConsumer | None = None
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create the queue consumer and start the consume loop."""
        if self._config.backend == "redis_streams":
            consumer_name = self._config.get_consumer_name()
            self._consumer = RedisStreamsConsumer(
                stream=self._config.stream,
                consumer_group=self._config.consumer_group,
                consumer_name=consumer_name,
                redis_url=self._redis_url,
            )
        elif self._config.backend == "kafka":
            from deep_agent.src.triggers.sources.kafka_consumer import (
                KafkaQueueConsumer,
            )

            self._consumer = KafkaQueueConsumer(
                topic=self._config.topic,
                bootstrap_servers=self._config.bootstrap_servers,
                consumer_group=self._config.consumer_group,
            )
        else:
            raise ValueError(f"unsupported queue backend: {self._config.backend}")

        self._task = asyncio.create_task(self._consume_loop())
        logger.info(
            "queue trigger source started",
            backend=self._config.backend,
            stream=self._config.stream,
            consumer_group=self._config.consumer_group,
            consumer_name=self._config.get_consumer_name(),
        )

    async def stop(self) -> None:
        """Cancel the consume loop and close the consumer."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._consumer is not None:
            await self._consumer.close()
            self._consumer = None
        logger.info("queue trigger source stopped")

    # ------------------------------------------------------------------
    # Async-iterator protocol
    # ------------------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[TriggerEvent]:
        """Return the async iterator."""
        return self

    async def __anext__(self) -> TriggerEvent:
        """Return the next queue trigger event."""
        return await self._queue.get()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _consume_loop(self) -> None:
        """Read messages from the consumer and adapt to TriggerEvent."""
        if self._consumer is None:
            return

        try:
            async for message in self._consumer.consume():
                event = TriggerEvent(
                    name=message.data.get("name", "queue-event"),
                    payload=dict(message.data),
                    source="queue",
                    metadata={
                        "message_id": message.id,
                        "stream": self._config.stream,
                        "_queue_message": message,
                        "_consumer": self._consumer,
                    },
                )
                await self._queue.put(event)
                # Do NOT ack here — middleware will ack after processing.
                logger.debug(
                    "queue event enqueued",
                    event_name=event.name,
                    message_id=message.id,
                )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("queue consume loop error", exc_info=True)
