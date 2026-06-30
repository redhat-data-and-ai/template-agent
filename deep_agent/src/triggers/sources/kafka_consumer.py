"""Kafka consumer implementation of the QueueConsumer protocol."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from deep_agent.src.triggers.sources.queue import QueueMessage
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


class KafkaQueueConsumer:
    """Consumes messages from a Kafka topic using aiokafka.

    Implements the ``QueueConsumer`` protocol so it can be used
    as a drop-in replacement for ``RedisStreamsConsumer``.
    """

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str = "localhost:9092",
        consumer_group: str = "agent-workers",
    ) -> None:
        """Initialize with Kafka connection settings."""
        self._topic = topic
        self._servers = bootstrap_servers
        self._group = consumer_group
        self._consumer: Any = None
        self._running = True

    async def consume(self) -> AsyncIterator[QueueMessage]:
        """Consume messages from the Kafka topic."""
        from aiokafka import AIOKafkaConsumer

        self._consumer = AIOKafkaConsumer(
            self._topic,
            bootstrap_servers=self._servers,
            group_id=self._group,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=False,
        )
        await self._consumer.start()
        logger.info(
            "kafka consumer started",
            topic=self._topic,
            group=self._group,
            servers=self._servers,
        )

        try:
            async for msg in self._consumer:
                if not self._running:
                    return
                data = (
                    msg.value
                    if isinstance(msg.value, dict)
                    else {"payload": str(msg.value)}
                )
                yield QueueMessage(
                    id=f"{msg.partition}-{msg.offset}",
                    data=data,
                )
        except Exception:
            if self._running:
                logger.error("kafka consumer error", exc_info=True)

    async def ack(self, message: QueueMessage) -> None:
        """Manually commit offsets after successful processing."""
        if self._consumer is not None:
            await self._consumer.commit()

    async def close(self) -> None:
        """Stop the Kafka consumer."""
        self._running = False
        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None
        logger.info("kafka consumer stopped")
