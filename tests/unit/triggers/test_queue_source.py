"""Unit tests for queue consumer trigger source."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from deep_agent.src.triggers.config import QueueTriggerConfig
from deep_agent.src.triggers.sources.queue import (
    QueueMessage,
    QueueTriggerSource,
    RedisStreamsConsumer,
)


class TestQueueMessage:
    """Test QueueMessage dataclass fields."""

    def test_fields(self):
        msg = QueueMessage(id="123-0", data={"name": "test", "key": "val"})
        assert msg.id == "123-0"
        assert msg.data == {"name": "test", "key": "val"}

    def test_empty_data(self):
        msg = QueueMessage(id="0-0", data={})
        assert msg.data == {}


class TestRedisStreamsConsumer:
    """Test RedisStreamsConsumer with mocked redis."""

    async def test_creates_consumer_group_on_first_consume(self):
        consumer = RedisStreamsConsumer(
            stream="test-stream",
            consumer_group="test-group",
            consumer_name="worker-1",
            redis_url="redis://localhost:6379/0",
        )

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()

        with patch(
            "redis.asyncio.from_url",
            return_value=mock_client,
        ):
            await consumer._ensure_client()

        mock_client.xgroup_create.assert_awaited_once_with(
            "test-stream", "test-group", id="0", mkstream=True
        )

    async def test_handles_busygroup_error(self):
        consumer = RedisStreamsConsumer(
            stream="s1",
            consumer_group="g1",
            consumer_name="w1",
        )

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock(
            side_effect=Exception("BUSYGROUP Consumer Group name already exists")
        )

        with patch(
            "redis.asyncio.from_url",
            return_value=mock_client,
        ):
            # Should not raise — BUSYGROUP is silently ignored.
            await consumer._ensure_client()

        mock_client.xgroup_create.assert_awaited_once()

    async def test_non_busygroup_error_propagates(self):
        consumer = RedisStreamsConsumer(
            stream="s1",
            consumer_group="g1",
            consumer_name="w1",
        )

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock(
            side_effect=Exception("NOPERM Insufficient permissions")
        )

        with (
            patch(
                "redis.asyncio.from_url",
                return_value=mock_client,
            ),
            pytest.raises(Exception, match="NOPERM"),
        ):
            await consumer._ensure_client()

    async def test_ack_calls_xack(self):
        consumer = RedisStreamsConsumer(
            stream="my-stream",
            consumer_group="my-group",
            consumer_name="w1",
        )

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.xack = AsyncMock()

        with patch(
            "redis.asyncio.from_url",
            return_value=mock_client,
        ):
            # Initialize client.
            await consumer._ensure_client()
            msg = QueueMessage(id="1234-0", data={"key": "val"})
            await consumer.ack(msg)

        mock_client.xack.assert_awaited_once_with("my-stream", "my-group", "1234-0")

    async def test_close_stops_running_and_closes_client(self):
        consumer = RedisStreamsConsumer(
            stream="s", consumer_group="g", consumer_name="w"
        )

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.aclose = AsyncMock()

        with patch(
            "redis.asyncio.from_url",
            return_value=mock_client,
        ):
            await consumer._ensure_client()
            assert consumer._running is True

            await consumer.close()

        assert consumer._running is False
        assert consumer._client is None
        mock_client.aclose.assert_awaited_once()

    async def test_close_without_client_is_safe(self):
        consumer = RedisStreamsConsumer(
            stream="s", consumer_group="g", consumer_name="w"
        )
        # Should not raise when no client is initialized.
        await consumer.close()
        assert consumer._running is False
        assert consumer._client is None


class TestQueueTriggerSource:
    """Test QueueTriggerSource lifecycle and event wrapping."""

    async def test_start_creates_consumer_and_task(self):
        config = QueueTriggerConfig(
            enabled=True,
            backend="redis_streams",
            stream="my-tasks",
            consumer_group="workers",
            consumer_name="w-1",
        )

        mock_consumer_instance = AsyncMock()

        async def _mock_consume():
            return
            yield  # Make it an async generator.

        mock_consumer_instance.consume = _mock_consume

        with patch(
            "deep_agent.src.triggers.sources.queue.RedisStreamsConsumer",
            return_value=mock_consumer_instance,
        ):
            source = QueueTriggerSource(config, redis_url="redis://test:6379/0")
            await source.start()

        assert source._consumer is not None
        assert source._task is not None

        # Clean up.
        await source.stop()

    async def test_unsupported_backend_raises_value_error(self):
        config = QueueTriggerConfig(
            enabled=True,
            backend="rabbitmq",
        )
        source = QueueTriggerSource(config)

        with pytest.raises(ValueError, match="(?i)unsupported queue backend: rabbitmq"):
            await source.start()

    async def test_wraps_messages_as_trigger_events(self):
        config = QueueTriggerConfig(
            enabled=True,
            backend="redis_streams",
            stream="tasks",
        )

        messages = [
            QueueMessage(id="1-0", data={"name": "task-a", "input": "hello"}),
            QueueMessage(id="2-0", data={"name": "task-b", "input": "world"}),
        ]

        mock_consumer = AsyncMock()

        async def _mock_consume():
            for m in messages:
                yield m

        mock_consumer.consume = _mock_consume
        mock_consumer.ack = AsyncMock()

        with patch(
            "deep_agent.src.triggers.sources.queue.RedisStreamsConsumer",
            return_value=mock_consumer,
        ):
            source = QueueTriggerSource(config)
            await source.start()

            # Wait for the consume loop to process the messages.
            await asyncio.sleep(0.1)

        assert source._queue.qsize() == 2

        event_a = source._queue.get_nowait()
        assert event_a.name == "task-a"
        assert event_a.source == "queue"
        assert event_a.payload == {"name": "task-a", "input": "hello"}
        assert event_a.metadata["message_id"] == "1-0"
        assert event_a.metadata["stream"] == "tasks"

        event_b = source._queue.get_nowait()
        assert event_b.name == "task-b"

        # Each message should have been acknowledged.
        assert mock_consumer.ack.await_count == 2

        await source.stop()

    async def test_stop_cancels_task_and_closes_consumer(self):
        config = QueueTriggerConfig(enabled=True, backend="redis_streams")

        mock_consumer = AsyncMock()

        async def _mock_consume():
            # Block indefinitely until cancelled.
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                return
            yield  # noqa: F841 — makes this an async generator

        mock_consumer.consume = _mock_consume
        mock_consumer.close = AsyncMock()

        with patch(
            "deep_agent.src.triggers.sources.queue.RedisStreamsConsumer",
            return_value=mock_consumer,
        ):
            source = QueueTriggerSource(config)
            await source.start()

            assert source._task is not None
            assert source._consumer is not None

            await source.stop()

        assert source._task is None
        assert source._consumer is None
        mock_consumer.close.assert_awaited_once()

    async def test_stop_when_not_started_is_safe(self):
        config = QueueTriggerConfig()
        source = QueueTriggerSource(config)
        # Should not raise.
        await source.stop()
        assert source._task is None
        assert source._consumer is None

    async def test_aiter_returns_self(self):
        config = QueueTriggerConfig()
        source = QueueTriggerSource(config)
        assert source.__aiter__() is source

    async def test_default_event_name_when_missing(self):
        config = QueueTriggerConfig(enabled=True, backend="redis_streams")

        # Message data without "name" key should default to "queue-event".
        mock_consumer = AsyncMock()

        async def _mock_consume():
            yield QueueMessage(id="99-0", data={"input": "something"})

        mock_consumer.consume = _mock_consume
        mock_consumer.ack = AsyncMock()

        with patch(
            "deep_agent.src.triggers.sources.queue.RedisStreamsConsumer",
            return_value=mock_consumer,
        ):
            source = QueueTriggerSource(config)
            await source.start()
            await asyncio.sleep(0.1)

        event = source._queue.get_nowait()
        assert event.name == "queue-event"

        await source.stop()
