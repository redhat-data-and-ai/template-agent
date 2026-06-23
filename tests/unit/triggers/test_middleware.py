"""Unit tests for EventTriggerMiddleware."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from deep_agent.src.triggers.config import (
    CronTriggerConfig,
    HeadlessConfig,
    OutputSinkConfig,
    QueueTriggerConfig,
    TriggerConfig,
    WebhookTriggerConfig,
)
from deep_agent.src.triggers.middleware import EventTriggerMiddleware
from deep_agent.src.triggers.sinks.protocol import TriggerResult
from deep_agent.src.triggers.sources.protocol import TriggerEvent


def _make_event(**overrides) -> TriggerEvent:
    defaults = {"name": "test-event", "payload": {"k": "v"}, "source": "unit-test"}
    defaults.update(overrides)
    return TriggerEvent(**defaults)


def _make_result(**overrides) -> TriggerResult:
    defaults = {
        "event": _make_event(),
        "output": "ok",
        "duration_ms": 10.0,
        "success": True,
    }
    defaults.update(overrides)
    return TriggerResult(**defaults)


class TestBuildSources:
    """Test _build_sources() returns the correct trigger sources."""

    def test_returns_empty_list_when_no_triggers_enabled(self):
        config = HeadlessConfig()
        mw = EventTriggerMiddleware(config=config, graph=MagicMock())

        sources = mw._build_sources()

        assert sources == []

    @patch("deep_agent.src.triggers.sources.webhook.WebhookTriggerSource")
    def test_returns_webhook_source_when_webhook_enabled(self, mock_cls):
        mock_cls.return_value = MagicMock()
        config = HeadlessConfig(
            triggers=TriggerConfig(webhook=WebhookTriggerConfig(enabled=True))
        )
        mw = EventTriggerMiddleware(config=config, graph=MagicMock())

        sources = mw._build_sources()

        assert len(sources) == 1
        mock_cls.assert_called_once()

    @patch("deep_agent.src.triggers.sources.cron.CronTriggerSource")
    def test_returns_cron_source_when_cron_enabled(self, mock_cls):
        mock_cls.return_value = MagicMock()
        config = HeadlessConfig(
            triggers=TriggerConfig(cron=CronTriggerConfig(enabled=True))
        )
        mw = EventTriggerMiddleware(config=config, graph=MagicMock())

        sources = mw._build_sources()

        assert len(sources) == 1
        mock_cls.assert_called_once()

    @patch("deep_agent.src.triggers.sources.queue.QueueTriggerSource")
    def test_returns_queue_source_when_queue_enabled(self, mock_cls):
        mock_cls.return_value = MagicMock()
        config = HeadlessConfig(
            triggers=TriggerConfig(queue=QueueTriggerConfig(enabled=True))
        )
        mw = EventTriggerMiddleware(config=config, graph=MagicMock())

        sources = mw._build_sources()

        assert len(sources) == 1
        mock_cls.assert_called_once()


class TestBuildSinks:
    """Test _build_sinks() returns the correct output sinks."""

    @patch("deep_agent.src.triggers.sinks.stdout.StdoutSink")
    def test_defaults_to_stdout_when_output_sinks_empty(self, mock_cls):
        mock_cls.return_value = MagicMock()
        config = HeadlessConfig(output_sinks=[])
        mw = EventTriggerMiddleware(config=config, graph=MagicMock())

        sinks = mw._build_sinks()

        assert len(sinks) == 1
        mock_cls.assert_called_once()

    @patch("deep_agent.src.triggers.sinks.stdout.StdoutSink")
    def test_creates_stdout_sink(self, mock_cls):
        mock_cls.return_value = MagicMock()
        config = HeadlessConfig(output_sinks=[OutputSinkConfig(type="stdout")])
        mw = EventTriggerMiddleware(config=config, graph=MagicMock())

        sinks = mw._build_sinks()

        assert len(sinks) == 1
        mock_cls.assert_called_once()

    @patch("deep_agent.src.triggers.sinks.file.FileSink")
    def test_creates_file_sink(self, mock_cls):
        mock_cls.return_value = MagicMock()
        config = HeadlessConfig(
            output_sinks=[OutputSinkConfig(type="file", path="/tmp/test.jsonl")]
        )
        mw = EventTriggerMiddleware(config=config, graph=MagicMock())

        sinks = mw._build_sinks()

        assert len(sinks) == 1
        mock_cls.assert_called_once_with(path="/tmp/test.jsonl")

    @patch("deep_agent.src.triggers.sinks.webhook.WebhookSink")
    def test_creates_webhook_sink(self, mock_cls):
        mock_cls.return_value = MagicMock()
        config = HeadlessConfig(
            output_sinks=[
                OutputSinkConfig(
                    type="webhook",
                    url="https://example.com/hook",
                    headers={"X-Key": "val"},
                )
            ]
        )
        mw = EventTriggerMiddleware(config=config, graph=MagicMock())

        sinks = mw._build_sinks()

        assert len(sinks) == 1
        mock_cls.assert_called_once_with(
            url="https://example.com/hook", headers={"X-Key": "val"}
        )

    @patch("deep_agent.src.triggers.sinks.redis.RedisSink")
    def test_creates_redis_sink(self, mock_cls):
        mock_cls.return_value = MagicMock()
        config = HeadlessConfig(
            output_sinks=[OutputSinkConfig(type="redis", stream="my-stream")]
        )
        mw = EventTriggerMiddleware(config=config, graph=MagicMock())

        sinks = mw._build_sinks()

        assert len(sinks) == 1
        mock_cls.assert_called_once_with(
            stream="my-stream", redis_url="redis://redis:6379/0"
        )


class TestStartStop:
    """Test start() and stop() lifecycle methods."""

    async def test_start_starts_all_sources_and_creates_loop_task(self):
        config = HeadlessConfig()
        mw = EventTriggerMiddleware(config=config, graph=MagicMock())

        mock_source = AsyncMock()
        mock_sink = MagicMock()
        mw._build_sources = MagicMock(return_value=[mock_source])
        mw._build_sinks = MagicMock(return_value=[mock_sink])

        await mw.start()

        mock_source.start.assert_awaited_once()
        assert mw._loop_task is not None

        # Clean up the task
        mw._stop_event.set()
        mw._loop_task.cancel()
        try:
            await mw._loop_task
        except asyncio.CancelledError:
            pass

    async def test_stop_sets_stop_event_and_closes_sinks(self):
        config = HeadlessConfig()
        mw = EventTriggerMiddleware(config=config, graph=MagicMock())

        mock_sink = AsyncMock()
        mw._sinks = [mock_sink]
        mw._sources = []
        mw._stop_event.clear()

        # Create a task that completes quickly once stop is set
        async def _quick_loop():
            await mw._stop_event.wait()

        mw._loop_task = asyncio.create_task(_quick_loop())

        await mw.stop()

        assert mw._stop_event.is_set()
        mock_sink.close.assert_awaited_once()
        assert mw._loop_task is None


class TestEmitResult:
    """Test _emit_result() fans out to all sinks."""

    async def test_emit_result_fans_out_to_all_sinks(self):
        config = HeadlessConfig()
        mw = EventTriggerMiddleware(config=config, graph=MagicMock())

        sink_a = AsyncMock()
        sink_b = AsyncMock()
        mw._sinks = [sink_a, sink_b]

        result = _make_result()
        await mw._emit_result(result)

        sink_a.emit.assert_awaited_once_with(result)
        sink_b.emit.assert_awaited_once_with(result)

    async def test_emit_result_catches_per_sink_errors(self):
        config = HeadlessConfig()
        mw = EventTriggerMiddleware(config=config, graph=MagicMock())

        failing_sink = AsyncMock()
        failing_sink.emit.side_effect = RuntimeError("sink exploded")
        healthy_sink = AsyncMock()
        mw._sinks = [failing_sink, healthy_sink]

        result = _make_result()

        with patch("deep_agent.src.triggers.middleware.logger"):
            await mw._emit_result(result)  # must not raise

        # Healthy sink still receives the result despite the first sink failing
        healthy_sink.emit.assert_awaited_once_with(result)
