"""Unit tests for headless mode Pydantic configuration models."""

import pytest

from deep_agent.src.triggers.config import (
    AgentMode,
    CronJobConfig,
    CronTriggerConfig,
    HeadlessConfig,
    OutputSinkConfig,
    QueueTriggerConfig,
    TriggerConfig,
    WebhookTriggerConfig,
)


class TestAgentMode:
    """Test AgentMode enum values."""

    def test_server_value(self):
        assert AgentMode.SERVER == "server"

    def test_headless_value(self):
        assert AgentMode.HEADLESS == "headless"

    def test_server_is_str(self):
        assert isinstance(AgentMode.SERVER, str)

    def test_headless_is_str(self):
        assert isinstance(AgentMode.HEADLESS, str)

    def test_construct_from_string(self):
        assert AgentMode("server") is AgentMode.SERVER
        assert AgentMode("headless") is AgentMode.HEADLESS

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            AgentMode("invalid")


class TestWebhookTriggerConfig:
    """Test WebhookTriggerConfig defaults and custom values."""

    def test_defaults(self):
        cfg = WebhookTriggerConfig()
        assert cfg.enabled is False
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8888
        assert cfg.path == "/trigger"

    def test_custom_values(self):
        cfg = WebhookTriggerConfig(
            enabled=True,
            host="127.0.0.1",
            port=9090,
            path="/webhook",
        )
        assert cfg.enabled is True
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 9090
        assert cfg.path == "/webhook"


class TestCronJobConfig:
    """Test CronJobConfig construction."""

    def test_required_fields(self):
        job = CronJobConfig(name="daily-report", schedule="0 9 * * *")
        assert job.name == "daily-report"
        assert job.schedule == "0 9 * * *"
        assert job.payload == {}

    def test_with_payload(self):
        payload = {"region": "us-east-1", "format": "pdf"}
        job = CronJobConfig(name="export", schedule="*/5 * * * *", payload=payload)
        assert job.payload == payload

    def test_payload_default_factory_isolation(self):
        a = CronJobConfig(name="a", schedule="* * * * *")
        b = CronJobConfig(name="b", schedule="* * * * *")
        a.payload["key"] = "value"
        assert "key" not in b.payload


class TestCronTriggerConfig:
    """Test CronTriggerConfig with jobs list."""

    def test_defaults(self):
        cfg = CronTriggerConfig()
        assert cfg.enabled is False
        assert cfg.jobs == []

    def test_with_jobs(self):
        jobs = [
            CronJobConfig(name="j1", schedule="0 * * * *"),
            CronJobConfig(name="j2", schedule="0 0 * * *", payload={"x": 1}),
        ]
        cfg = CronTriggerConfig(enabled=True, jobs=jobs)
        assert cfg.enabled is True
        assert len(cfg.jobs) == 2
        assert cfg.jobs[0].name == "j1"
        assert cfg.jobs[1].payload == {"x": 1}

    def test_jobs_default_factory_isolation(self):
        a = CronTriggerConfig()
        b = CronTriggerConfig()
        a.jobs.append(CronJobConfig(name="x", schedule="* * * * *"))
        assert len(b.jobs) == 0


class TestQueueTriggerConfig:
    """Test QueueTriggerConfig defaults and custom values."""

    def test_defaults(self):
        cfg = QueueTriggerConfig()
        assert cfg.enabled is False
        assert cfg.backend == "redis_streams"
        assert cfg.stream == "agent-tasks"
        assert cfg.consumer_group == "agent-workers"
        assert cfg.consumer_name == ""
        assert cfg.get_consumer_name() == "worker-1"  # falls back to default

    def test_custom_values(self):
        cfg = QueueTriggerConfig(
            enabled=True,
            backend="kafka",
            stream="custom-stream",
            consumer_group="my-group",
            consumer_name="worker-42",
        )
        assert cfg.enabled is True
        assert cfg.backend == "kafka"
        assert cfg.stream == "custom-stream"
        assert cfg.consumer_group == "my-group"
        assert cfg.consumer_name == "worker-42"


class TestTriggerConfig:
    """Test TriggerConfig nested defaults."""

    def test_nested_defaults(self):
        cfg = TriggerConfig()
        assert isinstance(cfg.webhook, WebhookTriggerConfig)
        assert isinstance(cfg.cron, CronTriggerConfig)
        assert isinstance(cfg.queue, QueueTriggerConfig)
        assert cfg.webhook.enabled is False
        assert cfg.cron.enabled is False
        assert cfg.queue.enabled is False

    def test_override_nested(self):
        cfg = TriggerConfig(
            webhook=WebhookTriggerConfig(enabled=True, port=7777),
        )
        assert cfg.webhook.enabled is True
        assert cfg.webhook.port == 7777
        # Other triggers remain default.
        assert cfg.cron.enabled is False

    def test_default_factory_isolation(self):
        a = TriggerConfig()
        b = TriggerConfig()
        assert a.webhook is not b.webhook


class TestOutputSinkConfig:
    """Test OutputSinkConfig with each sink type."""

    def test_stdout_sink(self):
        sink = OutputSinkConfig(type="stdout")
        assert sink.type == "stdout"
        assert sink.path is None
        assert sink.url is None
        assert sink.headers == {}
        assert sink.stream is None

    def test_file_sink(self):
        sink = OutputSinkConfig(type="file", path="/tmp/output.jsonl")
        assert sink.type == "file"
        assert sink.path == "/tmp/output.jsonl"

    def test_webhook_sink(self):
        sink = OutputSinkConfig(
            type="webhook",
            url="https://example.com/results",
            headers={"Authorization": "Bearer tok"},
        )
        assert sink.type == "webhook"
        assert sink.url == "https://example.com/results"
        assert sink.headers["Authorization"] == "Bearer tok"

    def test_redis_sink(self):
        sink = OutputSinkConfig(type="redis", stream="results-stream")
        assert sink.type == "redis"
        assert sink.stream == "results-stream"

    def test_headers_default_factory_isolation(self):
        a = OutputSinkConfig(type="webhook")
        b = OutputSinkConfig(type="webhook")
        a.headers["X-Custom"] = "value"
        assert "X-Custom" not in b.headers


class TestHeadlessConfig:
    """Test HeadlessConfig full construction and defaults."""

    def test_defaults(self):
        cfg = HeadlessConfig()
        assert cfg.mode is AgentMode.SERVER
        assert isinstance(cfg.triggers, TriggerConfig)
        assert cfg.output_sinks == []
        assert cfg.drain_timeout == 30.0

    def test_full_construction(self):
        cfg = HeadlessConfig(
            mode=AgentMode.HEADLESS,
            triggers=TriggerConfig(
                webhook=WebhookTriggerConfig(enabled=True, port=9000),
                cron=CronTriggerConfig(
                    enabled=True,
                    jobs=[CronJobConfig(name="nightly", schedule="0 0 * * *")],
                ),
                queue=QueueTriggerConfig(enabled=True, stream="tasks"),
            ),
            output_sinks=[
                OutputSinkConfig(type="stdout"),
                OutputSinkConfig(type="file", path="/data/out.jsonl"),
            ],
            drain_timeout=60.0,
        )
        assert cfg.mode is AgentMode.HEADLESS
        assert cfg.triggers.webhook.enabled is True
        assert cfg.triggers.webhook.port == 9000
        assert cfg.triggers.cron.enabled is True
        assert len(cfg.triggers.cron.jobs) == 1
        assert cfg.triggers.queue.stream == "tasks"
        assert len(cfg.output_sinks) == 2
        assert cfg.drain_timeout == 60.0

    def test_output_sinks_default_factory_isolation(self):
        a = HeadlessConfig()
        b = HeadlessConfig()
        a.output_sinks.append(OutputSinkConfig(type="stdout"))
        assert len(b.output_sinks) == 0

    def test_mode_from_string(self):
        cfg = HeadlessConfig(mode="headless")
        assert cfg.mode is AgentMode.HEADLESS
