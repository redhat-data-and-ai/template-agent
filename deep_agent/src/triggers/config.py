"""Pydantic configuration models for headless mode triggers and sinks."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AgentMode(StrEnum):
    """Agent runtime mode."""

    SERVER = "server"
    HEADLESS = "headless"


class WebhookTriggerConfig(BaseModel):
    """Config for the webhook trigger source."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8888
    path: str = "/trigger"


class CronJobConfig(BaseModel):
    """A single cron job definition."""

    name: str
    schedule: str
    payload: dict[str, Any] = Field(default_factory=dict)


class CronTriggerConfig(BaseModel):
    """Config for the cron trigger source."""

    enabled: bool = False
    jobs: list[CronJobConfig] = Field(default_factory=list)


class QueueTriggerConfig(BaseModel):
    """Config for the queue consumer trigger source."""

    enabled: bool = False
    backend: str = "redis_streams"
    stream: str = "agent-tasks"
    consumer_group: str = "agent-workers"
    consumer_name: str = ""
    bootstrap_servers: str = "localhost:9092"
    topic: str = "agent-tasks"

    def get_consumer_name(self) -> str:
        """Return consumer_name, defaulting to HOSTNAME for multi-replica support."""
        import os

        return self.consumer_name or os.environ.get("HOSTNAME", "worker-1")


class TriggerConfig(BaseModel):
    """Container for all trigger source configs."""

    webhook: WebhookTriggerConfig = Field(default_factory=WebhookTriggerConfig)
    cron: CronTriggerConfig = Field(default_factory=CronTriggerConfig)
    queue: QueueTriggerConfig = Field(default_factory=QueueTriggerConfig)


class OutputSinkFileConfig(BaseModel):
    """File-specific sink config."""

    path: str = "output.jsonl"


class OutputSinkWebhookConfig(BaseModel):
    """Webhook-specific sink config."""

    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)


class OutputSinkRedisConfig(BaseModel):
    """Redis-specific sink config."""

    stream: str = "agent-results"


class OutputSinkConfig(BaseModel):
    """Config for a single output sink."""

    type: str
    path: str | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    stream: str | None = None


class HealthCheckConfig(BaseModel):
    """Config for the headless worker health check endpoint."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080


class HeadlessConfig(BaseModel):
    """Top-level headless mode configuration."""

    mode: AgentMode = AgentMode.SERVER
    triggers: TriggerConfig = Field(default_factory=TriggerConfig)
    output_sinks: list[OutputSinkConfig] = Field(default_factory=list)
    drain_timeout: float = 30.0
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
