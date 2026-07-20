"""Configuration model for code execution middleware."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CodeExecutionConfig(BaseModel):
    """Configuration for code execution middleware."""

    enabled: bool = False
    max_timeout_seconds: int = Field(default=60, ge=5, le=300)
    max_code_length: int = Field(default=50_000, ge=100, le=500_000)
    max_output_bytes: int = Field(default=1_048_576)

    images: dict[str, str] = Field(
        default_factory=lambda: {
            "python": "python:3.12-slim",
            "shell": "bash:5",
            "node": "node:22-slim",
        }
    )

    entrypoints: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "python": ["python", "-c"],
            "shell": ["bash", "-c"],
            "node": ["node", "-e"],
        }
    )

    resource_requests: dict[str, str] = Field(
        default_factory=lambda: {"cpu": "100m", "memory": "128Mi"}
    )
    resource_limits: dict[str, str] = Field(
        default_factory=lambda: {"cpu": "500m", "memory": "256Mi"}
    )

    tmp_size_limit: str = "64Mi"
    job_ttl_after_finished: int = Field(default=30, ge=0, le=300)
    pod_poll_interval_seconds: float = Field(default=1.0, ge=0.5, le=10.0)
    pod_poll_timeout_seconds: float = Field(default=120.0, ge=10.0, le=600.0)

    network_access: Literal["deny", "allow_internet", "per_execution"] = "deny"
    max_concurrent_per_org: int = Field(default=3, ge=1, le=20)
    queue_timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    max_input_file_size: int = Field(default=1_048_576)
    cost_tracking_enabled: bool = False
    streaming_enabled: bool = False

    @property
    def supported_languages(self) -> set[str]:
        """Return supported language names."""
        return set(self.images.keys()) & set(self.entrypoints.keys())
