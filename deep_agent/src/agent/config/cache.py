"""Cache configuration models.

Provides validated Pydantic models for the ``cache:`` section of
config/agent/runtime/agent.yaml. Controls TTLs, feature flags, and
size limits for all cache layers (model, personalization, MCP tools,
compiled graph, Redis L2, warming, and metrics).

The template-agent user only touches YAML. This module converts
declarative config into parameters consumed by cache infrastructure.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelCacheConfig(BaseModel):
    """LLM model instance cache settings."""

    enabled: bool = True
    ttl: int = Field(default=600, ge=10, le=7200)
    max_size: int = Field(default=50, ge=1, le=100)


class PersonalizationCacheConfig(BaseModel):
    """User personalization (memories/rules) cache settings."""

    enabled: bool = True
    ttl: int = Field(default=120, ge=10, le=3600)


class McpCacheConfig(BaseModel):
    """MCP tool discovery cache settings."""

    ttl: int = Field(default=300, ge=10, le=3600)


class GraphCacheConfig(BaseModel):
    """Compiled graph cache settings."""

    ttl: int = Field(default=300, ge=10, le=3600)


class ToggleConfig(BaseModel):
    """Generic feature toggle with enabled flag."""

    enabled: bool = True


class CacheFileConfig(BaseModel):
    """Top-level cache configuration from agent.yaml ``cache:`` section."""

    enabled: bool = True
    model: ModelCacheConfig = Field(default_factory=ModelCacheConfig)
    personalization: PersonalizationCacheConfig = Field(
        default_factory=PersonalizationCacheConfig,
    )
    mcp: McpCacheConfig = Field(default_factory=McpCacheConfig)
    graph: GraphCacheConfig = Field(default_factory=GraphCacheConfig)
    redis: ToggleConfig = Field(default_factory=ToggleConfig)
    warming: ToggleConfig = Field(default_factory=ToggleConfig)
    metrics: ToggleConfig = Field(default_factory=ToggleConfig)
