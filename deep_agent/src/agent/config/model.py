"""Model configuration types for per-agent LLM provider selection.

Parses frontmatter ``model:`` fields that may be a legacy string or an
object with explicit provider, model name, and optional fallback chain.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, model_validator

from deep_agent.src.agent.llm import CLAUDE_MODELS, GEMINI_MODELS


class Provider(str, Enum):
    """Supported LLM provider backends."""

    VERTEX = "vertex"
    OPENAI = "openai"
    MAAS = "maas"  # Model as a Service (VLLM)


class ModelSpec(BaseModel):
    """Resolved model configuration with optional fallback."""

    provider: Provider
    name: str
    fallback: ModelSpec | None = None

    @model_validator(mode="after")
    def _validate_name(self) -> ModelSpec:
        if not self.name or not self.name.strip():
            raise ValueError("model name cannot be empty")
        return self

    def display_name(self) -> str:
        """Human-readable model identifier for logging."""
        base = f"{self.provider.value}:{self.name}"
        if self.fallback:
            return f"{base} (fallback: {self.fallback.display_name()})"
        return base


def infer_provider(model_name: str) -> Provider:
    """Infer provider from a legacy model name string.

    Inference logic:
    - Known Gemini/Claude models → VERTEX
    - GPT models (gpt-*, case-insensitive) → OPENAI
    - All other models → MAAS (VLLM for custom models)
    """
    if model_name in GEMINI_MODELS or model_name in CLAUDE_MODELS:
        return Provider.VERTEX
    if model_name.lower().startswith("gpt-"):
        return Provider.OPENAI
    return Provider.MAAS


def parse_model_config(raw: str | dict[str, Any]) -> ModelSpec:
    """Parse a frontmatter ``model`` field into a :class:`ModelSpec`.

    Accepts:
      - Legacy string: ``gemini-2.5-pro`` (provider inferred)
      - Object: ``{provider: vertex, name: gemini-2.5-pro, fallback: {...}}``

    Args:
        raw: Model value from parsed frontmatter.

    Returns:
        Validated ModelSpec.

    Raises:
        ValueError: If the config is invalid or missing required fields.
        TypeError: If raw is neither str nor dict.
    """
    if isinstance(raw, str):
        name = raw.strip()
        if not name:
            raise ValueError("model name cannot be empty")
        return ModelSpec(provider=infer_provider(name), name=name)

    if not isinstance(raw, dict):
        raise TypeError(
            f"model config must be str or dict, got {type(raw).__name__}"
        )

    allowed_keys = {"provider", "name", "fallback"}
    unknown = set(raw.keys()) - allowed_keys
    if unknown:
        raise ValueError(
            f"unknown model config keys: {sorted(unknown)}; "
            f"allowed: {sorted(allowed_keys)}"
        )

    name = raw.get("name")
    if not name or not str(name).strip():
        raise ValueError("model config object requires non-empty 'name'")

    # Provider is optional - infer from name if not provided
    provider_raw = raw.get("provider")
    if provider_raw is None:
        provider = infer_provider(str(name).strip())
    else:
        try:
            provider = Provider(provider_raw)
        except ValueError as e:
            raise ValueError(
                f"invalid provider '{provider_raw}'; "
                f"must be one of: {[p.value for p in Provider]}"
            ) from e

    fallback_raw = raw.get("fallback")
    fallback: ModelSpec | None = None
    if fallback_raw is not None:
        if not isinstance(fallback_raw, dict):
            raise ValueError("model fallback must be an object")
        if "fallback" in fallback_raw:
            raise ValueError("nested fallback chains are not supported")
        fallback = parse_model_config(fallback_raw)

    return ModelSpec(
        provider=provider,
        name=str(name).strip(),
        fallback=fallback,
    )


def model_spec_cache_key(spec: ModelSpec) -> str:
    """Stable cache identity string for a model spec including fallback."""
    parts = [f"{spec.provider.value}:{spec.name}"]
    if spec.fallback:
        parts.append(f"→{model_spec_cache_key(spec.fallback)}")
    return "".join(parts)
