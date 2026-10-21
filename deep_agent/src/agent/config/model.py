"""Model configuration types for per-agent LLM provider selection.

Parses frontmatter ``model:`` fields that may be a legacy string or an
object with explicit provider, model name, and optional fallback chain.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, model_validator

from deep_agent.src.agent.llm import CLAUDE_MODELS, GEMINI_MODELS

# Alternate spellings accepted in config, mapped to their canonical value.
# Keys are compared lowercased with surrounding whitespace stripped.
_PROVIDER_ALIASES: dict[str, str] = {
    "vertex_ai": "vertex",
    "vertex-ai": "vertex",
}


class Provider(str, Enum):
    """Supported LLM provider backends.

    Alternate spellings in :data:`_PROVIDER_ALIASES` resolve to the canonical
    member, so ``Provider("vertex_ai") is Provider.VERTEX``. Normalising here
    rather than at the parse site keeps ``value``, cache keys, and equality
    checks canonical no matter which spelling the config used.
    """

    VERTEX = "vertex"
    OPENAI = "openai"
    # Any other endpoint speaking the OpenAI chat API: vLLM, TGI, Ollama, or a
    # managed open-source serving platform.
    OPENAI_COMPATIBILITY = "openai-compatibility"

    @classmethod
    def _missing_(cls, value: object) -> Provider | None:
        if not isinstance(value, str):
            return None
        canonical = _PROVIDER_ALIASES.get(value.strip().lower())
        if canonical is None:
            return None
        return cls._value2member_map_[canonical]  # type: ignore[return-value]


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


def supported_provider_values() -> list[str]:
    """Every accepted ``provider:`` spelling: canonical values plus aliases."""
    return sorted({p.value for p in Provider} | set(_PROVIDER_ALIASES))


def _declared_provider(model_name: str) -> Provider | None:
    """Provider declared for this model in ``models/*.yaml``, if resolvable.

    Imported lazily because the loader imports :class:`Provider` from this
    module; a module-level import would be circular. Config problems must not
    turn provider inference into a hard failure, so any error degrades to
    "no declaration" and lets the caller fall through to the default.
    """
    try:
        from deep_agent.src.agent.config.loader import agent_config

        declared = agent_config.get_declared_provider(model_name)
    except Exception:  # pragma: no cover - defensive
        return None
    if declared is None:
        return None
    try:
        return Provider(declared)
    except ValueError:  # pragma: no cover - loader already canonicalises
        return None


def infer_provider(model_name: str) -> Provider:
    """Infer the provider for a frontmatter ``model:`` that declares none.

    Resolution order:
    1. The provider declared alongside this model in ``models/*.yaml``
    2. Known Gemini/Claude model names → VERTEX
    3. GPT models (gpt-*, case-insensitive) → OPENAI
    4. OPENAI_COMPATIBILITY for everything else

    Steps 2-4 only run when neither the frontmatter nor ``models/*.yaml``
    declares a provider. The known-model lists are the discriminator there:
    a name on them is served by Vertex, and anything else is assumed to come
    from the OpenAI-compatible endpoint at VLLM_BASE_URL.
    """
    declared = _declared_provider(model_name)
    if declared is not None:
        return declared
    if model_name in GEMINI_MODELS or model_name in CLAUDE_MODELS:
        return Provider.VERTEX
    if model_name.lower().startswith("gpt-"):
        return Provider.OPENAI
    return Provider.OPENAI_COMPATIBILITY


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
        raise TypeError(f"model config must be str or dict, got {type(raw).__name__}")

    allowed_keys = {"provider", "name", "fallback"}
    unknown = set(raw.keys()) - allowed_keys
    if unknown:
        raise ValueError(
            f"unknown model config keys: {sorted(unknown)}; "
            f"allowed: {sorted(allowed_keys)}"
        )

    name_raw = raw.get("name")
    if not isinstance(name_raw, str) or not name_raw.strip():
        raise ValueError("model config object requires non-empty 'name'")
    name = name_raw.strip()

    # Provider is optional - infer from name if not provided
    provider_raw = raw.get("provider")
    if provider_raw is None:
        provider = infer_provider(name)
    else:
        try:
            provider = Provider(provider_raw)
        except ValueError as e:
            raise ValueError(
                f"invalid provider '{provider_raw}'; "
                f"must be one of: {supported_provider_values()}"
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
