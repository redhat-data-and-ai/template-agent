"""Phase 1 configuration.

Published model ids match the guardrail specification. The default backend is
``builtin`` so tests and offline evaluation run without downloading weights.
Set ``backend`` to ``huggingface`` to load the published checkpoints.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HAP_MODEL_ID = "ibm-granite/granite-guardian-hap-125m"
HAP_FALLBACK_MODEL_ID = "classification_transformer_en_slate.38m.hap"
PROMPT_GUARD_MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
SAFETY_MODEL_ID = "hivetrace/gliner-guard-uniencoder"
BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"

Backend = Literal["builtin", "huggingface"]
UnavailablePolicy = Literal["deny", "builtin"]


class BudgetConfig(BaseModel):
    """Wall-clock budget for the input gate. A model past the budget denies."""

    input_ms: float = 100


class RulesConfig(BaseModel):
    """Code rules for length and permitted characters. No model."""

    enabled: bool = True
    max_input_chars: int = 8000
    permitted_characters: str | None = None


class SchemaConfig(BaseModel):
    """Valid fields for a structured query. Free text is not a query.

    When ``allowed_fields`` or ``required_fields`` is non-empty, a message
    that is a JSON object is checked. Other messages pass this check.
    """

    enabled: bool = True
    allowed_fields: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)


class ClassifierConfig(BaseModel):
    """A thresholded text classifier."""

    enabled: bool = True
    backend: Backend = "builtin"
    model_id: str
    threshold: float = 0.5
    on_unavailable: UnavailablePolicy = "builtin"


class ContentFilterConfig(ClassifierConfig):
    """Hate, abuse, and profanity. Block at probability 0.5 or higher."""

    model_id: str = HAP_MODEL_ID
    fallback_model_id: str | None = HAP_FALLBACK_MODEL_ID


class InjectionConfig(ClassifierConfig):
    """Prompt injection and jailbreak detection. User messages only."""

    model_id: str = PROMPT_GUARD_MODEL_ID


class SafetyConfig(ClassifierConfig):
    """Secondary moderation vote: violence, sexual content, self-harm, crime."""

    model_id: str = SAFETY_MODEL_ID
    labels: tuple[str, ...] = ("violence", "sexual", "self_harm", "crime")


class PIIConfig(BaseModel):
    """Mask PII and continue. ``auto`` uses Presidio when it is installed."""

    enabled: bool = True
    backend: Literal["pattern", "presidio", "auto"] = "auto"
    spacy_model: str = "en_core_web_sm"


class SecretConfig(BaseModel):
    """Regex masking for keys, tokens, and private-key blocks."""

    enabled: bool = True
    mask: str = "[SECRET]"


class IntentConfig(BaseModel):
    """Embedding plus logistic regression.

    ``bge`` is the published encoder. ``hashing`` is the portable encoder
    with the same classifier head, used when that model is not installed.
    Confidence below ``min_confidence`` restricts tools to ``read_only_tools``.
    An unfitted head does not narrow tools.
    """

    enabled: bool = True
    embedding: Literal["hashing", "bge"] = "hashing"
    model_id: str = BGE_MODEL_ID
    min_confidence: float = 0.6
    tools_for_label: dict[str, list[str]] = Field(default_factory=dict)
    read_only_tools: list[str] = Field(default_factory=list)


class InputGateConfig(BaseModel):
    """Configuration for Phase 1 only.

    There is no human-approval switch. The gate returns allow, block, or
    redact and never waits.
    """

    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    schema_check: SchemaConfig = Field(default_factory=SchemaConfig)
    content_filter: ContentFilterConfig = Field(default_factory=ContentFilterConfig)
    injection: InjectionConfig = Field(default_factory=InjectionConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    pii: PIIConfig = Field(default_factory=PIIConfig)
    secrets: SecretConfig = Field(default_factory=SecretConfig)
    intent: IntentConfig = Field(default_factory=IntentConfig)
