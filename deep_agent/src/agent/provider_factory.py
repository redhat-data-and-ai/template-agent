"""Unified LLM provider factory for per-agent model resolution.

Routes model creation to Vertex AI or OpenAI-compatible backends based on
an explicit :class:`ModelSpec`, optionally chaining a fallback model via
LangChain's ``with_fallbacks``.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from deep_agent.src.agent.config.model import ModelSpec, Provider
from deep_agent.src.agent.llm import _create_vertex_model, _create_vllm_model
from deep_agent.src.settings import settings


def create_model_from_spec(
    spec: ModelSpec,
    *,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    location: str | None = None,
) -> BaseChatModel:
    """Create a chat model from a :class:`ModelSpec`.

    When ``spec.fallback`` is set, wraps the primary model with
    ``primary.with_fallbacks([secondary])`` so invocation failures on the
    primary route to the secondary model.

    Args:
        spec: Parsed model configuration.
        temperature: Model temperature.
        max_output_tokens: Maximum output tokens (defaults to settings).

    Returns:
        A BaseChatModel instance, optionally with fallback chain.
    """
    tokens = max_output_tokens or settings.MAX_OUTPUT_TOKENS
    primary = _create_by_provider(
        spec.provider,
        spec.name,
        temperature=temperature,
        max_output_tokens=tokens,
        location=location,
    )

    if spec.fallback is None:
        return primary

    secondary = _create_by_provider(
        spec.fallback.provider,
        spec.fallback.name,
        temperature=temperature,
        max_output_tokens=tokens,
        location=location,
    )
    return primary.with_fallbacks([secondary])


def _create_by_provider(
    provider: Provider,
    model_name: str,
    *,
    temperature: float,
    max_output_tokens: int,
    location: str | None = None,
) -> BaseChatModel:
    """Route model creation to the appropriate backend.

    Routes:
    - VERTEX → Google Vertex AI (Gemini, Claude)
    - OPENAI → OpenAI API (GPT models)
    - OPENAI_COMPATIBILITY → any other OpenAI-compatible endpoint

    Both OpenAI routes share the VLLM_BASE_URL / VLLM_API_KEY endpoint
    settings, and model names are sent verbatim — no prefixing or rewriting.
    Deployments that namespace their models (e.g. a serving platform exposing
    ``/data/granite-3.3-8b-instruct``) must spell that out in the config.
    """
    if provider == Provider.VERTEX:
        return _create_vertex_model(
            model_name, temperature, max_output_tokens, location=location
        )
    if provider in (Provider.OPENAI, Provider.OPENAI_COMPATIBILITY):
        return _create_vllm_model(model_name, temperature, max_output_tokens)
    raise ValueError(f"unsupported provider: {provider}")
