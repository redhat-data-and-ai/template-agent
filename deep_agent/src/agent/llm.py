"""LLM factory for creating configured model instances.

Supports three provider paths:
  1. Gemini (via langchain_google_genai + Vertex AI service account)
  2. Claude (via langchain_google_vertexai Model Garden)
  3. vLLM / OpenAI-compatible (via langchain_openai + custom base_url)

Any model name that is not a Gemini or Claude name is assumed to be
served by a vLLM (or OpenAI-compatible) endpoint. Set VLLM_BASE_URL
to the inference server's /v1 endpoint.
"""

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from deep_agent.src.error_handling import llm_retry
from deep_agent.src.exceptions import LLMError
from deep_agent.src.settings import settings
from deep_agent.utils.google_creds import get_service_account_credentials
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

_DEFAULT_MAX_OUTPUT_TOKENS: int = settings.MAX_OUTPUT_TOKENS

# Provider is decided by name prefix, not by an allowlist.
#
# There used to be explicit GEMINI_MODELS / CLAUDE_MODELS lists. A list
# of model names is a snapshot of the day it was written: every model
# released afterwards fell through to the vLLM branch and failed with a
# connection error to an endpoint nobody had configured, which reads as
# an infrastructure problem rather than "that model is not in a list".
# Naming is the one thing these providers are consistent about.
# The prefix lists live in settings, not here — a new model family
# should not need a code change to route correctly.


def is_vertex_model(model_name: str) -> bool:
    """True for Vertex-hosted Gemini and Claude names."""
    return model_name.lower().startswith(settings.vertex_model_prefixes)


def is_claude_model(model_name: str) -> bool:
    """True for Claude on Vertex Model Garden, which has its own client."""
    return model_name.lower().startswith(settings.claude_model_prefixes)


def is_openai_model(model_name: str) -> bool:
    """Does this model name belong to OpenAI rather than another provider?"""
    return model_name.lower().startswith(settings.openai_model_prefixes)


_THRESHOLD_MAP = {v.name: v for v in HarmBlockThreshold}

safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: _THRESHOLD_MAP[
        settings.SAFETY_DANGEROUS_CONTENT
    ],
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: _THRESHOLD_MAP[settings.SAFETY_HATE_SPEECH],
    HarmCategory.HARM_CATEGORY_HARASSMENT: _THRESHOLD_MAP[settings.SAFETY_HARASSMENT],
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: _THRESHOLD_MAP[
        settings.SAFETY_SEXUALLY_EXPLICIT
    ],
}


def _thinking_budget_for(model_name: str, max_output_tokens: int | None) -> int | None:
    """Cap reasoning tokens so there is room left to answer.

    Found against real Vertex: ``gemini-2.5-pro`` with the shipped
    ``max_output_tokens: 8192`` and no budget returns **zero content
    chunks** and no error. The run completes "successfully" with an
    empty message, which is indistinguishable from the model having
    nothing to say — and 2.5-pro is the model the shipped PROMPT.md
    names, so this was the out-of-the-box experience.

    Which models reason, and how much of the allowance they may
    spend, are in settings (``THINKING_MODEL_PREFIXES``,
    ``THINKING_BUDGET_SHARE``, ``MIN_THINKING_BUDGET``).

    One correction worth recording: this used to say "Flash does not
    reason by default and is left alone", which was true of Gemini
    2.5 Flash and false of 3.x — and the prefix list said ``gemini-3``,
    so Flash was getting a budget while the comment claimed it was
    not. The behaviour was right; the comment was wrong.
    """
    # Lowercased, like the routing prefix match. `Gemini-2.5-pro`
    # otherwise matched nothing, got no budget, and returned the
    # empty reply this function exists to prevent.
    prefixes = settings.thinking_model_prefixes
    lowered = model_name.lower()
    if not any(lowered.startswith(prefix.lower()) for prefix in prefixes):
        return None
    if not max_output_tokens:
        return None

    budget = max(
        settings.MIN_THINKING_BUDGET,
        int(max_output_tokens * settings.THINKING_BUDGET_SHARE),
    )
    # The floor can exceed a small allowance — 512 against
    # max_output_tokens=128 leaves nothing to answer with, which is the
    # same empty-reply failure this function exists to prevent, just
    # reached from the other side. Never take more than half.
    return min(budget, max_output_tokens // 2)


@llm_retry
def create_model(
    model_name: str,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    location: str | None = None,
) -> BaseChatModel:
    """Create a model instance (Vertex AI, or vLLM/OpenAI-compatible).

    Resolution order:
      1. If model_name starts with ``gemini-`` → Vertex AI Gemini
      2. If model_name starts with ``claude-`` → Vertex AI Claude (Model Garden)
      3. Otherwise → vLLM / OpenAI-compatible endpoint (requires VLLM_BASE_URL)

    Args:
        model_name: Model identifier (Gemini/Claude name, or vLLM model path).
        temperature: Model temperature (default: 0.0).
        max_output_tokens: Maximum tokens in model response (default: 8192).

    Returns:
        Configured model instance.

    Raises:
        ValueError: If model_name is empty or vLLM is needed but not configured.
        LLMError: If model creation fails after retries.
    """
    if not model_name or not model_name.strip():
        raise ValueError("model_name cannot be empty")

    max_output_tokens = max_output_tokens or _DEFAULT_MAX_OUTPUT_TOKENS

    is_gemini = is_vertex_model(model_name) and not is_claude_model(model_name)
    is_claude = is_claude_model(model_name)

    if is_gemini or is_claude:
        return _create_vertex_model(
            model_name, temperature, max_output_tokens, location=location
        )

    return _create_vllm_model(model_name, temperature, max_output_tokens)


def _create_vertex_model(
    model_name: str,
    temperature: float,
    max_output_tokens: int,
    location: str | None = None,
) -> BaseChatModel:
    """Create a Vertex AI model (Gemini or Claude)."""
    is_claude = is_claude_model(model_name)
    model_type = "Claude" if is_claude else "Gemini"

    try:
        credentials, project = get_service_account_credentials()

        logger.info(
            f"Creating {model_type} model via Vertex AI",
            model=model_name,
            project=project,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        if is_claude:
            return ChatAnthropicVertex(
                model=model_name,
                project=project,
                location=location or settings.CLAUDE_LOCATION,
                credentials=credentials,
                # temperature=temperature, # this is not supported in the latest version of the library
                max_tokens=max_output_tokens,
                max_retries=2,
                streaming=True,
            )
        else:
            kwargs: dict = {
                "model": model_name,
                "temperature": temperature,
                "credentials": credentials,
                "project": project,
                "max_output_tokens": max_output_tokens,
                "max_retries": 2,
                "streaming": True,
                "safety_settings": safety_settings,
            }

            budget = _thinking_budget_for(model_name, max_output_tokens)
            if budget is not None:
                kwargs["thinking_budget"] = budget

            return ChatGoogleGenerativeAI(**kwargs)

    except (ValueError, LLMError):
        raise
    except Exception as e:
        logger.error(
            f"Failed to create {model_type} model '{model_name}'",
            error_type=type(e).__name__,
            model=model_name,
            error_message=str(e),
            exc_info=True,
        )
        raise LLMError(
            f"Failed to create {model_type} model '{model_name}': {e}"
        ) from e


def _create_vllm_model(
    model_name: str,
    temperature: float,
    max_output_tokens: int,
) -> BaseChatModel:
    """Create a model via vLLM / OpenAI-compatible endpoint.

    vLLM, TGI, Ollama, and any server exposing /v1/chat/completions works.
    """
    if not settings.VLLM_BASE_URL:
        raise ValueError(
            f"Model '{model_name}' is not a Vertex AI model name "
            f"(those start with {', '.join(settings.vertex_model_prefixes)}). "
            f"Set VLLM_BASE_URL to serve it from an OpenAI-compatible endpoint."
        )

    try:
        from langchain_openai import ChatOpenAI

        logger.info(
            "Creating model via vLLM/OpenAI-compatible endpoint",
            model=model_name,
            base_url=settings.VLLM_BASE_URL,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        return ChatOpenAI(
            model=model_name,
            base_url=settings.VLLM_BASE_URL,
            api_key=settings.VLLM_API_KEY,
            temperature=temperature,
            max_tokens=max_output_tokens,
            max_retries=2,
            streaming=True,
        )

    except ImportError:
        raise LLMError(
            "langchain-openai is required for vLLM support. "
            "Add 'langchain-openai' to your dependencies."
        )
    except Exception as e:
        logger.error(
            f"Failed to create vLLM model '{model_name}'",
            error_type=type(e).__name__,
            model=model_name,
            base_url=settings.VLLM_BASE_URL,
            error_message=str(e),
            exc_info=True,
        )
        raise LLMError(f"Failed to create vLLM model '{model_name}': {e}") from e
