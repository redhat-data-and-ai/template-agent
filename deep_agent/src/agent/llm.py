"""LLM factory for creating configured model instances.

Supports three provider paths:
  1. Gemini (via langchain_google_genai + Vertex AI service account)
  2. Claude (via langchain_google_vertexai Model Garden)
  3. vLLM / OpenAI-compatible (via langchain_openai + custom base_url)

Any model name not in GEMINI_MODELS or CLAUDE_MODELS is assumed to be
served by a vLLM (or OpenAI-compatible) endpoint. Set VLLM_BASE_URL
to the inference server's /v1 endpoint.
"""

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from deep_agent.src.error_handling import llm_retry
from deep_agent.src.exceptions import LLMError
from deep_agent.src.settings import settings
from deep_agent.utils.google_creds import get_service_account_credentials
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

_DEFAULT_MAX_OUTPUT_TOKENS: int = settings.MAX_OUTPUT_TOKENS

GEMINI_MODELS: list[str] = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3.1-pro-preview",
]

CLAUDE_MODELS: list[str] = [
    "claude-sonnet-4",
    "claude-sonnet-4-6@default",
]


@llm_retry
def create_model(
    model_name: str,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
) -> BaseChatModel:
    """Create a model instance (Vertex AI, or vLLM/OpenAI-compatible).

    Resolution order:
      1. If model_name is in GEMINI_MODELS → Vertex AI Gemini
      2. If model_name is in CLAUDE_MODELS → Vertex AI Claude (Model Garden)
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

    is_gemini = model_name in GEMINI_MODELS
    is_claude = model_name in CLAUDE_MODELS

    if is_gemini or is_claude:
        return _create_vertex_model(model_name, temperature, max_output_tokens)

    return _create_vllm_model(model_name, temperature, max_output_tokens)


def _create_vertex_model(
    model_name: str,
    temperature: float,
    max_output_tokens: int,
) -> BaseChatModel:
    """Create a Vertex AI model (Gemini or Claude)."""
    is_claude = model_name in CLAUDE_MODELS
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
                credentials=credentials,
                temperature=temperature,
                max_tokens=max_output_tokens,
                max_retries=2,
            )
        else:
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                credentials=credentials,
                project=project,
                max_output_tokens=max_output_tokens,
                max_retries=2,
            )

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
            f"Model '{model_name}' is not a known Vertex AI model. "
            f"Set VLLM_BASE_URL to use it via an OpenAI-compatible endpoint. "
            f"Known Vertex AI models: {GEMINI_MODELS + CLAUDE_MODELS}"
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
