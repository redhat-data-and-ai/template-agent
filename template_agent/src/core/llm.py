"""Model configuration and initialization utilities.

This module provides utilities for configuring and initializing language models
with consistent settings across the application using Google Vertex AI.
"""

from typing import Union

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from template_agent.src.settings import settings
from template_agent.utils.google_creds import get_service_account_credentials
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

# Supported Gemini models on Vertex AI
GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.1-pro-preview"]

# Supported Claude models on Vertex AI
CLAUDE_MODELS = [
    "claude-sonnet-4",
]


def create_model(
    model_name: str,
    temperature: float = 0.0,
) -> Union[ChatGoogleGenerativeAI, ChatAnthropicVertex]:
    """Create a Vertex AI model (Gemini or Claude).

    Args:
        model_name: Model name from GEMINI_MODELS or CLAUDE_MODELS
        temperature: Model temperature (default: 0.0)

    Returns:
        Configured model instance
    """
    if not model_name or not model_name.strip():
        raise ValueError("model_name cannot be empty")

    credentials, project = get_service_account_credentials()

    # Detect model type based on model lists
    is_claude = model_name in CLAUDE_MODELS
    is_gemini = model_name in GEMINI_MODELS

    if not is_claude and not is_gemini:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported models: {GEMINI_MODELS + CLAUDE_MODELS}"
        )

    model_type = "Claude" if is_claude else "Gemini"

    try:
        logger.info(
            f"Creating {model_type} model via Vertex AI",
            model=model_name,
            model_type=model_type,
            project=project,
            temperature=temperature,
        )

        if is_claude:
            # Use ChatAnthropicVertex for Claude models
            return ChatAnthropicVertex(
                model=model_name,
                project=project,
                credentials=credentials,
                temperature=temperature,
                max_retries=2,
            )
        else:
            # Use ChatGoogleGenerativeAI for Gemini models
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                credentials=credentials,
                project=project,
                max_retries=2,
            )

    except Exception as e:
        logger.error(
            f"Failed to create {model_type} model '{model_name}'",
            error_type=type(e).__name__,
            model=model_name,
            model_type=model_type,
            project=project,
            error_message=str(e),
            exc_info=True,
        )
        raise
