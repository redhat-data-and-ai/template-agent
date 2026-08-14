"""Google credentials management utilities.

This module provides functions for initializing Google Generative AI with
service account credentials from environment variables.
"""

import json

from google.auth.credentials import Credentials
from google.oauth2 import service_account

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

# Google Cloud authentication scope for Vertex AI
GOOGLE_AUTH_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Cache for credentials to avoid repeated credential fetches
_credentials_cache: tuple[Credentials, str] | None = None


def get_service_account_credentials() -> tuple[Credentials, str]:
    """Get Google Cloud credentials from service account JSON.

    Reads service account JSON from GOOGLE_APPLICATION_CREDENTIALS_CONTENT
    environment variable and creates credentials. Uses caching to avoid
    repeated credential fetches.

    Returns:
        Tuple of (credentials, project_id)

    Raises:
        RuntimeError: If credentials cannot be loaded or project ID is missing
    """
    global _credentials_cache

    if _credentials_cache is not None:
        return _credentials_cache

    if not settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT:
        raise RuntimeError("No Google service account credentials configured")

    try:
        service_account_info = json.loads(
            settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT
        )
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in credentials: {e}")
        raise RuntimeError(f"Invalid JSON in credentials: {e}") from e

    project = service_account_info.get("project_id")
    if not project:
        raise RuntimeError("Service account JSON does not contain 'project_id' field")

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=GOOGLE_AUTH_SCOPES
    )

    logger.info(f"Loaded Google credentials for project: {project}")
    _credentials_cache = (credentials, project)
    return _credentials_cache


def clear_credentials_cache() -> None:
    """Clear the cached Google Cloud credentials.

    Useful for testing or when credentials need to be refreshed.
    """
    global _credentials_cache
    _credentials_cache = None
