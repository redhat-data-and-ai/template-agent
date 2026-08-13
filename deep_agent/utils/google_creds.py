"""Google credentials management utilities.

This module provides functions for obtaining Google Cloud credentials
for Vertex AI access. Supports Application Default Credentials (ADC)
as the primary method, with inline JSON as a fallback.
"""

import json
import os
from pathlib import Path

import google.auth
import google.auth.exceptions
from google.auth.credentials import Credentials
from google.oauth2 import service_account

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

# Google Cloud authentication scope for Vertex AI
GOOGLE_AUTH_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Cache for credentials to avoid repeated credential fetches
_credentials_cache: tuple[Credentials, str] | None = None


def _project_from_adc_file() -> str | None:
    """Read quota_project_id / project_id from the mounted ADC JSON file."""
    adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not adc_path:
        return None
    try:
        adc_info = json.loads(Path(adc_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    project = adc_info.get("quota_project_id") or adc_info.get("project_id")
    return project if project else None


def _resolve_adc_project(project: str | None) -> str | None:
    """Resolve Vertex project when google.auth.default omits it (user OAuth ADC)."""
    if project:
        return project
    if settings.GOOGLE_CLOUD_PROJECT:
        return settings.GOOGLE_CLOUD_PROJECT
    return _project_from_adc_file()


def get_service_account_credentials() -> tuple[Credentials, str]:
    """Get Google Cloud credentials using ADC or inline JSON.

    Tries credential sources in priority order:
      1. Application Default Credentials (ADC) — discovered automatically
         from GOOGLE_APPLICATION_CREDENTIALS env var, well-known file
         location (~/.config/gcloud/), or GCE metadata server.
      2. Inline JSON from GOOGLE_APPLICATION_CREDENTIALS_CONTENT env var.

    Returns:
        Tuple of (credentials, project_id)

    Raises:
        RuntimeError: If credentials cannot be loaded or project ID is missing
    """
    global _credentials_cache

    if _credentials_cache is not None:
        return _credentials_cache

    # Priority 1: Application Default Credentials
    try:
        credentials, project = google.auth.default(scopes=GOOGLE_AUTH_SCOPES)
        resolved_project = _resolve_adc_project(project)
        if resolved_project:
            logger.info(f"Loaded ADC credentials for project: {resolved_project}")
            _credentials_cache = (credentials, resolved_project)
            return _credentials_cache
    except google.auth.exceptions.DefaultCredentialsError:
        pass

    # Priority 2: Inline JSON from env var (existing behavior)
    if settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT:
        try:
            service_account_info = json.loads(
                settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT
            )
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in credentials: {e}") from e

        project = service_account_info.get("project_id")
        if not project:
            raise RuntimeError(
                "Service account JSON does not contain 'project_id' field"
            )

        credentials = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=GOOGLE_AUTH_SCOPES
        )

        logger.info(
            f"Loaded credentials from GOOGLE_APPLICATION_CREDENTIALS_CONTENT "
            f"for project: {project}"
        )
        _credentials_cache = (credentials, project)
        return _credentials_cache

    raise RuntimeError(
        "No Google credentials found. Either run "
        "'gcloud auth application-default login' "
        "or set GOOGLE_APPLICATION_CREDENTIALS_CONTENT."
    )


def clear_credentials_cache() -> None:
    """Clear the cached Google Cloud credentials.

    Useful for testing or when credentials need to be refreshed.
    """
    global _credentials_cache
    _credentials_cache = None
