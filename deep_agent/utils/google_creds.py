"""Google credentials management utilities.

This module provides functions for obtaining Google Cloud credentials
for Vertex AI access. Prefers inline JSON from
GOOGLE_APPLICATION_CREDENTIALS_CONTENT when set, with Application
Default Credentials (ADC) as the fallback.
"""

import json
import os
from collections.abc import Mapping
from pathlib import Path

import google.auth
import google.auth.exceptions
from google.auth import _cloud_sdk
from google.auth.credentials import Credentials
from google.oauth2 import service_account

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

# Google Cloud authentication scope for Vertex AI
GOOGLE_AUTH_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Cache for credentials to avoid repeated credential fetches
_credentials_cache: tuple[Credentials, str] | None = None

_ERR_INVALID_JSON = "Invalid JSON in credentials"
_ERR_MISSING_PROJECT_ID = "Service account JSON does not contain 'project_id' field"
_ERR_NO_CREDENTIALS = (
    "No Google credentials found. Either run "
    "'gcloud auth application-default login' "
    "or set GOOGLE_APPLICATION_CREDENTIALS_CONTENT."
)


def _well_known_adc_path() -> Path:
    """Return the platform-aware gcloud application-default credentials path.

    Uses google-auth so Windows resolves %APPDATA%/gcloud and CLOUDSDK_CONFIG
    is honored, matching google.auth.default().
    """
    return Path(_cloud_sdk.get_application_default_credentials_path())


def _adc_file_paths() -> list[Path]:
    """Return ADC JSON file paths in lookup order."""
    paths: list[Path] = []
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path:
        paths.append(Path(env_path))
    paths.append(_well_known_adc_path())
    return paths


def _project_from_adc_file() -> str | None:
    """Read quota_project_id / project_id from ADC JSON files."""
    for adc_path in _adc_file_paths():
        if not adc_path.is_file():
            continue
        try:
            adc_info = json.loads(adc_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        project = adc_info.get("quota_project_id") or adc_info.get("project_id")
        if isinstance(project, str) and project:
            return project
    return None


def _resolve_adc_project(project: str | None) -> str | None:
    """Resolve Vertex project when google.auth.default omits it (user OAuth ADC)."""
    if project:
        return project
    if settings.GOOGLE_CLOUD_PROJECT:
        return settings.GOOGLE_CLOUD_PROJECT
    return _project_from_adc_file()


def get_service_account_credentials() -> tuple[Credentials, str]:
    """Get Google Cloud credentials using inline JSON or ADC.

    Tries credential sources in priority order:
      1. Inline JSON from GOOGLE_APPLICATION_CREDENTIALS_CONTENT env var.
         When present, invalid JSON or a missing project_id fails hard
         (no ADC fallback).
      2. Application Default Credentials (ADC) — discovered automatically
         from GOOGLE_APPLICATION_CREDENTIALS env var, the platform well-known
         file location, or GCE metadata server.

    Returns:
        Tuple of (credentials, project_id)

    Raises:
        RuntimeError: If credentials cannot be loaded or project ID is missing
    """
    global _credentials_cache

    if _credentials_cache is not None:
        return _credentials_cache

    # Priority 1: Inline JSON from env var
    if settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT:
        try:
            service_account_info = json.loads(
                settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT
            )
        except json.JSONDecodeError as e:
            raise RuntimeError(f"{_ERR_INVALID_JSON}: {e}") from e

        if not isinstance(service_account_info, Mapping):
            raise RuntimeError(_ERR_INVALID_JSON)

        project = service_account_info.get("project_id")
        if not project:
            raise RuntimeError(_ERR_MISSING_PROJECT_ID)

        credentials = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=GOOGLE_AUTH_SCOPES
        )

        logger.info(
            "Loaded credentials from GOOGLE_APPLICATION_CREDENTIALS_CONTENT "
            "for project: %s",
            project,
        )
        _credentials_cache = (credentials, project)
        return _credentials_cache

    # Priority 2: Application Default Credentials
    try:
        credentials, project = google.auth.default(scopes=GOOGLE_AUTH_SCOPES)
        resolved_project = _resolve_adc_project(project)
        if resolved_project:
            logger.info("Loaded ADC credentials for project: %s", resolved_project)
            _credentials_cache = (credentials, resolved_project)
            return _credentials_cache
    except google.auth.exceptions.DefaultCredentialsError:
        pass

    raise RuntimeError(_ERR_NO_CREDENTIALS)


def clear_credentials_cache() -> None:
    """Clear the cached Google Cloud credentials.

    Useful for testing or when credentials need to be refreshed.
    """
    global _credentials_cache
    _credentials_cache = None
