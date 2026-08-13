"""Unit tests for Google credentials management."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2 import service_account

from deep_agent.utils.google_creds import (
    clear_credentials_cache,
    get_service_account_credentials,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear credentials cache before and after each test."""
    clear_credentials_cache()
    yield
    clear_credentials_cache()


@pytest.fixture
def mock_service_account_info():
    """Fixture providing valid service account JSON."""
    return {
        "type": "service_account",
        "project_id": "test-project-123",
        "private_key_id": "key123",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMOCK_KEY\n-----END PRIVATE KEY-----",
        "client_email": "test@test-project-123.iam.gserviceaccount.com",
        "client_id": "123456789",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }


@pytest.fixture
def mock_adc_failure():
    """Patch ADC lookup to fail so inline JSON tests exercise the fallback path."""
    with patch(
        "deep_agent.utils.google_creds.google.auth.default",
        side_effect=DefaultCredentialsError(),
    ):
        yield


class TestGetServiceAccountCredentials:
    """Tests for get_service_account_credentials function."""

    def test_adc_credential_loading(self):
        """Test loading credentials via Application Default Credentials."""
        mock_creds = MagicMock()

        with patch(
            "deep_agent.utils.google_creds.google.auth.default",
            return_value=(mock_creds, "adc-project-456"),
        ):
            with patch("deep_agent.utils.google_creds.settings") as mock_settings:
                mock_settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT = None
                mock_settings.GOOGLE_CLOUD_PROJECT = None
                mock_settings.PYTHON_LOG_LEVEL = "INFO"

                credentials, project = get_service_account_credentials()

                assert credentials is mock_creds
                assert project == "adc-project-456"

    def test_adc_user_oauth_with_quota_project_from_file(self, tmp_path):
        """User OAuth ADC often returns project=None; resolve from ADC file."""
        mock_creds = MagicMock()
        adc_file = tmp_path / "application_default_credentials.json"
        adc_file.write_text(
            json.dumps(
                {
                    "type": "authorized_user",
                    "quota_project_id": "quota-project-789",
                }
            ),
            encoding="utf-8",
        )

        with patch(
            "deep_agent.utils.google_creds.google.auth.default",
            return_value=(mock_creds, None),
        ):
            with patch.dict(
                os.environ,
                {"GOOGLE_APPLICATION_CREDENTIALS": str(adc_file)},
                clear=False,
            ):
                with patch("deep_agent.utils.google_creds.settings") as mock_settings:
                    mock_settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT = None
                    mock_settings.GOOGLE_CLOUD_PROJECT = None
                    mock_settings.PYTHON_LOG_LEVEL = "INFO"

                    credentials, project = get_service_account_credentials()

                    assert credentials is mock_creds
                    assert project == "quota-project-789"

    def test_successful_credential_loading(
        self, mock_service_account_info, mock_adc_failure
    ):
        """Test successful loading of credentials from valid JSON."""
        mock_creds = MagicMock(spec=service_account.Credentials)

        with patch("deep_agent.utils.google_creds.settings") as mock_settings:
            mock_settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT = json.dumps(
                mock_service_account_info
            )
            mock_settings.PYTHON_LOG_LEVEL = "INFO"

            with patch(
                "deep_agent.utils.google_creds.service_account.Credentials.from_service_account_info",
                return_value=mock_creds,
            ) as mock_from_info:
                credentials, project = get_service_account_credentials()

                assert credentials == mock_creds
                assert project == "test-project-123"

                mock_from_info.assert_called_once()
                call_args = mock_from_info.call_args
                assert call_args[0][0] == mock_service_account_info

    def test_credentials_caching(self, mock_service_account_info, mock_adc_failure):
        """Test that credentials are cached after first call."""
        mock_creds = MagicMock(spec=service_account.Credentials)

        with patch("deep_agent.utils.google_creds.settings") as mock_settings:
            mock_settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT = json.dumps(
                mock_service_account_info
            )
            mock_settings.PYTHON_LOG_LEVEL = "INFO"

            with patch(
                "deep_agent.utils.google_creds.service_account.Credentials.from_service_account_info",
                return_value=mock_creds,
            ) as mock_from_info:
                creds1, project1 = get_service_account_credentials()
                creds2, project2 = get_service_account_credentials()

                assert creds1 is creds2
                assert project1 == project2
                assert mock_from_info.call_count == 1

    @pytest.mark.parametrize("creds_content", [None, ""])
    def test_missing_or_empty_credentials(self, creds_content):
        """Test error when ADC and inline credentials are unavailable."""
        with patch(
            "deep_agent.utils.google_creds.google.auth.default",
            side_effect=DefaultCredentialsError(),
        ):
            with patch("deep_agent.utils.google_creds.settings") as mock_settings:
                mock_settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT = creds_content
                mock_settings.PYTHON_LOG_LEVEL = "INFO"

                with pytest.raises(RuntimeError, match="No Google credentials found"):
                    get_service_account_credentials()

    def test_invalid_json(self, mock_adc_failure):
        """Test error when credentials content is not valid JSON."""
        with patch("deep_agent.utils.google_creds.settings") as mock_settings:
            mock_settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT = "not valid json {"
            mock_settings.PYTHON_LOG_LEVEL = "INFO"

            with pytest.raises(RuntimeError, match="Invalid JSON in credentials"):
                get_service_account_credentials()

    @pytest.mark.parametrize("action", ["remove", "empty"])
    def test_invalid_project_id(
        self, mock_service_account_info, action, mock_adc_failure
    ):
        """Test error when project_id is missing or empty."""
        if action == "remove":
            mock_service_account_info.pop("project_id")
        else:
            mock_service_account_info["project_id"] = ""

        with patch("deep_agent.utils.google_creds.settings") as mock_settings:
            mock_settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT = json.dumps(
                mock_service_account_info
            )
            mock_settings.PYTHON_LOG_LEVEL = "INFO"

            with pytest.raises(
                RuntimeError,
                match="Service account JSON does not contain 'project_id' field",
            ):
                get_service_account_credentials()

    def test_clear_cache_allows_reload(self, mock_service_account_info, mock_adc_failure):
        """Test that clearing cache allows credentials to be reloaded."""
        mock_creds1 = MagicMock(spec=service_account.Credentials)
        mock_creds2 = MagicMock(spec=service_account.Credentials)

        with patch("deep_agent.utils.google_creds.settings") as mock_settings:
            mock_settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT = json.dumps(
                mock_service_account_info
            )
            mock_settings.PYTHON_LOG_LEVEL = "INFO"

            with patch(
                "deep_agent.utils.google_creds.service_account.Credentials.from_service_account_info",
                side_effect=[mock_creds1, mock_creds2],
            ) as mock_from_info:
                creds1, _ = get_service_account_credentials()
                assert creds1 is mock_creds1

                clear_credentials_cache()

                creds2, _ = get_service_account_credentials()
                assert creds2 is mock_creds2
                assert creds2 is not creds1
                assert mock_from_info.call_count == 2
