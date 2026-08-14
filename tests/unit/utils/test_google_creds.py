"""Unit tests for Google credentials management."""

import json
from unittest.mock import MagicMock, patch

import pytest
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


class TestGetServiceAccountCredentials:
    """Tests for get_service_account_credentials function."""

    def test_successful_credential_loading(self, mock_service_account_info):
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

                # Verify the service account info was parsed correctly
                mock_from_info.assert_called_once()
                call_args = mock_from_info.call_args
                assert call_args[0][0] == mock_service_account_info

    def test_credentials_caching(self, mock_service_account_info):
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
                # First call
                creds1, project1 = get_service_account_credentials()

                # Second call
                creds2, project2 = get_service_account_credentials()

                # Should be the same instances
                assert creds1 is creds2
                assert project1 == project2

                # Should only create credentials once
                assert mock_from_info.call_count == 1

    @pytest.mark.parametrize("creds_content", [None, ""])
    def test_missing_or_empty_credentials(self, creds_content):
        """Test error when credentials are None or empty."""
        with patch("deep_agent.utils.google_creds.settings") as mock_settings:
            mock_settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT = creds_content
            mock_settings.PYTHON_LOG_LEVEL = "INFO"

            with pytest.raises(
                RuntimeError, match="No Google service account credentials configured"
            ):
                get_service_account_credentials()

    def test_invalid_json(self):
        """Test error when credentials content is not valid JSON."""
        with patch("deep_agent.utils.google_creds.settings") as mock_settings:
            mock_settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT = "not valid json {"
            mock_settings.PYTHON_LOG_LEVEL = "INFO"

            with pytest.raises(RuntimeError, match="Invalid JSON in credentials"):
                get_service_account_credentials()

    @pytest.mark.parametrize("action", ["remove", "empty"])
    def test_invalid_project_id(self, mock_service_account_info, action):
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

    def test_clear_cache_allows_reload(self, mock_service_account_info):
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
                # First load
                creds1, _ = get_service_account_credentials()
                assert creds1 is mock_creds1

                # Clear cache
                clear_credentials_cache()

                # Second load should create new credentials
                creds2, _ = get_service_account_credentials()
                assert creds2 is mock_creds2
                assert creds2 is not creds1

                # Should have been called twice
                assert mock_from_info.call_count == 2
