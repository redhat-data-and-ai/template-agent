"""Unit tests for LLM model configuration and initialization."""

from unittest.mock import MagicMock, patch

import pytest

from template_agent.src.agent.llm import CLAUDE_MODELS, GEMINI_MODELS, create_model


class TestCreateModel:
    """Tests for create_model function."""

    def test_create_gemini_model(self):
        """Test creating Gemini model."""
        mock_creds = MagicMock()

        with patch(
            "template_agent.src.agent.llm.get_service_account_credentials"
        ) as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-project")

            with patch(
                "template_agent.src.agent.llm.ChatGoogleGenerativeAI"
            ) as mock_chat:
                create_model("gemini-2.5-pro", temperature=0.5)
                mock_chat.assert_called_once_with(
                    model="gemini-2.5-pro",
                    temperature=0.5,
                    credentials=mock_creds,
                    project="test-project",
                    max_retries=2,
                )

    def test_create_claude_model(self):
        """Test creating Claude model."""
        mock_creds = MagicMock()

        with patch(
            "template_agent.src.agent.llm.get_service_account_credentials"
        ) as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-project")

            with patch("template_agent.src.agent.llm.ChatAnthropicVertex") as mock_chat:
                create_model("claude-sonnet-4", temperature=0.7)
                mock_chat.assert_called_once_with(
                    model="claude-sonnet-4",
                    project="test-project",
                    credentials=mock_creds,
                    temperature=0.7,
                    max_retries=2,
                )

    @pytest.mark.parametrize(
        "invalid_name",
        ["", "   ", None],
    )
    def test_invalid_model_name_raises_error(self, invalid_name):
        """Test that empty/whitespace/None model names raise ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            create_model(invalid_name)

    def test_unknown_model_raises_error_with_supported_list(self):
        """Test that unknown model raises error listing supported models."""
        mock_creds = MagicMock()

        with patch(
            "template_agent.src.agent.llm.get_service_account_credentials"
        ) as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-project")

            with pytest.raises(ValueError) as exc_info:
                create_model("gpt-4")

            error_msg = str(exc_info.value)
            assert "Unknown model 'gpt-4'" in error_msg
            assert "Supported models:" in error_msg

    def test_model_creation_errors_are_raised(self):
        """Test that model creation errors are raised."""
        mock_creds = MagicMock()

        with patch(
            "template_agent.src.agent.llm.get_service_account_credentials"
        ) as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-project")

            with patch(
                "template_agent.src.agent.llm.ChatGoogleGenerativeAI",
                side_effect=RuntimeError("API error"),
            ):
                with pytest.raises(RuntimeError, match="API error"):
                    create_model("gemini-2.5-pro")

    def test_all_supported_models_work(self):
        """Test that all models in GEMINI_MODELS and CLAUDE_MODELS are supported."""
        mock_creds = MagicMock()

        with patch(
            "template_agent.src.agent.llm.get_service_account_credentials"
        ) as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-project")

            with patch("template_agent.src.agent.llm.ChatGoogleGenerativeAI"):
                for model_name in GEMINI_MODELS:
                    create_model(model_name)

            with patch("template_agent.src.agent.llm.ChatAnthropicVertex"):
                for model_name in CLAUDE_MODELS:
                    create_model(model_name)
