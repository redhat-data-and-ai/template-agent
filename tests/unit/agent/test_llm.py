"""Unit tests for LLM model configuration and initialization."""

from unittest.mock import MagicMock, patch

from deep_agent.src.settings import settings

import pytest

from deep_agent.src.agent.llm import (
    create_model,
    safety_settings,
)
from deep_agent.src.exceptions import LLMError


class TestCreateModel:
    """Tests for create_model function."""

    def test_create_gemini_model(self):
        """Test creating Gemini model."""
        mock_creds = MagicMock()

        with patch(
            "deep_agent.src.agent.llm.get_service_account_credentials"
        ) as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-project")

            with patch("deep_agent.src.agent.llm.ChatGoogleGenerativeAI") as mock_chat:
                create_model("gemini-2.5-pro", temperature=0.5)
                mock_chat.assert_called_once_with(
                    model="gemini-2.5-pro",
                    temperature=0.5,
                    credentials=mock_creds,
                    project="test-project",
                    max_output_tokens=8192,
                    max_retries=2,
                    streaming=True,
                    safety_settings=safety_settings,
                    # 2.5-pro reasons before answering and its thinking
                    # tokens come out of max_output_tokens. Unbounded, it
                    # spends the lot and returns an empty message.
                    thinking_budget=2048,
                )

    def test_create_claude_model(self):
        """Test creating Claude model."""
        mock_creds = MagicMock()

        with patch(
            "deep_agent.src.agent.llm.get_service_account_credentials"
        ) as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-project")

            with patch("deep_agent.src.agent.llm.ChatAnthropicVertex") as mock_chat:
                create_model("claude-sonnet-4", temperature=0.7)
                mock_chat.assert_called_once_with(
                    model="claude-sonnet-4",
                    project="test-project",
                    location="us-east5",
                    credentials=mock_creds,
                    max_tokens=8192,
                    max_retries=2,
                    streaming=True,
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

        # Override the one field under test rather than replacing the
        # whole settings object: routing now reads its prefix lists
        # from settings too, and a MagicMock stands in for those as a
        # MagicMock, which `startswith` rejects.
        with (
            patch(
                "deep_agent.src.agent.llm.get_service_account_credentials"
            ) as mock_get_creds,
            patch.object(settings, "VLLM_BASE_URL", ""),
        ):
            mock_get_creds.return_value = (mock_creds, "test-project")

            with pytest.raises(ValueError) as exc_info:
                create_model("gpt-4")

            error_msg = str(exc_info.value)
            assert "gpt-4" in error_msg
            assert "is not a Vertex AI model name" in error_msg

    def test_model_creation_errors_are_raised(self):
        """Test that model creation errors are raised."""
        mock_creds = MagicMock()

        with patch(
            "deep_agent.src.agent.llm.get_service_account_credentials"
        ) as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-project")

            with patch(
                "deep_agent.src.agent.llm.ChatGoogleGenerativeAI",
                side_effect=RuntimeError("API error"),
            ):
                with pytest.raises(LLMError, match="API error"):
                    create_model("gemini-2.5-pro")

    def test_vertex_models_route_by_name_not_by_allowlist(self):
        """A model released after this code was written must still work.

        There used to be explicit GEMINI_MODELS / CLAUDE_MODELS lists,
        so anything newer fell through to the vLLM branch and failed
        with a connection error to an endpoint nobody had configured.
        The names deliberately include ones that did not exist when the
        lists did.
        """
        mock_creds = MagicMock()

        with patch(
            "deep_agent.src.agent.llm.get_service_account_credentials"
        ) as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-project")

            with patch("deep_agent.src.agent.llm.ChatGoogleGenerativeAI") as gemini:
                for model_name in (
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-3.1-pro-preview",
                    "gemini-3.1-flash",
                    "gemini-9-not-invented-yet",
                ):
                    create_model(model_name)
                assert gemini.call_count == 5

            with patch("deep_agent.src.agent.llm.ChatAnthropicVertex") as claude:
                for model_name in (
                    "claude-sonnet-4",
                    "claude-sonnet-4-6@default",
                    "claude-opus-5-5",
                ):
                    create_model(model_name)
                assert claude.call_count == 3

    def test_an_unknown_name_still_needs_a_vllm_endpoint(self):
        """Prefix routing must not swallow genuinely unknown models."""
        with patch.object(settings, "VLLM_BASE_URL", ""):
            with pytest.raises(ValueError, match="gemini-, claude-"):
                create_model("llama-3.1-70b")
