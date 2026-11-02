"""Unit tests for LLM model configuration and initialization."""

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.agent.llm import (
    CLAUDE_MODELS,
    GEMINI_MODELS,
    create_model,
    safety_settings,
)
from deep_agent.src.exceptions import LLMError


class TestCreateModel:
    """Tests for create_model function."""

    def test_create_gemini_model(self):
        """Test creating Gemini model — no location settings ⇒ location=None."""
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
                    location=None,
                    max_output_tokens=8192,
                    max_retries=2,
                    streaming=True,
                    safety_settings=safety_settings,
                )

    def test_create_gemini_model_with_location(self):
        """Test that explicit location is passed to ChatGoogleGenerativeAI.

        Regression test for GitHub issue #408: location was silently dropped
        for Gemini models even though ChatGoogleGenerativeAI accepts it and
        uses it to derive the Vertex AI endpoint URL.
        """
        mock_creds = MagicMock()

        with patch(
            "deep_agent.src.agent.llm.get_service_account_credentials"
        ) as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-project")

            with patch("deep_agent.src.agent.llm.ChatGoogleGenerativeAI") as mock_chat:
                create_model(
                    "gemini-2.5-pro",
                    temperature=0.5,
                    location="europe-west1",
                )
                mock_chat.assert_called_once_with(
                    model="gemini-2.5-pro",
                    temperature=0.5,
                    credentials=mock_creds,
                    project="test-project",
                    location="europe-west1",
                    max_output_tokens=8192,
                    max_retries=2,
                    streaming=True,
                    safety_settings=safety_settings,
                )

    def test_create_claude_model_with_location(self):
        """Test that explicit location overrides CLAUDE_LOCATION default for Claude."""
        mock_creds = MagicMock()

        with patch(
            "deep_agent.src.agent.llm.get_service_account_credentials"
        ) as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-project")

            with patch("deep_agent.src.agent.llm.ChatAnthropicVertex") as mock_chat:
                create_model(
                    "claude-sonnet-4",
                    temperature=0.7,
                    location="europe-west1",
                )
                mock_chat.assert_called_once_with(
                    model="claude-sonnet-4",
                    project="test-project",
                    location="europe-west1",
                    credentials=mock_creds,
                    max_tokens=8192,
                    max_retries=2,
                    streaming=True,
                )

    def test_create_claude_model(self):
        """Test creating Claude model — no location settings ⇒ falls back to us-east5."""
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
        mock_settings = MagicMock()
        mock_settings.VLLM_BASE_URL = ""

        with (
            patch(
                "deep_agent.src.agent.llm.get_service_account_credentials"
            ) as mock_get_creds,
            patch("deep_agent.src.agent.llm.settings", mock_settings),
        ):
            mock_get_creds.return_value = (mock_creds, "test-project")

            with pytest.raises(ValueError) as exc_info:
                create_model("gpt-4")

            error_msg = str(exc_info.value)
            assert "gpt-4" in error_msg
            assert "not a known Vertex AI model" in error_msg

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

    def test_all_supported_models_work(self):
        """Test that all models in GEMINI_MODELS and CLAUDE_MODELS are supported."""
        mock_creds = MagicMock()

        with patch(
            "deep_agent.src.agent.llm.get_service_account_credentials"
        ) as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-project")

            with patch("deep_agent.src.agent.llm.ChatGoogleGenerativeAI"):
                for model_name in GEMINI_MODELS:
                    create_model(model_name)

            with patch("deep_agent.src.agent.llm.ChatAnthropicVertex"):
                for model_name in CLAUDE_MODELS:
                    create_model(model_name)

    def test_create_gemini_model_falls_back_to_google_cloud_location(self):
        """GOOGLE_CLOUD_LOCATION is used when GEMINI_LOCATION is unset."""
        mock_creds = MagicMock()
        mock_settings = MagicMock()
        mock_settings.GEMINI_LOCATION = None
        mock_settings.GOOGLE_CLOUD_LOCATION = "us-central1"

        with (
            patch(
                "deep_agent.src.agent.llm.get_service_account_credentials"
            ) as mock_get_creds,
            patch("deep_agent.src.agent.llm.settings", mock_settings),
            patch("deep_agent.src.agent.llm.ChatGoogleGenerativeAI") as mock_chat,
        ):
            mock_get_creds.return_value = (mock_creds, "test-project")
            create_model("gemini-2.5-pro", temperature=0.5)
            mock_chat.assert_called_once()
            assert mock_chat.call_args[1]["location"] == "us-central1"

    def test_create_claude_model_ignores_google_cloud_location(self):
        """Claude falls back to us-east5, NOT GOOGLE_CLOUD_LOCATION (region-restricted)."""
        mock_creds = MagicMock()
        mock_settings = MagicMock()
        mock_settings.CLAUDE_LOCATION = None
        mock_settings.GOOGLE_CLOUD_LOCATION = "global"

        with (
            patch(
                "deep_agent.src.agent.llm.get_service_account_credentials"
            ) as mock_get_creds,
            patch("deep_agent.src.agent.llm.settings", mock_settings),
            patch("deep_agent.src.agent.llm.ChatAnthropicVertex") as mock_chat,
        ):
            mock_get_creds.return_value = (mock_creds, "test-project")
            create_model("claude-sonnet-4", temperature=0.7)
            mock_chat.assert_called_once()
            assert mock_chat.call_args[1]["location"] == "us-east5"

    def test_vertex_value_error_re_raised_directly(self):
        """ValueError from Vertex model constructors propagates without wrapping."""
        mock_creds = MagicMock()

        with patch(
            "deep_agent.src.agent.llm.get_service_account_credentials"
        ) as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-project")

            with patch(
                "deep_agent.src.agent.llm.ChatGoogleGenerativeAI",
                side_effect=ValueError("bad param"),
            ):
                with pytest.raises(ValueError, match="bad param"):
                    create_model("gemini-2.5-pro")

    def test_vertex_llm_error_re_raised_directly(self):
        """LLMError from Vertex model constructors propagates without wrapping."""
        mock_creds = MagicMock()

        with patch(
            "deep_agent.src.agent.llm.get_service_account_credentials"
        ) as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-project")

            with patch(
                "deep_agent.src.agent.llm.ChatAnthropicVertex",
                side_effect=LLMError("model unavailable"),
            ):
                with pytest.raises(LLMError, match="model unavailable"):
                    create_model("claude-sonnet-4")


class TestCreateVllmModel:
    """Tests for the vLLM/OpenAI-compatible model creation path."""

    def test_create_vllm_model_success(self):
        """Test successful vLLM model creation when VLLM_BASE_URL is set."""
        mock_settings = MagicMock()
        mock_settings.VLLM_BASE_URL = "http://localhost:8000/v1"
        mock_settings.VLLM_API_KEY = "test-key"

        mock_chat_instance = MagicMock()

        with (
            patch("deep_agent.src.agent.llm.settings", mock_settings),
            patch(
                "langchain_openai.ChatOpenAI", return_value=mock_chat_instance
            ) as mock_chat,
        ):
            from deep_agent.src.agent.llm import _create_vllm_model

            result = _create_vllm_model("custom-model", 0.5, 4096)

            mock_chat.assert_called_once_with(
                model="custom-model",
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                temperature=0.5,
                max_tokens=4096,
                max_retries=2,
                streaming=True,
            )
            assert result == mock_chat_instance

    def test_create_vllm_model_import_error(self):
        """Test that ImportError when langchain-openai is missing raises LLMError."""
        mock_settings = MagicMock()
        mock_settings.VLLM_BASE_URL = "http://localhost:8000/v1"

        with patch("deep_agent.src.agent.llm.settings", mock_settings):
            import builtins

            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "langchain_openai":
                    raise ImportError("No module named 'langchain_openai'")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                from deep_agent.src.agent.llm import _create_vllm_model

                with pytest.raises(LLMError, match="langchain-openai is required"):
                    _create_vllm_model("custom-model", 0.5, 4096)

    def test_create_vllm_model_general_exception(self):
        """Test that general exceptions from ChatOpenAI are wrapped in LLMError."""
        mock_settings = MagicMock()
        mock_settings.VLLM_BASE_URL = "http://localhost:8000/v1"
        mock_settings.VLLM_API_KEY = "test-key"

        with (
            patch("deep_agent.src.agent.llm.settings", mock_settings),
            patch(
                "langchain_openai.ChatOpenAI",
                side_effect=RuntimeError("connection refused"),
            ),
        ):
            from deep_agent.src.agent.llm import _create_vllm_model

            with pytest.raises(LLMError, match="connection refused"):
                _create_vllm_model("custom-model", 0.5, 4096)

    def test_create_model_routes_to_vllm(self):
        """Test that create_model routes unknown models to vLLM when configured."""
        mock_settings = MagicMock()
        mock_settings.VLLM_BASE_URL = "http://localhost:8000/v1"
        mock_settings.VLLM_API_KEY = "test-key"

        with (
            patch("deep_agent.src.agent.llm.settings", mock_settings),
            patch("langchain_openai.ChatOpenAI") as mock_chat,
        ):
            create_model("custom-model", temperature=0.3, max_output_tokens=2048)
            mock_chat.assert_called_once()
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model"] == "custom-model"
            assert call_kwargs["base_url"] == "http://localhost:8000/v1"
