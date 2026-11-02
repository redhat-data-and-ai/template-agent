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


# Patch target note: _create_vllm_model imports ExtraContentChatOpenAI from
# openai_compat inside the function body, so these patch it where it is
# defined. Patching langchain_openai.ChatOpenAI does NOT intercept it — the
# wrapper is a subclass that is already bound by the time the call happens.
_WRAPPER = "deep_agent.src.agent.openai_compat.ExtraContentChatOpenAI"


class TestCreateVllmModel:
    """Tests for the vLLM/OpenAI-compatible model creation path."""

    @staticmethod
    def _settings():
        mock_settings = MagicMock()
        mock_settings.VLLM_BASE_URL = "http://localhost:8000/v1"
        mock_settings.VLLM_API_KEY = "test-key"
        return mock_settings

    def test_create_vllm_model_success(self):
        """The wrapper is constructed with the resolved endpoint settings."""
        mock_instance = MagicMock()

        with (
            patch("deep_agent.src.agent.llm.settings", self._settings()),
            patch(_WRAPPER, return_value=mock_instance) as mock_wrapper,
        ):
            from deep_agent.src.agent.llm import _create_vllm_model

            result = _create_vllm_model("custom-model", 0.5, 4096)

            mock_wrapper.assert_called_once_with(
                model="custom-model",
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                temperature=0.5,
                max_tokens=4096,
                max_retries=2,
                streaming=True,
            )
            assert result == mock_instance

    def test_openai_compatible_path_uses_the_wrapper_not_plain_chatopenai(self):
        """Guards the reason this module exists.

        Falling back to a bare ChatOpenAI would drop the Gemini 3
        thought_signature and break every multi-turn tool call, while every
        other assertion here still passed. Assert the concrete type.
        """
        from deep_agent.src.agent.openai_compat import ExtraContentChatOpenAI

        with patch("deep_agent.src.agent.llm.settings", self._settings()):
            from deep_agent.src.agent.llm import _create_vllm_model

            model = _create_vllm_model("gemini-3.5-flash", 0.0, 2048)

        assert isinstance(model, ExtraContentChatOpenAI)

    def test_create_vllm_model_import_error(self):
        """A missing wrapper module surfaces as LLMError, not ImportError."""
        with patch("deep_agent.src.agent.llm.settings", self._settings()):
            import builtins

            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "deep_agent.src.agent.openai_compat":
                    raise ImportError("No module named 'langchain_openai'")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                from deep_agent.src.agent.llm import _create_vllm_model

                with pytest.raises(LLMError, match="langchain-openai is required"):
                    _create_vllm_model("custom-model", 0.5, 4096)

    def test_create_vllm_model_general_exception(self):
        """Construction failures are wrapped in LLMError with the cause kept."""
        with (
            patch("deep_agent.src.agent.llm.settings", self._settings()),
            patch(_WRAPPER, side_effect=RuntimeError("connection refused")),
        ):
            from deep_agent.src.agent.llm import _create_vllm_model

            with pytest.raises(LLMError, match="connection refused"):
                _create_vllm_model("custom-model", 0.5, 4096)

    def test_missing_base_url_raises_value_error(self):
        """Without VLLM_BASE_URL an unknown model is a config error, not a call."""
        mock_settings = MagicMock()
        mock_settings.VLLM_BASE_URL = ""

        with patch("deep_agent.src.agent.llm.settings", mock_settings):
            from deep_agent.src.agent.llm import _create_vllm_model

            with pytest.raises(ValueError, match="not a known Vertex AI model"):
                _create_vllm_model("custom-model", 0.5, 4096)

    def test_create_model_routes_to_vllm(self):
        """create_model sends unknown model names down the compatible path."""
        with (
            patch("deep_agent.src.agent.llm.settings", self._settings()),
            patch(_WRAPPER) as mock_wrapper,
        ):
            create_model("custom-model", temperature=0.3, max_output_tokens=2048)

            mock_wrapper.assert_called_once()
            call_kwargs = mock_wrapper.call_args[1]
            assert call_kwargs["model"] == "custom-model"
            assert call_kwargs["base_url"] == "http://localhost:8000/v1"
