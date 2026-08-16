"""Unit tests for multi-provider LLM factory functions."""

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.agent.llm import (
    _create_anthropic_model,
    _create_azure_model,
    _create_ollama_model,
    _create_openai_model,
)


class TestCreateOpenAIModel:
    """Tests for _create_openai_model()."""

    def test_raises_when_api_key_missing(self):
        with patch("deep_agent.src.agent.llm.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = None
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                _create_openai_model("gpt-4o", 0.0, 4096)

    def test_creates_model_with_valid_key(self):
        mock_model = MagicMock()
        with patch("deep_agent.src.agent.llm.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            with patch(
                "langchain_openai.ChatOpenAI", return_value=mock_model
            ) as mock_cls:
                result = _create_openai_model("gpt-4o", 0.0, 4096)
        assert result is mock_model
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["model"] == "gpt-4o"
        assert mock_cls.call_args.kwargs["temperature"] == 0.0
        assert mock_cls.call_args.kwargs["max_tokens"] == 4096


class TestCreateAnthropicModel:
    """Tests for _create_anthropic_model()."""

    def test_raises_when_api_key_missing(self):
        with patch("deep_agent.src.agent.llm.settings") as mock_settings:
            mock_settings.ANTHROPIC_API_KEY = None
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                _create_anthropic_model("claude-3-haiku", 0.0, 4096)

    def test_creates_model_with_valid_key(self):
        mock_model = MagicMock()
        with patch("deep_agent.src.agent.llm.settings") as mock_settings:
            mock_settings.ANTHROPIC_API_KEY = "test-key"
            with patch(
                "langchain_anthropic.ChatAnthropic", return_value=mock_model
            ) as mock_cls:
                result = _create_anthropic_model("claude-3-haiku", 0.0, 4096)
        assert result is mock_model
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["model"] == "claude-3-haiku"
        assert mock_cls.call_args.kwargs["temperature"] == 0.0
        assert mock_cls.call_args.kwargs["max_tokens"] == 4096


class TestCreateAzureModel:
    """Tests for _create_azure_model()."""

    def test_raises_when_config_missing(self):
        with patch("deep_agent.src.agent.llm.settings") as mock_settings:
            mock_settings.AZURE_OPENAI_API_KEY = None
            mock_settings.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com/"
            mock_settings.AZURE_OPENAI_DEPLOYMENT = "test-deploy"
            mock_settings.AZURE_OPENAI_API_VERSION = "2024-02-01"
            with pytest.raises(ValueError, match="Missing Azure config"):
                _create_azure_model("gpt-4o", 0.0, 4096)

    def test_raises_when_multiple_configs_missing(self):
        with patch("deep_agent.src.agent.llm.settings") as mock_settings:
            mock_settings.AZURE_OPENAI_API_KEY = None
            mock_settings.AZURE_OPENAI_ENDPOINT = None
            mock_settings.AZURE_OPENAI_DEPLOYMENT = "test-deploy"
            mock_settings.AZURE_OPENAI_API_VERSION = "2024-02-01"
            with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY") as exc_info:
                _create_azure_model("gpt-4o", 0.0, 4096)
            assert "AZURE_OPENAI_ENDPOINT" in str(exc_info.value)


class TestCreateOllamaModel:
    """Tests for _create_ollama_model()."""

    def test_raises_when_base_url_missing(self):
        with patch("deep_agent.src.agent.llm.settings") as mock_settings:
            mock_settings.OLLAMA_BASE_URL = None
            with pytest.raises(ValueError, match="OLLAMA_BASE_URL"):
                _create_ollama_model("llama3", 0.0, 4096)

    def test_creates_model_with_valid_url(self):
        mock_model = MagicMock()
        with patch("deep_agent.src.agent.llm.settings") as mock_settings:
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            with patch(
                "langchain_ollama.ChatOllama", return_value=mock_model
            ) as mock_cls:
                result = _create_ollama_model("llama3", 0.0, 4096)
        assert result is mock_model
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["model"] == "llama3"
        assert mock_cls.call_args.kwargs["num_predict"] == 4096
