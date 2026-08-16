"""Unit tests for multi-provider LLM factory functions."""

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.agent.llm import (
    _create_anthropic_model,
    _create_azure_model,
    _create_ollama_model,
    _create_openai_model,
)
from deep_agent.src.exceptions import LLMError


class TestCreateOpenAIModel:
    """Tests for _create_openai_model()."""

    def test_raises_when_api_key_missing(self):
        with patch("deep_agent.src.agent.llm.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = None
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                _create_openai_model("gpt-4o", 0.0, 4096)

    def test_creates_model_with_valid_key(self):
        mock_model = MagicMock()
        with (
            patch("deep_agent.src.agent.llm.settings") as mock_settings,
            patch(
                "deep_agent.src.agent.llm.ChatOpenAI",
                return_value=mock_model,
                create=True,
            ),
        ):
            mock_settings.OPENAI_API_KEY = "test-key"
            with patch.dict("sys.modules", {"langchain_openai": MagicMock()}):
                result = _create_openai_model("gpt-4o", 0.0, 4096)
            assert result is not None


class TestCreateAnthropicModel:
    """Tests for _create_anthropic_model()."""

    def test_raises_when_api_key_missing(self):
        with patch("deep_agent.src.agent.llm.settings") as mock_settings:
            mock_settings.ANTHROPIC_API_KEY = None
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                _create_anthropic_model("claude-3-haiku", 0.0, 4096)

    def test_creates_model_with_valid_key(self):
        mock_model = MagicMock()
        with (
            patch("deep_agent.src.agent.llm.settings") as mock_settings,
            patch(
                "deep_agent.src.agent.llm.ChatAnthropic",
                return_value=mock_model,
                create=True,
            ),
        ):
            mock_settings.ANTHROPIC_API_KEY = "test-key"
            with patch.dict("sys.modules", {"langchain_anthropic": MagicMock()}):
                result = _create_anthropic_model("claude-3-haiku", 0.0, 4096)
            assert result is not None


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
            with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
                _create_azure_model("gpt-4o", 0.0, 4096)


class TestCreateOllamaModel:
    """Tests for _create_ollama_model()."""

    def test_raises_when_base_url_missing(self):
        with patch("deep_agent.src.agent.llm.settings") as mock_settings:
            mock_settings.OLLAMA_BASE_URL = None
            with pytest.raises(ValueError, match="OLLAMA_BASE_URL"):
                _create_ollama_model("llama3", 0.0, 4096)

    def test_creates_model_with_valid_url(self):
        mock_model = MagicMock()
        with (
            patch("deep_agent.src.agent.llm.settings") as mock_settings,
            patch(
                "deep_agent.src.agent.llm.ChatOllama",
                return_value=mock_model,
                create=True,
            ),
        ):
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            with patch.dict("sys.modules", {"langchain_ollama": MagicMock()}):
                result = _create_ollama_model("llama3", 0.0, 4096)
            assert result is not None
