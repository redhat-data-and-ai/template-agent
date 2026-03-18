"""Tests for the deep research utils module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.utils import (
    GIBBERISH_RESPONSE,
    aput_checkpoint,
    classify_input_quality,
    get_raw_checkpointer,
    get_setting,
    sanitize_error_for_client,
)


class TestSanitizeErrorForClient:
    """Test cases for sanitize_error_for_client."""

    def test_sanitize_error_short_message_unchanged(self):
        """Test that short messages are returned unchanged."""
        exc = ValueError("Short error")

        result = sanitize_error_for_client(exc)

        assert result == "Short error"

    def test_sanitize_error_exactly_500_chars_unchanged(self):
        """Test that exactly 500 chars are returned unchanged."""
        msg = "x" * 500
        exc = ValueError(msg)

        result = sanitize_error_for_client(exc)

        assert result == msg
        assert len(result) == 500

    def test_sanitize_error_long_message_truncated(self):
        """Test that messages over 500 chars are truncated with ellipsis."""
        msg = "a" * 600
        exc = ValueError(msg)

        result = sanitize_error_for_client(exc)

        assert len(result) == 500
        assert result.endswith("...")
        assert result == msg[:497] + "..."


class TestGetSetting:
    """Test cases for get_setting."""

    def test_get_setting_returns_value(self):
        """Test that get_setting returns value when attribute exists."""
        mock_settings = MagicMock()
        mock_settings.some_setting = "value"
        mock_module = MagicMock()
        mock_module.settings = mock_settings

        with patch.dict(
            "sys.modules",
            {"template_agent.src.settings": mock_module},
            clear=False,
        ):
            result = get_setting("some_setting", "default")

        assert result == "value"

    def test_get_setting_returns_default_on_missing(self):
        """Test that get_setting returns default when attribute missing."""
        mock_settings = object()  # No attributes
        mock_module = MagicMock()
        mock_module.settings = mock_settings

        with patch.dict(
            "sys.modules",
            {"template_agent.src.settings": mock_module},
            clear=False,
        ):
            result = get_setting("nonexistent", "fallback")

        assert result == "fallback"

    def test_get_setting_returns_default_on_import_error(self):
        """Test that get_setting returns default when import fails."""
        import builtins

        real_import = builtins.__import__

        def raise_on_settings_import(name, *args, **kwargs):
            if name == "template_agent.src.settings":
                raise ImportError("No module")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=raise_on_settings_import):
            result = get_setting("some_setting", "default")

        assert result == "default"


class TestGibberishResponse:
    """Test cases for GIBBERISH_RESPONSE constant."""

    def test_gibberish_response_is_nonempty_string(self):
        """Test that GIBBERISH_RESPONSE is a non-empty string."""
        assert isinstance(GIBBERISH_RESPONSE, str)
        assert len(GIBBERISH_RESPONSE) > 0


class TestGetRawCheckpointer:
    """Test cases for get_raw_checkpointer."""

    def test_get_raw_checkpointer_returns_raw_if_wrapped(self):
        """Test that raw checkpointer is returned when wrapped."""
        raw = MagicMock()
        wrapped = MagicMock(raw_checkpointer=raw)

        result = get_raw_checkpointer(wrapped)

        assert result is raw

    def test_get_raw_checkpointer_returns_self_if_not_wrapped(self):
        """Test that checkpointer is returned when not wrapped."""
        checkpointer = object()  # No raw_checkpointer attribute

        result = get_raw_checkpointer(checkpointer)

        assert result is checkpointer


class TestAputCheckpoint:
    """Test cases for aput_checkpoint."""

    @pytest.mark.asyncio
    async def test_aput_checkpoint_calls_aput_tuple(self):
        """Test that aput_checkpoint calls aput_tuple on checkpointer."""
        mock_checkpointer = MagicMock()
        mock_checkpointer.aput_tuple = AsyncMock()

        config = MagicMock()
        checkpoint = MagicMock()
        metadata = MagicMock()

        await aput_checkpoint(mock_checkpointer, config, checkpoint, metadata)

        mock_checkpointer.aput_tuple.assert_called_once()
        call_args = mock_checkpointer.aput_tuple.call_args[0][0]
        assert call_args.config == config
        assert call_args.checkpoint == checkpoint
        assert call_args.metadata == metadata

    @pytest.mark.asyncio
    async def test_aput_checkpoint_logs_warning_on_error(self, caplog):
        """Test that aput_checkpoint logs warning when aput_tuple fails."""
        import logging

        mock_checkpointer = MagicMock()
        mock_checkpointer.aput_tuple = AsyncMock(
            side_effect=RuntimeError("Checkpoint failed")
        )

        with caplog.at_level(logging.WARNING):
            await aput_checkpoint(
                mock_checkpointer,
                MagicMock(),
                MagicMock(),
                MagicMock(),
            )

        assert "Checkpoint persistence failed" in caplog.text


class TestClassifyInputQuality:
    """Test cases for classify_input_quality."""

    @pytest.mark.asyncio
    async def test_classify_input_quality_returns_research_query(self):
        """Test that classify_input_quality returns research_query when model says so."""
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(
            return_value=MagicMock(content='{"classification": "research_query"}')
        )

        result = await classify_input_quality("What is the data?", mock_model)

        assert result == "research_query"

    @pytest.mark.asyncio
    async def test_classify_input_quality_returns_gibberish(self):
        """Test that classify_input_quality returns gibberish when model says so."""
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(
            return_value=MagicMock(content='{"classification": "gibberish"}')
        )

        result = await classify_input_quality("asdf", mock_model)

        assert result == "gibberish"

    @pytest.mark.asyncio
    async def test_classify_input_quality_defaults_on_error(self):
        """Test that classify_input_quality returns research_query when model raises."""
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(side_effect=RuntimeError("Model error"))

        result = await classify_input_quality("anything", mock_model)

        assert result == "research_query"
