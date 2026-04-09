"""Tests for the prompt module."""

from unittest.mock import patch

from template_agent.src.core.prompt import get_current_date, get_system_prompt


class TestPrompt:
    """Test cases for prompt functions."""

    @patch("template_agent.src.core.prompt.get_current_date")
    def test_get_system_prompt_includes_date(self, mock_get_date):
        """Test that get_system_prompt includes the current date."""
        mock_get_date.return_value = "December 25, 2024"
        prompt = get_system_prompt()
        assert "Today's date is December 25, 2024" in prompt

    def test_get_system_prompt_no_template_vars(self):
        """Test that no unresolved template variables remain."""
        prompt = get_system_prompt()
        assert "{{" not in prompt
