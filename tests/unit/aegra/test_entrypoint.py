"""Unit tests for container entrypoint config validation."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from deep_agent.aegra.entrypoint import CONFIG_PATH, validate_config_mount


class TestValidateConfigMount:
    def _write_valid_config(self, tmp_path):
        (tmp_path / "PROMPT.md").write_text(
            "---\nname: test\nmodel: gpt-4\n---\nPrompt body.\n"
        )
        (tmp_path / "mcp.json").write_text(
            json.dumps({"mcpServers": {"test": {"url": "http://localhost"}}})
        )

    def test_exits_when_config_path_missing(self, tmp_path):
        missing = tmp_path / "nonexistent"
        with patch("deep_agent.aegra.entrypoint.CONFIG_PATH", missing):
            with pytest.raises(SystemExit, match="1"):
                validate_config_mount()

    def test_exits_when_prompt_md_missing(self, tmp_path):
        (tmp_path / "mcp.json").write_text("{}")
        with patch("deep_agent.aegra.entrypoint.CONFIG_PATH", tmp_path):
            with pytest.raises(SystemExit, match="1"):
                validate_config_mount()

    def test_exits_when_mcp_json_missing(self, tmp_path):
        (tmp_path / "PROMPT.md").write_text("---\nname: test\nmodel: gpt-4\n---\n")
        with patch("deep_agent.aegra.entrypoint.CONFIG_PATH", tmp_path):
            with pytest.raises(SystemExit, match="1"):
                validate_config_mount()

    def test_passes_with_valid_config(self, tmp_path):
        self._write_valid_config(tmp_path)
        with patch("deep_agent.aegra.entrypoint.CONFIG_PATH", tmp_path):
            validate_config_mount()

    def test_warns_on_invalid_mcp_json(self, tmp_path, capsys):
        (tmp_path / "PROMPT.md").write_text("---\nname: test\nmodel: gpt-4\n---\n")
        (tmp_path / "mcp.json").write_text("{ invalid json //")
        with patch("deep_agent.aegra.entrypoint.CONFIG_PATH", tmp_path):
            validate_config_mount()
        captured = capsys.readouterr()
        assert "WARNING" in captured.err or "Invalid" in captured.err
