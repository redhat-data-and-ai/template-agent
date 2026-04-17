"""Unit tests for subagent initialization and configuration."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from template_agent.src.core.subagents import (
    _resolve_skills,
    _resolve_tools,
    load_subagents,
    parse_agent_frontmatter,
)


class TestParseAgentFrontmatter:
    """Tests for parse_agent_frontmatter function."""

    def test_parse_valid_frontmatter(self, tmp_path):
        """Test parsing valid YAML frontmatter."""
        agent_file = tmp_path / "test_agent.md"
        agent_file.write_text(
            """---
name: test-agent
description: A test agent
model: gemini-2.5-flash
tools:
  - tool1
  - tool2
skills:
  - skill1
---

This is the system prompt for the agent.
It can have multiple lines.
"""
        )

        result = parse_agent_frontmatter(agent_file)

        assert result["name"] == "test-agent"
        assert result["description"] == "A test agent"
        assert result["model"] == "gemini-2.5-flash"
        assert result["tools"] == ["tool1", "tool2"]
        assert result["skills"] == ["skill1"]
        assert "This is the system prompt" in result["body"]
        assert "multiple lines" in result["body"]

    def test_parse_without_frontmatter(self, tmp_path):
        """Test parsing markdown file without frontmatter."""
        agent_file = tmp_path / "simple_agent.md"
        content = "Just a simple system prompt without frontmatter."
        agent_file.write_text(content)

        result = parse_agent_frontmatter(agent_file)

        assert result == {"body": content}

    def test_parse_incomplete_frontmatter(self, tmp_path):
        """Test parsing incomplete frontmatter (missing closing ---)."""
        agent_file = tmp_path / "incomplete.md"
        agent_file.write_text(
            """---
name: incomplete
description: Missing closing marker

This should be treated as body."""
        )

        result = parse_agent_frontmatter(agent_file)

        # Should treat entire content as body when frontmatter is incomplete
        assert "body" in result
        assert "---" in result["body"]

    def test_parse_empty_frontmatter(self, tmp_path):
        """Test parsing with empty frontmatter section."""
        agent_file = tmp_path / "empty_fm.md"
        agent_file.write_text(
            """---
---

System prompt here."""
        )

        result = parse_agent_frontmatter(agent_file)

        assert result["body"] == "System prompt here."
        # Empty frontmatter should still work
        assert len(result) == 1  # Just the body

    def test_parse_frontmatter_with_multiline_values(self, tmp_path):
        """Test parsing frontmatter with multiline string values."""
        agent_file = tmp_path / "multiline.md"
        agent_file.write_text(
            """---
name: multiline-agent
description: >
  This is a multiline
  description that spans
  multiple lines.
---

System prompt."""
        )

        result = parse_agent_frontmatter(agent_file)

        assert result["name"] == "multiline-agent"
        assert "multiline" in result["description"]
        assert "multiple lines" in result["description"]

    def test_parse_frontmatter_with_null_values(self, tmp_path):
        """Test parsing frontmatter with null values."""
        agent_file = tmp_path / "nulls.md"
        agent_file.write_text(
            """---
name: null-test
description: null
tools: null
---

Prompt."""
        )

        result = parse_agent_frontmatter(agent_file)

        assert result["name"] == "null-test"
        assert result["description"] is None
        assert result["tools"] is None


class TestResolveTools:
    """Tests for _resolve_tools function."""

    def test_resolve_all_tools_found(self):
        """Test resolving tools when all are available."""
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"

        tool_by_name = {"tool1": mock_tool1, "tool2": mock_tool2}
        tool_names = ["tool1", "tool2"]

        resolved = _resolve_tools("test-agent", tool_names, tool_by_name)

        assert len(resolved) == 2
        assert mock_tool1 in resolved
        assert mock_tool2 in resolved

    def test_resolve_some_tools_missing(self):
        """Test resolving tools when some are missing."""
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"

        tool_by_name = {"tool1": mock_tool1}
        tool_names = ["tool1", "tool2", "tool3"]

        resolved = _resolve_tools("test-agent", tool_names, tool_by_name)

        assert len(resolved) == 1
        assert mock_tool1 in resolved

    def test_resolve_all_tools_missing(self):
        """Test resolving tools when all are missing."""
        tool_by_name = {}
        tool_names = ["tool1", "tool2"]

        resolved = _resolve_tools("test-agent", tool_names, tool_by_name)

        assert len(resolved) == 0

    def test_resolve_empty_tool_list(self):
        """Test resolving empty tool list."""
        mock_tool = MagicMock()
        tool_by_name = {"tool1": mock_tool}
        tool_names = []

        resolved = _resolve_tools("test-agent", tool_names, tool_by_name)

        assert len(resolved) == 0

    def test_resolve_maintains_order(self):
        """Test that tool resolution maintains order."""
        mock_tools = [MagicMock() for _ in range(3)]
        for i, tool in enumerate(mock_tools):
            tool.name = f"tool{i}"

        tool_by_name = {f"tool{i}": tool for i, tool in enumerate(mock_tools)}
        tool_names = ["tool2", "tool0", "tool1"]

        resolved = _resolve_tools("test-agent", tool_names, tool_by_name)

        assert len(resolved) == 3
        assert resolved[0] == mock_tools[2]
        assert resolved[1] == mock_tools[0]
        assert resolved[2] == mock_tools[1]


class TestResolveSkills:
    """Tests for _resolve_skills function."""

    def test_resolve_existing_skills(self, tmp_path):
        """Test resolving skills that exist."""
        skills_base = tmp_path / "skills"
        skills_base.mkdir()
        (skills_base / "skill1").mkdir()
        (skills_base / "skill2").mkdir()

        skill_names = ["skill1", "skill2"]
        resolved = _resolve_skills("test-agent", skill_names, skills_base)

        assert len(resolved) == 2
        assert str(skills_base / "skill1") in resolved
        assert str(skills_base / "skill2") in resolved

    def test_resolve_missing_skills(self, tmp_path):
        """Test resolving skills that don't exist."""
        skills_base = tmp_path / "skills"
        skills_base.mkdir()

        skill_names = ["skill1", "skill2"]
        resolved = _resolve_skills("test-agent", skill_names, skills_base)

        assert len(resolved) == 0

    def test_resolve_mixed_skills(self, tmp_path):
        """Test resolving mix of existing and missing skills."""
        skills_base = tmp_path / "skills"
        skills_base.mkdir()
        (skills_base / "skill1").mkdir()

        skill_names = ["skill1", "skill2", "skill3"]
        resolved = _resolve_skills("test-agent", skill_names, skills_base)

        assert len(resolved) == 1
        assert str(skills_base / "skill1") in resolved

    def test_resolve_empty_skill_list(self, tmp_path):
        """Test resolving empty skill list."""
        skills_base = tmp_path / "skills"
        skills_base.mkdir()

        skill_names = []
        resolved = _resolve_skills("test-agent", skill_names, skills_base)

        assert len(resolved) == 0


class TestLoadSubagents:
    """Tests for load_subagents function."""

    def test_load_subagents_missing_directory(self, tmp_path):
        """Test loading subagents when directory doesn't exist."""
        agents_dir = tmp_path / "nonexistent"
        skills_base = tmp_path / "skills"

        result = load_subagents(agents_dir, [], skills_base)

        assert result is None

    def test_load_subagents_empty_directory(self, tmp_path):
        """Test loading subagents from empty directory."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        skills_base = tmp_path / "skills"
        skills_base.mkdir()

        result = load_subagents(agents_dir, [], skills_base)

        assert result == []

    def test_load_single_subagent_minimal(self, tmp_path):
        """Test loading single subagent with minimal configuration."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        skills_base = tmp_path / "skills"
        skills_base.mkdir()

        agent_file = agents_dir / "simple.md"
        agent_file.write_text(
            """---
model: gemini-2.5-flash
---

Simple system prompt."""
        )

        mock_model = MagicMock()
        with patch("template_agent.src.core.subagents.create_model") as mock_create:
            mock_create.return_value = mock_model

            result = load_subagents(agents_dir, [], skills_base)

            assert len(result) == 1
            assert result[0]["name"] == "simple"
            assert result[0]["system_prompt"] == "Simple system prompt."
            assert result[0]["description"] == ""
            assert result[0]["model"] == mock_model

    def test_load_subagent_with_frontmatter(self, tmp_path):
        """Test loading subagent with full frontmatter."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        skills_base = tmp_path / "skills"
        skills_base.mkdir()

        agent_file = agents_dir / "sample_agent.md"
        agent_file.write_text(
            """---
name: sample_agent
description: A sample agent for testing
model: gemini-2.5-flash
---

You are a sample agent for testing purposes."""
        )

        mock_model = MagicMock()
        with patch("template_agent.src.core.subagents.create_model") as mock_create:
            mock_create.return_value = mock_model

            result = load_subagents(agents_dir, [], skills_base)

            assert len(result) == 1
            assert result[0]["name"] == "sample_agent"
            assert result[0]["description"] == "A sample agent for testing"
            assert "sample agent" in result[0]["system_prompt"]

    def test_load_subagent_with_tools(self, tmp_path):
        """Test loading subagent with tools."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        skills_base = tmp_path / "skills"
        skills_base.mkdir()

        mock_tool1 = MagicMock()
        mock_tool1.name = "tool_one"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool_two"

        agent_file = agents_dir / "sample_agent.md"
        agent_file.write_text(
            """---
name: sample_agent
model: gemini-2.5-flash
tools:
  - tool_one
  - tool_two
---

Analyst prompt."""
        )

        mock_model = MagicMock()
        with patch("template_agent.src.core.subagents.create_model") as mock_create:
            mock_create.return_value = mock_model

            result = load_subagents(agents_dir, [mock_tool1, mock_tool2], skills_base)

            assert len(result) == 1
            assert "tools" in result[0]
            assert len(result[0]["tools"]) == 2

    def test_load_subagent_with_skills(self, tmp_path):
        """Test loading subagent with skills."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        skills_base = tmp_path / "skills"
        skills_base.mkdir()
        (skills_base / "test-skill").mkdir()

        agent_file = agents_dir / "sample_agent.md"
        agent_file.write_text(
            """---
name: sample_agent
model: gemini-2.5-flash
skills:
  - test-skill
---

Analyst with skills."""
        )

        mock_model = MagicMock()
        with patch("template_agent.src.core.subagents.create_model") as mock_create:
            mock_create.return_value = mock_model

            result = load_subagents(agents_dir, [], skills_base)

            assert len(result) == 1
            assert "skills" in result[0]
            assert len(result[0]["skills"]) == 1
            assert "test-skill" in result[0]["skills"][0]

    def test_load_subagent_with_model(self, tmp_path):
        """Test loading subagent with model specification."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        skills_base = tmp_path / "skills"
        skills_base.mkdir()

        agent_file = agents_dir / "sample_agent.md"
        agent_file.write_text(
            """---
name: sample_agent
model: gemini-2.5-flash
---

Analyst with model."""
        )

        mock_model = MagicMock()

        with patch("template_agent.src.core.subagents.create_model") as mock_create:
            mock_create.return_value = mock_model

            result = load_subagents(agents_dir, [], skills_base)

            assert len(result) == 1
            mock_create.assert_called_once_with(model_name="gemini-2.5-flash")
            assert result[0]["model"] == mock_model

    def test_load_multiple_subagents(self, tmp_path):
        """Test loading multiple subagents."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        skills_base = tmp_path / "skills"
        skills_base.mkdir()

        # Create multiple agent files
        (agents_dir / "agent1.md").write_text(
            """---
model: gemini-2.5-flash
---

Agent 1 prompt."""
        )
        (agents_dir / "agent2.md").write_text(
            """---
model: gemini-2.5-flash
---

Agent 2 prompt."""
        )
        (agents_dir / "agent3.md").write_text(
            """---
model: gemini-2.5-flash
---

Agent 3 prompt."""
        )

        mock_model = MagicMock()
        with patch("template_agent.src.core.subagents.create_model") as mock_create:
            mock_create.return_value = mock_model

            result = load_subagents(agents_dir, [], skills_base)

            assert len(result) == 3
            agent_names = {agent["name"] for agent in result}
            assert agent_names == {"agent1", "agent2", "agent3"}

    def test_load_subagents_sorted_by_filename(self, tmp_path):
        """Test that subagents are loaded in sorted order."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        skills_base = tmp_path / "skills"
        skills_base.mkdir()

        # Create files in reverse order
        (agents_dir / "z_agent.md").write_text(
            """---
model: gemini-2.5-flash
---

Z agent."""
        )
        (agents_dir / "a_agent.md").write_text(
            """---
model: gemini-2.5-flash
---

A agent."""
        )
        (agents_dir / "m_agent.md").write_text(
            """---
model: gemini-2.5-flash
---

M agent."""
        )

        mock_model = MagicMock()
        with patch("template_agent.src.core.subagents.create_model") as mock_create:
            mock_create.return_value = mock_model

            result = load_subagents(agents_dir, [], skills_base)

            assert len(result) == 3
            assert result[0]["name"] == "a_agent"
            assert result[1]["name"] == "m_agent"
            assert result[2]["name"] == "z_agent"

    def test_load_subagent_ignores_non_md_files(self, tmp_path):
        """Test that only .md files are loaded."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        skills_base = tmp_path / "skills"
        skills_base.mkdir()

        (agents_dir / "agent.md").write_text(
            """---
model: gemini-2.5-flash
---

Agent prompt."""
        )
        (agents_dir / "readme.txt").write_text("Not an agent.")
        (agents_dir / "config.yaml").write_text("name: test")

        mock_model = MagicMock()
        with patch("template_agent.src.core.subagents.create_model") as mock_create:
            mock_create.return_value = mock_model

            result = load_subagents(agents_dir, [], skills_base)

            assert len(result) == 1
            assert result[0]["name"] == "agent"

    def test_load_subagent_name_from_frontmatter_overrides_filename(self, tmp_path):
        """Test that name in frontmatter overrides filename."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        skills_base = tmp_path / "skills"
        skills_base.mkdir()

        agent_file = agents_dir / "filename.md"
        agent_file.write_text(
            """---
name: custom-name
model: gemini-2.5-flash
---

Prompt."""
        )

        mock_model = MagicMock()
        with patch("template_agent.src.core.subagents.create_model") as mock_create:
            mock_create.return_value = mock_model

            result = load_subagents(agents_dir, [], skills_base)

            assert len(result) == 1
            assert result[0]["name"] == "custom-name"

    def test_load_subagent_with_missing_tools(self, tmp_path):
        """Test loading subagent when some tools are missing."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        skills_base = tmp_path / "skills"
        skills_base.mkdir()

        mock_tool = MagicMock()
        mock_tool.name = "existing_tool"

        agent_file = agents_dir / "agent.md"
        agent_file.write_text(
            """---
name: test
model: gemini-2.5-flash
tools:
  - existing_tool
  - missing_tool
---

Prompt."""
        )

        mock_model = MagicMock()
        with patch("template_agent.src.core.subagents.create_model") as mock_create:
            mock_create.return_value = mock_model

            result = load_subagents(agents_dir, [mock_tool], skills_base)

            assert len(result) == 1
            assert len(result[0]["tools"]) == 1  # Only existing tool
            assert result[0]["tools"][0].name == "existing_tool"

    @pytest.mark.parametrize(
        "frontmatter",
        [
            "---\nname: no-model\n---\n\nPrompt.",
            "---\nname: null-model\nmodel: null\n---\n\nPrompt.",
        ],
    )
    def test_load_subagent_missing_model_raises_error(self, tmp_path, frontmatter):
        """Test loading subagent without or with null model raises ValueError."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        skills_base = tmp_path / "skills"
        skills_base.mkdir()

        agent_file = agents_dir / "agent.md"
        agent_file.write_text(frontmatter)

        with pytest.raises(ValueError, match="missing required 'model' field"):
            load_subagents(agents_dir, [], skills_base)
