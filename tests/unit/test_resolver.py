"""Unit tests for resolver module."""

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deep_agent.src.agent.config.resolver import (
    resolve_skill_paths,
    resolve_tools,
    to_virtual_skill_paths,
)


class TestToVirtualSkillPaths:
    """Tests for to_virtual_skill_paths function."""

    def test_converts_absolute_paths_to_virtual_paths(self):
        skill_paths = [
            "/home/user/skills/my-skill",
            "/home/user/skills/other-skill",
        ]
        result = to_virtual_skill_paths(skill_paths)
        assert result == ["/skills/my-skill", "/skills/other-skill"]

    def test_handles_empty_list(self):
        result = to_virtual_skill_paths([])
        assert result == []

    def test_deduplicates_skill_paths(self):
        """Validate that duplicate paths are deduplicated (requires set behavior)."""
        skill_paths = [
            "/home/user/skills/my-skill",
            "/home/user/skills/my-skill",
            "/home/user/skills/other-skill",
        ]
        result = to_virtual_skill_paths(skill_paths)
        assert result == ["/skills/my-skill", "/skills/other-skill"]
        assert len(result) == 2

    def test_returns_sorted_list(self):
        skill_paths = [
            "/home/user/skills/zebra-skill",
            "/home/user/skills/alpha-skill",
            "/home/user/skills/middle-skill",
        ]
        result = to_virtual_skill_paths(skill_paths)
        assert result == [
            "/skills/alpha-skill",
            "/skills/middle-skill",
            "/skills/zebra-skill",
        ]

    def test_uses_leaf_directory_name_only(self):
        skill_paths = [
            "/deep/nested/path/skills/my-skill",
        ]
        result = to_virtual_skill_paths(skill_paths)
        assert result == ["/skills/my-skill"]

    def test_virtual_variable_is_set_type(self):
        """Validate that the 'virtual' variable in to_virtual_skill_paths is a set[str].

        This test ensures line 75 remains unchanged:
            virtual: set[str] = set()

        We validate this by inspecting the function's source code to ensure
        the implementation uses a set, which is critical for deduplication behavior.
        """
        source = inspect.getsource(to_virtual_skill_paths)

        # Verify the exact line exists in the source
        assert "virtual: set[str] = set()" in source, (
            "Expected 'virtual: set[str] = set()' initialization on line 75. "
            "This ensures proper deduplication of skill paths."
        )

        # Additionally verify the behavior that depends on set usage
        skill_paths = [
            "/path/to/skills/duplicate",
            "/path/to/skills/duplicate",
        ]
        result = to_virtual_skill_paths(skill_paths)
        assert len(result) == 1, "Set behavior ensures duplicates are removed"
        assert result == ["/skills/duplicate"]


class TestResolveSkillPaths:
    """Tests for resolve_skill_paths function."""

    def test_resolves_existing_skills(self):
        skill_names = ["skill-a", "skill-b"]
        available_skills = {
            "skill-a": Path("/skills/skill-a"),
            "skill-b": Path("/skills/skill-b"),
            "skill-c": Path("/skills/skill-c"),
        }
        result = resolve_skill_paths(skill_names, available_skills, "test-agent")
        assert result == ["/skills/skill-a", "/skills/skill-b"]

    def test_handles_missing_skills(self, caplog):
        skill_names = ["skill-a", "missing-skill"]
        available_skills = {
            "skill-a": Path("/skills/skill-a"),
        }
        result = resolve_skill_paths(skill_names, available_skills, "test-agent")
        assert result == ["/skills/skill-a"]
        assert "references unknown skills: ['missing-skill']" in caplog.text

    def test_handles_empty_skill_list(self):
        result = resolve_skill_paths([], {}, "test-agent")
        assert result == []

    def test_preserves_order(self):
        skill_names = ["skill-c", "skill-a", "skill-b"]
        available_skills = {
            "skill-a": Path("/skills/skill-a"),
            "skill-b": Path("/skills/skill-b"),
            "skill-c": Path("/skills/skill-c"),
        }
        result = resolve_skill_paths(skill_names, available_skills, "test-agent")
        assert result == ["/skills/skill-c", "/skills/skill-a", "/skills/skill-b"]


class TestResolveTools:
    """Tests for resolve_tools function."""

    def test_resolves_existing_tools(self):
        tool_a = MagicMock(name="tool-a")
        tool_a.name = "tool-a"
        tool_b = MagicMock(name="tool-b")
        tool_b.name = "tool-b"
        tool_c = MagicMock(name="tool-c")
        tool_c.name = "tool-c"

        tool_names = ["tool-a", "tool-b"]
        available_tools = [tool_a, tool_b, tool_c]

        result = resolve_tools(tool_names, available_tools, "test-agent")
        assert result == [tool_a, tool_b]

    def test_handles_missing_tools(self, caplog):
        tool_a = MagicMock(name="tool-a")
        tool_a.name = "tool-a"

        tool_names = ["tool-a", "missing-tool"]
        available_tools = [tool_a]

        result = resolve_tools(tool_names, available_tools, "test-agent")
        assert result == [tool_a]
        assert "references unknown tools: ['missing-tool']" in caplog.text

    def test_handles_empty_tool_list(self):
        result = resolve_tools([], [], "test-agent")
        assert result == []

    def test_preserves_order(self):
        tool_a = MagicMock(name="tool-a")
        tool_a.name = "tool-a"
        tool_b = MagicMock(name="tool-b")
        tool_b.name = "tool-b"
        tool_c = MagicMock(name="tool-c")
        tool_c.name = "tool-c"

        tool_names = ["tool-c", "tool-a", "tool-b"]
        available_tools = [tool_a, tool_b, tool_c]

        result = resolve_tools(tool_names, available_tools, "test-agent")
        assert result == [tool_c, tool_a, tool_b]
