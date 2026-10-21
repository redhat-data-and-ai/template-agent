"""Unit tests for resolver module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deep_agent.src.agent.config.resolver import (
    resolve_skill_paths,
    resolve_tools,
    to_virtual_skill_sources,
)


class TestToVirtualSkillSources:
    """SkillsMiddleware is given directories to scan, not skills.

    Handing it a skill's own directory makes it look *inside* that
    skill for skills. `/skills/dataverse` holds `references/`, not
    another skill, so eight sources found nothing between them and
    the prompt read "(No skills available yet)" with eight skills on
    disk.
    """

    def test_a_skill_becomes_the_directory_holding_it(self):
        assert to_virtual_skill_sources(
            ["/home/user/skills/my-skill", "/home/user/skills/other-skill"]
        ) == ["/skills/"]

    def test_handles_empty_list(self):
        assert to_virtual_skill_sources([]) == []

    def test_many_skills_are_one_source(self):
        """Listing the directory once finds all of them."""
        result = to_virtual_skill_sources(
            [
                "/home/user/skills/zebra-skill",
                "/home/user/skills/alpha-skill",
                "/home/user/skills/alpha-skill",
            ]
        )
        assert result == ["/skills/"]

    def test_the_absolute_prefix_does_not_leak(self):
        """`/skills/` is a virtual path the backend routes, not a real one."""
        assert to_virtual_skill_sources(["/deep/nested/path/skills/my-skill"]) == [
            "/skills/"
        ]

    def test_an_unexpected_layout_still_scans_the_root(self, caplog):
        """Nested skill directories are unsupported — say so, do not crash."""
        with caplog.at_level("WARNING"):
            result = to_virtual_skill_sources(["/somewhere/else/category/my-skill"])
        assert result == ["/skills/"]
        assert "nested skill directories" in caplog.text.lower()


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
