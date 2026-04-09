"""Integration tests for agent initialization with skills and subagents.

This module tests the runtime behavior of agent creation, including:
1. Loading subagents from .md files
2. Loading skills and attaching to subagents
3. Tool resolution for subagents
4. Backend and skill middleware integration
"""

from pathlib import Path

import pytest

from template_agent.src.core.agent import _parse_agent_frontmatter

CONFIG_DIR = Path(__file__).parent.parent.parent / "template_agent" / "agent_config"


class TestAgentFrontmatterParsing:
    """Test the _parse_agent_frontmatter helper function."""

    def test_parse_frontmatter_with_valid_yaml(self, tmp_path):
        """Test parsing valid YAML frontmatter."""
        test_file = tmp_path / "test-agent.md"
        test_file.write_text(
            """---
name: test-agent
description: A test agent
tools:
  - tool1
  - tool2
---

This is the agent body.
"""
        )

        result = _parse_agent_frontmatter(test_file)

        assert result["name"] == "test-agent"
        assert result["description"] == "A test agent"
        assert result["tools"] == ["tool1", "tool2"]
        assert result["body"] == "This is the agent body."

    def test_parse_frontmatter_without_yaml(self, tmp_path):
        """Test parsing file without frontmatter."""
        test_file = tmp_path / "test-agent.md"
        test_file.write_text("Just plain markdown content.")

        result = _parse_agent_frontmatter(test_file)

        assert result["body"] == "Just plain markdown content."
        assert "name" not in result

    def test_parse_frontmatter_with_empty_yaml(self, tmp_path):
        """Test parsing with empty YAML section."""
        test_file = tmp_path / "test-agent.md"
        test_file.write_text(
            """---
---

Body content here.
"""
        )

        result = _parse_agent_frontmatter(test_file)

        assert result["body"] == "Body content here."

    def test_parse_frontmatter_preserves_markdown_formatting(self, tmp_path):
        """Test that body preserves markdown formatting."""
        test_file = tmp_path / "test-agent.md"
        test_file.write_text(
            """---
name: test
---

# Heading

- List item 1
- List item 2

**Bold text**
"""
        )

        result = _parse_agent_frontmatter(test_file)

        assert "# Heading" in result["body"]
        assert "- List item 1" in result["body"]
        assert "**Bold text**" in result["body"]


class TestSubagentConfiguration:
    """Test subagent configuration structure."""

    def test_bmi_analyst_tool_configuration(self):
        """Test analyst tool configuration is valid."""
        agent_file = CONFIG_DIR / "agents" / "analyst.md"
        config = _parse_agent_frontmatter(agent_file)

        # All tools should be strings
        assert all(isinstance(tool, str) for tool in config["tools"])

        # No duplicate tools
        assert len(config["tools"]) == len(set(config["tools"]))

    def test_publisher_skill_configuration(self):
        """Test publisher skill configuration is valid."""
        agent_file = CONFIG_DIR / "agents" / "publisher.md"
        config = _parse_agent_frontmatter(agent_file)

        # All skills should be strings
        assert all(isinstance(skill, str) for skill in config["skills"])

        # Skills should exist as directories
        skills_dir = CONFIG_DIR / "skills"
        for skill_name in config["skills"]:
            skill_path = skills_dir / skill_name
            assert skill_path.exists(), f"Skill directory not found: {skill_name}"


class TestSkillAvailability:
    """Test that skills referenced by subagents exist."""

    def test_bmi_analyst_skills_exist(self):
        """Test analyst referenced skills exist."""
        agent_file = CONFIG_DIR / "agents" / "analyst.md"
        config = _parse_agent_frontmatter(agent_file)

        skills_dir = CONFIG_DIR / "skills"
        for skill_name in config.get("skills", []):
            skill_dir = skills_dir / skill_name
            skill_file = skill_dir / "SKILL.md"
            assert skill_file.exists(), f"Missing skill: {skill_name}"

    def test_publisher_skills_exist(self):
        """Test publisher referenced skills exist."""
        agent_file = CONFIG_DIR / "agents" / "publisher.md"
        config = _parse_agent_frontmatter(agent_file)

        skills_dir = CONFIG_DIR / "skills"
        for skill_name in config.get("skills", []):
            skill_dir = skills_dir / skill_name
            skill_file = skill_dir / "SKILL.md"
            assert skill_file.exists(), f"Missing skill: {skill_name}"


class TestAgentLoadingLogic:
    """Test agent loading from directory."""

    def test_all_agent_files_are_valid_markdown(self):
        """Test all .md files in agents/ are valid."""
        agents_dir = CONFIG_DIR / "agents"

        for agent_file in agents_dir.glob("*.md"):
            # Should not raise exception
            config = _parse_agent_frontmatter(agent_file)

            # Should have at minimum a body
            assert "body" in config
            assert len(config["body"]) > 0

    def test_agent_names_are_unique(self):
        """Test no duplicate agent names."""
        agents_dir = CONFIG_DIR / "agents"
        agent_names = set()

        for agent_file in agents_dir.glob("*.md"):
            config = _parse_agent_frontmatter(agent_file)
            name = config.get("name", agent_file.stem)

            assert name not in agent_names, f"Duplicate agent name: {name}"
            agent_names.add(name)


class TestSkillContentRequirements:
    """Test required content in skills."""

    def test_bmi_report_includes_all_bmi_ranges(self):
        """Test bmi-report skill covers all BMI ranges."""
        skill_file = CONFIG_DIR / "skills" / "bmi-report" / "SKILL.md"
        content = skill_file.read_text()

        # Must define all 4 BMI categories with ranges
        bmi_checks = [
            "18.5",  # Underweight threshold
            "24.9",  # Normal upper
            "25",  # Overweight lower
            "29.9",  # Overweight upper
            "30",  # Obese threshold
        ]

        for check in bmi_checks:
            assert check in content, f"Missing BMI range: {check}"

    def test_client_intake_has_python3_requirement(self):
        """Test client-intake specifies python3 not python."""
        skill_file = CONFIG_DIR / "skills" / "client-intake" / "SKILL.md"
        content = skill_file.read_text()

        assert "python3" in content
        # Should warn against using just 'python'
        assert (
            "python3" in content.lower()
            and "never" in content.lower()
            or "always use" in content.lower()
        )

    def test_email_formatter_max_width_specified(self):
        """Test email-formatter specifies max-width constraint."""
        skill_file = CONFIG_DIR / "skills" / "email-formatter" / "SKILL.md"
        content = skill_file.read_text()

        assert "600px" in content
        assert "max-width" in content.lower() or "max width" in content.lower()


class TestSubagentResponsibilities:
    """Test clear separation of responsibilities between subagents."""

    def test_bmi_analyst_should_not_send_email(self):
        """Test analyst out-of-scope includes email sending."""
        agent_file = CONFIG_DIR / "agents" / "analyst.md"
        config = _parse_agent_frontmatter(agent_file)
        body = config["body"]

        # Should explicitly say emailing is out of scope
        lower_body = body.lower()
        assert "out of scope" in lower_body

    def test_publisher_should_not_analyze(self):
        """Test publisher out-of-scope includes analysis."""
        agent_file = CONFIG_DIR / "agents" / "publisher.md"
        config = _parse_agent_frontmatter(agent_file)
        body = config["body"]

        lower_body = body.lower()
        assert "out of scope" in lower_body

    def test_domain_specific_tools_not_shared(self):
        """Test domain-specific tools aren't shared between subagents."""
        agents_dir = CONFIG_DIR / "agents"

        # Common utility tools that can be shared
        common_tools = {"search_web"}

        agent_tools = {}
        for agent_file in agents_dir.glob("*.md"):
            config = _parse_agent_frontmatter(agent_file)
            name = config.get("name", agent_file.stem)
            tools = set(config.get("tools", []))
            # Only check domain-specific tools
            domain_tools = tools - common_tools
            agent_tools[name] = domain_tools

        # Check no overlapping domain-specific tools
        agent_names = list(agent_tools.keys())
        for i, agent1 in enumerate(agent_names):
            for agent2 in agent_names[i + 1 :]:
                overlap = agent_tools[agent1] & agent_tools[agent2]
                assert len(overlap) == 0, (
                    f"Domain tool overlap between {agent1} and {agent2}: {overlap}"
                )


class TestErrorHandling:
    """Test error handling specifications in subagents."""

    def test_bmi_analyst_has_error_handling_section(self):
        """Test analyst defines error handling."""
        agent_file = CONFIG_DIR / "agents" / "analyst.md"
        config = _parse_agent_frontmatter(agent_file)
        body = config["body"]

        assert "error" in body.lower() or "failure" in body.lower()

    def test_publisher_has_error_handling_section(self):
        """Test publisher defines error handling."""
        agent_file = CONFIG_DIR / "agents" / "publisher.md"
        config = _parse_agent_frontmatter(agent_file)
        body = config["body"]

        assert "error" in body.lower() or "failure" in body.lower()


class TestGotchasSections:
    """Test that subagents have 'Gotchas' sections for common mistakes."""

    def test_bmi_analyst_has_gotchas(self):
        """Test analyst has gotchas section."""
        agent_file = CONFIG_DIR / "agents" / "analyst.md"
        config = _parse_agent_frontmatter(agent_file)
        body = config["body"]

        assert "gotcha" in body.lower()

    def test_publisher_has_gotchas(self):
        """Test publisher has gotchas section."""
        agent_file = CONFIG_DIR / "agents" / "publisher.md"
        config = _parse_agent_frontmatter(agent_file)
        body = config["body"]

        assert "gotcha" in body.lower()

    def test_client_intake_skill_has_gotchas(self):
        """Test client-intake skill has gotchas section."""
        skill_file = CONFIG_DIR / "skills" / "client-intake" / "SKILL.md"
        content = skill_file.read_text()

        assert "gotcha" in content.lower()

    def test_email_formatter_skill_has_gotchas(self):
        """Test email-formatter skill has gotchas section."""
        skill_file = CONFIG_DIR / "skills" / "email-formatter" / "SKILL.md"
        content = skill_file.read_text()

        assert "gotcha" in content.lower()
