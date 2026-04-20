"""Unit tests for agent_config skill path resolution."""

from pathlib import Path

import pytest

from template_agent.src.agent.config import AgentConfig


class TestAgentConfigSkillResolution:
    """Test that skills are resolved during config loading."""

    def setup_method(self):
        """Reset the singleton before each test."""
        AgentConfig._instance = None

    def test_orchestrator_loads_with_skill_paths(self, tmp_path):
        """Test that orchestrator config includes resolved skill paths."""
        # Create mock directory structure
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()

        skills_dir = config_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "client-intake").mkdir()

        orchestrator_dir = config_dir / "orchestrator"
        orchestrator_dir.mkdir()

        main_md = orchestrator_dir / "main.md"
        main_md.write_text("""---
name: test-orchestrator
model: gemini-2.5-flash
skills:
  - client-intake
---

Test orchestrator prompt.
""")

        # Initialize agent config with test directory
        agent_cfg = AgentConfig(config_dir)
        orchestrator = agent_cfg.get_orchestrator_config()

        # Verify skill_paths are resolved
        assert "skill_paths" in orchestrator
        assert len(orchestrator["skill_paths"]) == 1
        assert "client-intake" in orchestrator["skill_paths"][0]

    def test_subagent_loads_with_skill_paths(self, tmp_path):
        """Test that subagent configs include resolved skill paths."""
        # Create mock directory structure
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()

        skills_dir = config_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "bmi-report").mkdir()

        # Create minimal orchestrator (required by _ensure_loaded)
        orchestrator_dir = config_dir / "orchestrator"
        orchestrator_dir.mkdir()
        (orchestrator_dir / "main.md").write_text("""---
name: orchestrator
model: gemini-2.5-flash
---
Minimal orchestrator.
""")

        subagents_dir = config_dir / "subagents"
        subagents_dir.mkdir()

        analyst_md = subagents_dir / "analyst.md"
        analyst_md.write_text("""---
name: analyst
model: gemini-2.5-flash
skills:
  - bmi-report
---

Test analyst prompt.
""")

        # Initialize agent config with test directory
        agent_cfg = AgentConfig(config_dir)
        subagents = agent_cfg.get_all_subagent_configs()

        # Verify skill_paths are resolved
        assert "analyst" in subagents
        assert "skill_paths" in subagents["analyst"]
        assert len(subagents["analyst"]["skill_paths"]) == 1
        assert "bmi-report" in subagents["analyst"]["skill_paths"][0]

    def test_missing_skills_are_logged(self, tmp_path, caplog):
        """Test that missing skills generate warnings."""
        # Create mock directory structure
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()

        skills_dir = config_dir / "skills"
        skills_dir.mkdir()
        # Don't create the skill directory

        orchestrator_dir = config_dir / "orchestrator"
        orchestrator_dir.mkdir()

        main_md = orchestrator_dir / "main.md"
        main_md.write_text("""---
name: test-orchestrator
model: gemini-2.5-flash
skills:
  - nonexistent-skill
---

Test orchestrator prompt.
""")

        # Initialize agent config with test directory
        agent_cfg = AgentConfig(config_dir)
        orchestrator = agent_cfg.get_orchestrator_config()

        # Verify skill_paths is empty or not present for missing skills
        skill_paths = orchestrator.get("skill_paths", [])
        assert len(skill_paths) == 0

        # Verify warning was logged
        assert "unknown skills" in caplog.text.lower()
