"""Unit tests for agent_config skill path resolution."""

import pytest

from deep_agent.src.agent.config import AgentConfig
from deep_agent.src.exceptions import AppException


class TestAgentConfigSkillResolution:
    """Test that skills are resolved during config loading."""

    def setup_method(self):
        """Reset the singleton before each test."""
        AgentConfig._instance = None

    def test_orchestrator_loads_with_skill_paths(self, tmp_path):
        """Test that orchestrator config includes resolved skill paths."""
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()

        skills_dir = config_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "client-intake").mkdir()

        prompt_md = config_dir / "PROMPT.md"
        prompt_md.write_text("""---
name: test-orchestrator
model: gemini-2.5-flash
skills:
  - client-intake
---

Test orchestrator prompt.
""")

        agent_cfg = AgentConfig(config_dir)
        orchestrator = agent_cfg.get_orchestrator_config()

        assert "skill_paths" in orchestrator
        assert len(orchestrator["skill_paths"]) == 1
        assert "client-intake" in orchestrator["skill_paths"][0]

    def test_subagent_loads_with_skill_paths(self, tmp_path):
        """Test that subagent configs include resolved skill paths."""
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()

        skills_dir = config_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "bmi-report").mkdir()

        prompt_md = config_dir / "PROMPT.md"
        prompt_md.write_text("""---
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

        agent_cfg = AgentConfig(config_dir)
        subagents = agent_cfg.get_all_subagent_configs()

        assert "analyst" in subagents
        assert "skill_paths" in subagents["analyst"]
        assert len(subagents["analyst"]["skill_paths"]) == 1
        assert "bmi-report" in subagents["analyst"]["skill_paths"][0]

    def test_missing_skills_are_logged(self, tmp_path, caplog):
        """Test that missing skills generate warnings."""
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()

        skills_dir = config_dir / "skills"
        skills_dir.mkdir()

        prompt_md = config_dir / "PROMPT.md"
        prompt_md.write_text("""---
name: test-orchestrator
model: gemini-2.5-flash
skills:
  - nonexistent-skill
---

Test orchestrator prompt.
""")

        agent_cfg = AgentConfig(config_dir)
        orchestrator = agent_cfg.get_orchestrator_config()

        skill_paths = orchestrator.get("skill_paths", [])
        assert len(skill_paths) == 0

        assert "unknown skills" in caplog.text.lower()


class TestMcpsValidation:
    """Test mcps field validation for orchestrator and subagents."""

    def setup_method(self):
        AgentConfig._instance = None

    def test_orchestrator_valid_mcps(self, tmp_path):
        """Valid mcps list of strings loads without error."""
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()
        (config_dir / "skills").mkdir()

        (config_dir / "PROMPT.md").write_text("""---
name: orch
model: gemini-2.5-flash
mcps:
  - web-search
  - dataverse-mcp
---
Orchestrator.
""")

        cfg = AgentConfig(config_dir)
        orch = cfg.get_orchestrator_config()
        assert orch["mcps"] == ["web-search", "dataverse-mcp"]

    def test_orchestrator_invalid_mcps_raises(self, tmp_path):
        """Non-list mcps raises AppException."""
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()
        (config_dir / "skills").mkdir()

        (config_dir / "PROMPT.md").write_text("""---
name: orch
model: gemini-2.5-flash
mcps: "not-a-list"
---
Orchestrator.
""")

        with pytest.raises(AppException, match="must be a list of strings"):
            cfg = AgentConfig(config_dir)
            cfg.get_orchestrator_config()

    def test_subagent_invalid_mcps_is_skipped(self, tmp_path, caplog):
        """Subagent with non-string mcps entries is skipped and logged."""
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()
        (config_dir / "skills").mkdir()

        (config_dir / "PROMPT.md").write_text("""---
name: orch
model: gemini-2.5-flash
---
Orchestrator.
""")

        sub_dir = config_dir / "subagents"
        sub_dir.mkdir()
        (sub_dir / "bad.md").write_text("""---
name: bad-agent
model: gemini-2.5-flash
mcps:
  - 123
---
Bad agent.
""")

        cfg = AgentConfig(config_dir)
        subs = cfg.get_all_subagent_configs()
        assert "bad-agent" not in subs
        assert "must be a list of strings" in caplog.text
