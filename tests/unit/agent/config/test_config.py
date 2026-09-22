"""Unit tests for agent_config skill path resolution."""

import pytest
from unittest.mock import patch

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


class TestLoadGuardrailsConfig:
    """Tests for _load_guardrails_config taking agent_yaml_guardrail dict."""

    def setup_method(self):
        """Reset the singleton before each test."""
        AgentConfig._instance = None

    def _make_config_dir(self, tmp_path):
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()
        (config_dir / "PROMPT.md").write_text("""---
name: test-agent
model: gemini-2.5-flash
---

Test prompt.
""")
        return config_dir

    def test_disabled_when_section_absent(self, tmp_path):
        """Passing None disables guardrails."""
        config_dir = self._make_config_dir(tmp_path)
        cfg = AgentConfig(config_dir)
        result = cfg._load_guardrails_config(None)
        assert result.enabled is False

    def test_disabled_when_enabled_false(self, tmp_path):
        """Passing enabled=False disables guardrails."""
        config_dir = self._make_config_dir(tmp_path)
        cfg = AgentConfig(config_dir)
        result = cfg._load_guardrails_config({"enabled": False})
        assert result.enabled is False

    def test_enabled_when_enabled_true(self, tmp_path):
        """Passing enabled=True with a model enables guardrails."""
        config_dir = self._make_config_dir(tmp_path)
        cfg = AgentConfig(config_dir)
        result = cfg._load_guardrails_config(
            {"enabled": True, "model": "ibm-granite/granite-guardian-3.2-5b"}
        )
        assert result.enabled is True
        assert result.model == "ibm-granite/granite-guardian-3.2-5b"

    def test_disabled_on_parse_failure(self, tmp_path):
        """A parse failure disables guardrails rather than crashing."""
        config_dir = self._make_config_dir(tmp_path)
        cfg = AgentConfig(config_dir)
        with patch(
            "deep_agent.src.guardrails.config.GuardrailsConfig.model_validate",
            side_effect=Exception("bad"),
        ):
            result = cfg._load_guardrails_config({"enabled": True, "model": "x"})
        assert result.enabled is False


class TestCatalogueExclusion:
    """Tests for AgentConfig.exclude_subagent / exclude_skill (OFFSEC-379)."""

    def setup_method(self):
        """Reset the singleton before each test."""
        AgentConfig._instance = None

    def _make_config_dir(self, tmp_path):
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()

        skills_dir = config_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "safe-skill").mkdir()
        (skills_dir / "unsafe-skill").mkdir()

        (config_dir / "PROMPT.md").write_text("""---
name: orchestrator
model: gemini-2.5-flash
skills:
  - safe-skill
  - unsafe-skill
---

Orchestrator prompt.
""")

        subagents_dir = config_dir / "subagents"
        subagents_dir.mkdir()
        (subagents_dir / "analyst.md").write_text("""---
name: analyst
model: gemini-2.5-flash
description: Analyzes things.
skills:
  - safe-skill
  - unsafe-skill
---

Analyst prompt.
""")
        (subagents_dir / "researcher.md").write_text("""---
name: researcher
model: gemini-2.5-flash
description: Researches things.
---

Researcher prompt.
""")
        return config_dir

    def test_get_available_skills_returns_copy(self, tmp_path):
        config_dir = self._make_config_dir(tmp_path)
        cfg = AgentConfig(config_dir)

        skills = cfg.get_available_skills()
        assert set(skills) == {"safe-skill", "unsafe-skill"}

        # Mutating the returned dict must not affect internal state.
        skills.pop("safe-skill")
        assert "safe-skill" in cfg.get_available_skills()

    def test_exclude_subagent_removes_it(self, tmp_path):
        config_dir = self._make_config_dir(tmp_path)
        cfg = AgentConfig(config_dir)

        assert "researcher" in cfg.get_all_subagent_configs()
        removed = cfg.exclude_subagent("researcher", reason="unsafe")
        assert removed is True
        assert "researcher" not in cfg.get_all_subagent_configs()

    def test_exclude_subagent_missing_returns_false(self, tmp_path):
        config_dir = self._make_config_dir(tmp_path)
        cfg = AgentConfig(config_dir)

        assert cfg.exclude_subagent("does-not-exist") is False

    def test_exclude_skill_removes_it_and_scrubs_paths(self, tmp_path):
        config_dir = self._make_config_dir(tmp_path)
        cfg = AgentConfig(config_dir)

        orchestrator = cfg.get_orchestrator_config()
        analyst = cfg.get_all_subagent_configs()["analyst"]
        assert len(orchestrator["skill_paths"]) == 2
        assert len(analyst["skill_paths"]) == 2

        removed = cfg.exclude_skill("unsafe-skill", reason="injection detected")
        assert removed is True

        assert "unsafe-skill" not in cfg.get_available_skills()

        # The excluded skill's path must be scrubbed from every config that
        # had already resolved it, while the safe skill's path remains.
        orchestrator = cfg.get_orchestrator_config()
        analyst = cfg.get_all_subagent_configs()["analyst"]
        assert len(orchestrator["skill_paths"]) == 1
        assert "unsafe-skill" not in orchestrator["skill_paths"][0]
        assert "safe-skill" in orchestrator["skill_paths"][0]
        assert len(analyst["skill_paths"]) == 1
        assert "unsafe-skill" not in analyst["skill_paths"][0]

    def test_exclude_skill_missing_returns_false(self, tmp_path):
        config_dir = self._make_config_dir(tmp_path)
        cfg = AgentConfig(config_dir)

        assert cfg.exclude_skill("does-not-exist") is False

    def test_exclusions_survive_config_auto_reload(self, tmp_path):
        """CONFIG_AUTO_RELOAD (default True) must not resurrect excluded items.

        Regression test for OFFSEC-379: _ensure_loaded() reloads everything
        from disk on every access when CONFIG_AUTO_RELOAD is set, which would
        otherwise silently undo exclude_subagent/exclude_skill on the very
        next getter call.
        """
        config_dir = self._make_config_dir(tmp_path)
        cfg = AgentConfig(config_dir)

        cfg.exclude_subagent("researcher", reason="unsafe")
        cfg.exclude_skill("unsafe-skill", reason="injection detected")

        with patch("deep_agent.src.agent.config.loader.settings") as mock_settings:
            mock_settings.CONFIG_AUTO_RELOAD = True

            # Force a reload from disk, simulating a later request that
            # triggers CONFIG_AUTO_RELOAD after the safety scan already ran.
            subs = cfg.get_all_subagent_configs()
            skills = cfg.get_available_skills()
            orchestrator = cfg.get_orchestrator_config()

        # "researcher.md" is still on disk, but the reload must re-apply
        # the exclusion rather than resurrecting it.
        assert "researcher" not in subs
        assert "researcher" not in cfg.get_all_subagent_configs()
        assert "unsafe-skill" not in skills
        assert "analyst" in subs
        assert all("unsafe-skill" not in p for p in orchestrator["skill_paths"])
        assert all(
            "unsafe-skill" not in p for p in subs["analyst"]["skill_paths"]
        )
