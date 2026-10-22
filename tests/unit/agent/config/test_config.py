"""Unit tests for agent_config skill path resolution."""

from unittest.mock import patch

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

    def test_available_skills_reads_each_skill_instruction_file(self, tmp_path):
        """Expose complete skill instructions and tolerate an incomplete skill."""
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()
        (config_dir / "PROMPT.md").write_text(
            "---\nname: test-orchestrator\n---\nTest prompt.\n"
        )
        skills_dir = config_dir / "skills"
        skills_dir.mkdir()

        documented_skill = skills_dir / "client-intake"
        documented_skill.mkdir()
        (documented_skill / "SKILL.md").write_text(
            "# Client intake\n\nGather requirements.\n"
        )

        undocumented_skill = skills_dir / "draft"
        undocumented_skill.mkdir()

        contents = AgentConfig(config_dir).get_available_skills()

        assert contents == {
            "client-intake": "# Client intake\n\nGather requirements.\n",
            "draft": "",
        }


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


class TestResourcesValidation:
    """Test optional resources URI allowlist frontmatter."""

    def setup_method(self):
        AgentConfig._instance = None

    def test_orchestrator_omits_resources_key(self, tmp_path):
        """Missing resources means allow-all later — do not default to []."""
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()
        (config_dir / "skills").mkdir()

        (config_dir / "PROMPT.md").write_text("""---
name: orch
model: gemini-2.5-flash
---
Orchestrator.
""")

        cfg = AgentConfig(config_dir)
        orch = cfg.get_orchestrator_config()
        assert "resources" not in orch

    def test_orchestrator_valid_resources(self, tmp_path):
        """Valid resources list of URI strings loads without error."""
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()
        (config_dir / "skills").mkdir()

        (config_dir / "PROMPT.md").write_text("""---
name: orch
model: gemini-2.5-flash
resources:
  - template://about
  - template://echo/{text}
---
Orchestrator.
""")

        cfg = AgentConfig(config_dir)
        orch = cfg.get_orchestrator_config()
        assert orch["resources"] == ["template://about", "template://echo/{text}"]

    def test_orchestrator_empty_resources_list(self, tmp_path):
        """Loader preserves empty list; wiring normalizes to unrestricted (same as omit)."""
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()
        (config_dir / "skills").mkdir()

        (config_dir / "PROMPT.md").write_text("""---
name: orch
model: gemini-2.5-flash
resources: []
---
Orchestrator.
""")

        cfg = AgentConfig(config_dir)
        orch = cfg.get_orchestrator_config()
        assert orch["resources"] == []

    def test_orchestrator_invalid_resources_raises(self, tmp_path):
        """Non-list resources raises AppException."""
        config_dir = tmp_path / "agent_config"
        config_dir.mkdir()
        (config_dir / "skills").mkdir()

        (config_dir / "PROMPT.md").write_text("""---
name: orch
model: gemini-2.5-flash
resources: "not-a-list"
---
Orchestrator.
""")

        with pytest.raises(AppException, match="must be a list of strings"):
            cfg = AgentConfig(config_dir)
            cfg.get_orchestrator_config()

    def test_subagent_valid_resources(self, tmp_path):
        """Subagent resources list of strings is kept on the config."""
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
        (sub_dir / "analyst.md").write_text("""---
name: analyst
model: gemini-2.5-flash
resources:
  - template://about
---
Analyst.
""")

        cfg = AgentConfig(config_dir)
        subs = cfg.get_all_subagent_configs()
        assert subs["analyst"]["resources"] == ["template://about"]

    def test_subagent_invalid_resources_is_skipped(self, tmp_path, caplog):
        """Subagent with non-string resources entries is skipped and logged."""
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
resources:
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


class TestValidateMcpServerApiKey:
    """_validate_mcp_server api_key branches (RHITAIF-894).

    Validation logs errors rather than raising, so a misconfigured MCP is
    visible at config load without failing the whole agent.
    """

    @staticmethod
    def _cfg(**overrides):
        base = {
            "auth_mode": "api_key",
            "auth_env_var": "MCP_X_API_KEY",
            "api_key_name": "X-API-Key",
            "api_key_location": "header",
        }
        base.update(overrides)
        return base

    def test_valid_config_logs_no_error(self, caplog):
        import logging

        with caplog.at_level(logging.ERROR):
            AgentConfig._validate_mcp_server("m", self._cfg())
        assert caplog.text == ""

    def test_defaults_are_valid_when_optional_keys_absent(self, caplog):
        """Only auth_env_var is mandatory; location/scheme/name default cleanly."""
        import logging

        with caplog.at_level(logging.ERROR):
            AgentConfig._validate_mcp_server(
                "m", {"auth_mode": "api_key", "auth_env_var": "MCP_X_API_KEY"}
            )
        assert caplog.text == ""

    def test_missing_auth_env_var_logs_error(self, caplog):
        import logging

        with caplog.at_level(logging.ERROR):
            AgentConfig._validate_mcp_server("m", self._cfg(auth_env_var=""))
        assert "requires 'auth_env_var'" in caplog.text

    def test_invalid_location_logs_error(self, caplog):
        import logging

        with caplog.at_level(logging.ERROR):
            AgentConfig._validate_mcp_server("m", self._cfg(api_key_location="cookie"))
        assert "invalid api_key_location 'cookie'" in caplog.text

    def test_location_without_name_logs_error(self, caplog):
        import logging

        with caplog.at_level(logging.ERROR):
            AgentConfig._validate_mcp_server("m", self._cfg(api_key_name=""))
        assert "'api_key_name' is missing" in caplog.text

    def test_invalid_auth_mode_returns_before_api_key_checks(self, caplog):
        import logging

        with caplog.at_level(logging.ERROR):
            AgentConfig._validate_mcp_server("m", {"auth_mode": "magic"})
        assert "invalid auth_mode 'magic'" in caplog.text
        assert "auth_env_var" not in caplog.text
