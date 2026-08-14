"""Unit tests for PII config loading from pii.yaml."""

from deep_agent.src.agent.config import AgentConfig
from deep_agent.src.agent.config.middleware import PIIConfig, PIIRule


class TestPIIConfigModel:
    """Test PIIConfig and PIIRule Pydantic models."""

    def test_pii_config_defaults_to_disabled(self):
        config = PIIConfig()
        assert config.enabled is False
        assert config.rules == []
        assert config.trace_strategy == "hash"

    def test_pii_rule_requires_name(self):
        rule = PIIRule(name="email")
        assert rule.name == "email"
        assert rule.strategy == "redact"
        assert rule.provider == "default"

    def test_pii_rule_normalises_legacy_type_field(self):
        rule = PIIRule.model_validate({"type": "credit_card", "strategy": "mask"})
        assert rule.name == "credit_card"

    def test_pii_rule_custom_regex(self):
        rule = PIIRule(
            name="pan_card",
            strategy="block",
            provider="custom",
            regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        )
        assert rule.regex == r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"

    def test_pii_config_from_dict(self):
        config = PIIConfig.model_validate(
            {
                "enabled": True,
                "trace_strategy": "redact",
                "rules": [{"name": "email", "strategy": "scrub", "provider": "regex"}],
            }
        )
        assert config.enabled is True
        assert config.trace_strategy == "redact"
        assert len(config.rules) == 1
        assert config.rules[0].name == "email"


class TestLoadPIIConfig:
    """Test AgentConfig._load_pii_config() — mirrors the observability.yaml pattern."""

    def setup_method(self):
        AgentConfig._instance = None

    def _make_config_dir(self, tmp_path):
        config_dir = tmp_path / "agent"
        (config_dir / "runtime").mkdir(parents=True)
        (config_dir / "PROMPT.md").write_text(
            "---\nname: test\nmodel: gemini-2.5-flash\n---\nPrompt.\n"
        )
        return config_dir

    def test_no_file_returns_disabled(self, tmp_path):
        config_dir = self._make_config_dir(tmp_path)
        cfg = AgentConfig(config_dir)
        pii = cfg.get_custom_pii_config()
        assert pii.enabled is False
        assert pii.rules == []

    def test_enabled_false_returns_disabled(self, tmp_path):
        config_dir = self._make_config_dir(tmp_path)
        (config_dir / "runtime" / "pii.yaml").write_text("enabled: false\nrules: []\n")
        cfg = AgentConfig(config_dir)
        pii = cfg.get_custom_pii_config()
        assert pii.enabled is False

    def test_enabled_true_loads_rules(self, tmp_path):
        config_dir = self._make_config_dir(tmp_path)
        (config_dir / "runtime" / "pii.yaml").write_text(
            "enabled: true\n"
            "trace_strategy: hash\n"
            "rules:\n"
            "  - name: email\n"
            "    strategy: scrub\n"
            "    provider: regex\n"
            "  - name: credit_card\n"
            "    strategy: mask\n"
            "    provider: default\n"
        )
        cfg = AgentConfig(config_dir)
        pii = cfg.get_custom_pii_config()
        assert pii.enabled is True
        assert pii.trace_strategy == "hash"
        assert len(pii.rules) == 2
        assert pii.rules[0].name == "email"
        assert pii.rules[1].name == "credit_card"

    def test_invalid_yaml_falls_back_to_disabled(self, tmp_path):
        config_dir = self._make_config_dir(tmp_path)
        (config_dir / "runtime" / "pii.yaml").write_text("not: [valid: yaml: {{")
        cfg = AgentConfig(config_dir)
        pii = cfg.get_custom_pii_config()
        assert pii.enabled is False

    def test_custom_rule_with_regex(self, tmp_path):
        config_dir = self._make_config_dir(tmp_path)
        (config_dir / "runtime" / "pii.yaml").write_text(
            "enabled: true\n"
            "rules:\n"
            r"  - name: pan_card" + "\n"
            r"    strategy: block" + "\n"
            r"    provider: custom" + "\n"
            r"    regex: '\b[A-Z]{5}[0-9]{4}[A-Z]\b'" + "\n"
        )
        cfg = AgentConfig(config_dir)
        pii = cfg.get_custom_pii_config()
        assert pii.rules[0].name == "pan_card"
        assert pii.rules[0].strategy == "block"
        assert pii.rules[0].regex is not None
