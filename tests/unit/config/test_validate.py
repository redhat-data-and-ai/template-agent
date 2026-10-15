"""Tests for `make validate-config`.

The templates are config-driven, so a mistake in `config/` must be caught
up front rather than surfacing as a runtime error or a silent fallback.
"""

from pathlib import Path

import pytest
import yaml

from deep_agent.src.agent.config.validate import validate_config

SHIPPED = Path(__file__).parents[3] / "config" / "agent"


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    """A writable copy of the shipped example config."""
    import shutil

    target = tmp_path / "agent"
    shutil.copytree(SHIPPED, target)
    return target


class TestShippedConfig:
    def test_example_config_has_no_errors(self):
        """The shipped example must validate — it is what forks start from."""
        assert validate_config(SHIPPED).errors == []

    def test_example_config_is_flagged_as_unmodified(self):
        """Forking and shipping without editing config ships someone else's
        agent, so the example content warns."""
        # The agent's own name and prompt are what a fork must change.
        # The shipped skills are deliberately not flagged — they are
        # reusable rather than tutorial content, so warning about them
        # would train people to ignore this report.
        warnings = {w.message for w in validate_config(SHIPPED).warnings}
        assert any("still the example's" in w for w in warnings)


class TestFieldValidation:
    def test_invalid_enum_is_reported_with_its_path(self, config_copy: Path):
        path = config_copy / "runtime" / "agent.yaml"
        raw = yaml.safe_load(path.read_text())
        raw["middleware"]["human_approval"]["policies"][0]["decision"] = "maybe"
        path.write_text(yaml.safe_dump(raw))

        errors = validate_config(config_copy).errors

        assert any("decision" in e.location for e in errors)
        assert any("allow" in e.message for e in errors)

    def test_wrong_type_is_reported(self, config_copy: Path):
        path = config_copy / "runtime" / "agent.yaml"
        raw = yaml.safe_load(path.read_text())
        raw["middleware"]["human_approval"]["enabled"] = "yes-please"
        path.write_text(yaml.safe_dump(raw))

        assert validate_config(config_copy).errors

    def test_malformed_yaml_is_reported_not_raised(self, config_copy: Path):
        (config_copy / "runtime" / "agent.yaml").write_text("key: [unclosed\n")

        report = validate_config(config_copy)

        assert any("valid YAML" in e.message for e in report.errors)

    def test_router_config_is_validated(self, config_copy: Path):
        """router.yaml rejects a single tier — there is nothing to route between."""
        path = config_copy / "runtime" / "router.yaml"
        raw = yaml.safe_load(path.read_text())
        raw["enabled"] = True
        raw["tiers"] = [{"name": "only", "model": "m", "criteria": "c"}]
        path.write_text(yaml.safe_dump(raw))

        assert validate_config(config_copy).errors


class TestRequiredFiles:
    def test_missing_prompt_is_an_error(self, config_copy: Path):
        (config_copy / "PROMPT.md").unlink()

        report = validate_config(config_copy)

        assert any(e.file == "PROMPT.md" for e in report.errors)

    def test_absent_config_directory_is_an_error(self, tmp_path: Path):
        assert validate_config(tmp_path / "nope").errors


class TestCleanFork:
    def test_a_fully_customised_config_produces_no_findings(self, config_copy: Path):
        """What a real agent should look like: no errors, no example warnings."""
        import shutil

        agent_yaml = config_copy / "runtime" / "agent.yaml"
        raw = yaml.safe_load(agent_yaml.read_text())
        raw["name"] = "Invoice Assistant"
        agent_yaml.write_text(yaml.safe_dump(raw))

        (config_copy / "PROMPT.md").write_text(
            "---\nname: orchestrator\ndescription: Handles invoices.\n---\n"
        )
        (config_copy / "mcp.json").write_text('{"mcpServers": {}}')
        shutil.rmtree(config_copy / "skills")
        (config_copy / "skills").mkdir()

        report = validate_config(config_copy)

        assert report.errors == []
        assert report.warnings == []


class TestExampleMarkersComeFromConfig:
    """The needles live in config, not in Python.

    A brand name in a .py is one a fork has to find in source; and the
    needle has to track whatever the shipped config carries anyway, so
    the two belong in the same place.
    """

    def test_a_renamed_example_is_caught_by_editing_config_alone(
        self, config_copy: Path
    ):
        markers = config_copy / "runtime" / "template-markers.yaml"
        markers.write_text(
            yaml.safe_dump(
                {
                    "markers": [
                        {
                            "needle": "Sales Copilot",
                            "file": "runtime/agent.yaml",
                            "message": "agent name is still the example's",
                        }
                    ]
                }
            )
        )
        agent_yaml = config_copy / "runtime" / "agent.yaml"
        raw = yaml.safe_load(agent_yaml.read_text())
        raw["name"] = "Sales Copilot"
        agent_yaml.write_text(yaml.safe_dump(raw))

        warnings = [w.message for w in validate_config(config_copy).warnings]
        assert any("still the example's" in w for w in warnings)

    def test_deleting_the_marker_file_turns_the_checks_off(self, config_copy: Path):
        # A reasonable thing for a finished fork to do, and an
        # explicit act rather than a silent one.
        (config_copy / "runtime" / "template-markers.yaml").unlink()

        warnings = [w.message for w in validate_config(config_copy).warnings]
        assert not any("still the example's" in w for w in warnings)

    def test_a_malformed_marker_file_does_not_fail_validation(self, config_copy: Path):
        (config_copy / "runtime" / "template-markers.yaml").write_text("{[not yaml")

        report = validate_config(config_copy)
        assert not any("still the example's" in w.message for w in report.warnings)

    def test_an_entry_missing_a_field_is_skipped(self, config_copy: Path):
        (config_copy / "runtime" / "template-markers.yaml").write_text(
            yaml.safe_dump({"markers": [{"needle": "Example Agent"}, "not a mapping"]})
        )

        report = validate_config(config_copy)
        assert not any("still the example's" in w.message for w in report.warnings)
