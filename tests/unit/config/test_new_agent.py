"""Tests for `make new-agent`.

The shipped config is a health assistant. A fork that starts from it
and forgets to replace something ships someone else's agent; the
scaffolder exists so the starting point is clean rather than
nearly-clean.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).parents[3]


@pytest.fixture
def scaffolded(tmp_path: Path) -> Path:
    import sys

    sys.path.insert(0, str(REPO))
    from scripts.new_agent import scaffold

    out = tmp_path / "agent"
    scaffold("Invoice Assistant", out)
    return out


class TestOutput:
    def test_creates_the_required_files(self, scaffolded: Path):
        assert (scaffolded / "PROMPT.md").is_file()
        assert (scaffolded / "runtime" / "agent.yaml").is_file()
        assert (scaffolded / "integrations" / "mcp.json").is_file()

    def test_writes_mcp_at_the_preferred_path(self, scaffolded: Path):
        """A new agent should not be born owing a rename."""
        assert not (scaffolded / "mcp.json").exists()
        assert (scaffolded / "integrations" / "mcp.json").is_file()

    def test_carries_over_settings_worth_keeping(self, scaffolded: Path):
        for relative in (
            "runtime/observability.yaml",
            "runtime/router.yaml",
            "compliance/policies",
        ):
            assert (scaffolded / relative).exists(), relative

    def test_skills_and_subagents_start_empty(self, scaffolded: Path):
        """The demo's three skills must not come along."""
        for directory in ("skills", "subagents"):
            entries = [
                p for p in (scaffolded / directory).iterdir() if p.name != ".gitkeep"
            ]
            assert entries == [], f"{directory} is not empty"


class TestNoDemoContentSurvives:
    def test_agent_name_is_the_new_one(self, scaffolded: Path):
        import yaml

        config = yaml.safe_load((scaffolded / "runtime" / "agent.yaml").read_text())
        assert config["name"] == "Invoice Assistant"

    @pytest.mark.parametrize("needle", ["Health Assistant", "BMI", "fitness"])
    def test_demo_strings_are_gone(self, scaffolded: Path, needle: str):
        for path in scaffolded.rglob("*"):
            if not path.is_file():
                continue
            assert needle not in path.read_text(errors="ignore"), (
                f"{needle!r} survived in {path.relative_to(scaffolded)}"
            )

    def test_mcp_servers_start_empty(self, scaffolded: Path):
        import json

        data = json.loads((scaffolded / "integrations" / "mcp.json").read_text())
        assert data["mcpServers"] == {}


class TestValidatesClean:
    def test_scaffolded_config_has_no_errors_and_no_warnings(self, scaffolded: Path):
        """This is the contract: the example warns, a new agent does not."""
        from deep_agent.src.agent.config.validate import validate_config

        report = validate_config(scaffolded)

        assert report.errors == [], [e.message for e in report.errors]
        assert report.warnings == [], [w.message for w in report.warnings]


class TestSafety:
    def test_refuses_to_overwrite_a_populated_directory(self, tmp_path: Path):
        import sys

        sys.path.insert(0, str(REPO))
        from scripts.new_agent import scaffold

        out = tmp_path / "agent"
        out.mkdir()
        (out / "PROMPT.md").write_text("mine")

        with pytest.raises(FileExistsError):
            scaffold("Other", out)

    def test_force_overwrites(self, tmp_path: Path):
        import sys

        sys.path.insert(0, str(REPO))
        from scripts.new_agent import scaffold

        out = tmp_path / "agent"
        out.mkdir()
        (out / "PROMPT.md").write_text("mine")

        scaffold("Other", out, force=True)
        assert "Other" in (out / "PROMPT.md").read_text()
