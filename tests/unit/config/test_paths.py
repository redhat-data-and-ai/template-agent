"""Tests for config path resolution.

The problem these solve: a layout can always be argued about, but a
layout that cannot change without breaking every forked agent is a trap.
Renames go through the resolver, so an old path keeps working and warns.
"""

from pathlib import Path

import pytest

from deep_agent.src.agent.config import paths as cfg_paths


@pytest.fixture(autouse=True)
def _reset_warnings():
    cfg_paths._warned.clear()
    yield
    cfg_paths._warned.clear()


@pytest.fixture
def root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path))
    return tmp_path


class TestResolution:
    def test_preferred_path_wins(self, root: Path):
        (root / "integrations").mkdir()
        (root / "integrations" / "mcp.json").write_text("{}")
        (root / "mcp.json").write_text("{}")

        assert cfg_paths.config_file("mcp") == root / "integrations" / "mcp.json"

    def test_legacy_path_still_resolves(self, root: Path):
        """A fork on the old layout must keep working."""
        (root / "mcp.json").write_text("{}")

        assert cfg_paths.config_file("mcp") == root / "mcp.json"

    def test_absent_file_returns_none(self, root: Path):
        assert cfg_paths.config_file("mcp") is None

    def test_unknown_name_is_a_programming_error(self, root: Path):
        with pytest.raises(KeyError):
            cfg_paths.config_file("not-a-config")

    def test_directories_resolve_too(self, root: Path):
        (root / "skills").mkdir()
        assert cfg_paths.config_dir("skills") == root / "skills"

    def test_a_file_does_not_satisfy_a_directory_lookup(self, root: Path):
        (root / "skills").write_text("oops")
        assert cfg_paths.config_dir("skills") is None


class TestLegacyWarning:
    def test_legacy_use_warns_once(self, root: Path, caplog):
        (root / "mcp.json").write_text("{}")

        with caplog.at_level("WARNING"):
            cfg_paths.config_file("mcp")
            cfg_paths.config_file("mcp")

        warnings = [r for r in caplog.records if "legacy path" in r.getMessage()]
        assert len(warnings) == 1, "should warn once per name, not per read"

    def test_warning_names_the_new_path(self, root: Path, caplog):
        (root / "mcp.json").write_text("{}")

        with caplog.at_level("WARNING"):
            cfg_paths.config_file("mcp")

        assert any("integrations/mcp.json" in r.getMessage() for r in caplog.records)

    def test_preferred_path_does_not_warn(self, root: Path, caplog):
        (root / "integrations").mkdir()
        (root / "integrations" / "mcp.json").write_text("{}")

        with caplog.at_level("WARNING"):
            cfg_paths.config_file("mcp")

        assert not [r for r in caplog.records if "legacy path" in r.getMessage()]


class TestPreferredPath:
    def test_reports_the_new_layout_regardless_of_what_exists(self, root: Path):
        """Scaffolding and error messages should name the new layout."""
        (root / "mcp.json").write_text("{}")

        assert cfg_paths.preferred_path("mcp") == root / "integrations" / "mcp.json"


class TestLayoutReport:
    def test_flags_legacy_entries(self, root: Path):
        (root / "mcp.json").write_text("{}")

        report = cfg_paths.describe_layout()

        assert report["mcp"]["legacy"] is True
        assert report["mcp"]["present"] is True

    def test_absent_entries_are_neither_present_nor_legacy(self, root: Path):
        report = cfg_paths.describe_layout()

        assert report["mcp"]["present"] is False
        assert report["mcp"]["legacy"] is False


class TestRenameIsNonBreaking:
    """The acceptance test for this whole mechanism.

    Asserts at the resolver level rather than through `AgentConfig`:
    `_load_mcp_servers` catches `Exception` and returns `{}`, so a loader
    failure there is indistinguishable from "no servers configured" and
    would make this test pass for the wrong reason.
    """

    def test_same_file_resolves_under_either_layout(self, tmp_path: Path, monkeypatch):
        import shutil

        # The shipped layout is now the preferred one, so the legacy
        # copy is synthesised from it rather than the other way round.
        shipped = Path(__file__).parents[3] / "config" / "agent"
        source = shipped / "integrations" / "mcp.json"

        legacy = tmp_path / "legacy"
        legacy.mkdir()
        shutil.copy(source, legacy / "mcp.json")

        renamed = tmp_path / "renamed"
        (renamed / "integrations").mkdir(parents=True)
        shutil.copy(source, renamed / "integrations" / "mcp.json")

        cfg_paths._warned.add("mcp")

        monkeypatch.setenv("CONFIG_PATH", str(legacy))
        before = cfg_paths.config_file("mcp")

        monkeypatch.setenv("CONFIG_PATH", str(renamed))
        after = cfg_paths.config_file("mcp")

        assert before is not None and after is not None
        assert before.name == after.name
        assert before.read_text() == after.read_text()

    def test_loader_still_reads_a_legacy_mcp_file(self, tmp_path: Path, monkeypatch):
        """End to end through the loader, on a fresh singleton.

        `AgentConfig` is a singleton, so it must be reset here — otherwise
        this silently asserts against whichever instance an earlier test
        created.
        """
        import shutil

        from deep_agent.src.agent.config.loader import AgentConfig

        # The shipped config now uses integrations/, so the direction
        # worth proving is the other one: a fork still on the old flat
        # layout must keep loading.
        shipped = Path(__file__).parents[3] / "config" / "agent"
        legacy = tmp_path / "legacy"
        shutil.copytree(shipped, legacy)
        shutil.move(str(legacy / "integrations" / "mcp.json"), legacy / "mcp.json")

        cfg_paths._warned.add("mcp")
        monkeypatch.setenv("CONFIG_PATH", str(legacy))
        monkeypatch.setattr(AgentConfig, "_instance", None)

        servers = AgentConfig(base_dir=legacy).get_mcp_servers()

        # Empty would mean the loader swallowed an error rather than
        # finding the legacy file; both look the same from outside.
        assert servers, "legacy mcp.json did not load (loader fails open to {})"
