"""Config hot-reload must not re-parse the world on every access.

`CONFIG_AUTO_RELOAD` defaults on, and it used to mean "re-read every
YAML, subagent and skill from disk on every property access". Config is
read many times while building one graph, so that was roughly 250ms of
the time to first token on every request — in a template whose whole
pitch is that behaviour lives in config/.

These tests pin both halves: an edit is still picked up, and not editing
is cheap.
"""

from __future__ import annotations

import time

import pytest
import yaml

from deep_agent.src.agent.config.loader import AgentConfig


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    root = tmp_path / "agent"
    (root / "runtime").mkdir(parents=True)
    (root / "PROMPT.md").write_text("---\nmodel: gemini-2.5-pro\n---\nBe helpful.")
    (root / "mcp.json").write_text("{}")
    (root / "runtime" / "agent.yaml").write_text(yaml.safe_dump({"name": "First"}))

    monkeypatch.setenv("CONFIG_PATH", str(root))
    # AgentConfig is a singleton; without resetting it every test in the
    # session would share one loaded config.
    monkeypatch.setattr(AgentConfig, "_instance", None, raising=False)
    monkeypatch.setattr(AgentConfig, "_initialized", False, raising=False)
    return root


class TestReloadIsGatedOnMtime:
    def test_an_unchanged_tree_does_not_reparse(self, config_dir):
        config = AgentConfig(base_dir=config_dir)
        config._ensure_loaded()

        calls = {"n": 0}
        original = config._load_agent_yaml

        def counting():
            calls["n"] += 1
            return original()

        config._load_agent_yaml = counting
        for _ in range(20):
            config._ensure_loaded()

        assert calls["n"] == 0, (
            f"re-parsed config {calls['n']} times with nothing changed on disk"
        )

    def test_an_edit_is_picked_up(self, config_dir):
        """Hot reload is the feature; it must survive the optimisation."""
        config = AgentConfig(base_dir=config_dir)
        config._ensure_loaded()
        first = config._disk_signature()

        time.sleep(0.01)
        (config_dir / "runtime" / "agent.yaml").write_text(
            yaml.safe_dump({"name": "Second"})
        )

        assert config._disk_signature() != first

        config._ensure_loaded()
        assert config._configs_loaded

    def test_a_new_skill_file_counts_as_a_change(self, config_dir):
        """Adding a skill is a config change, not just editing one."""
        config = AgentConfig(base_dir=config_dir)
        config._ensure_loaded()
        before = config._disk_signature()

        skill = config_dir / "skills" / "new-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("# New")

        assert config._disk_signature() != before

    def test_an_unreadable_tree_does_not_pin_a_stale_config(self, config_dir):
        """Returning a stable signature on error would cache forever."""
        config = AgentConfig(base_dir=config_dir)
        config._base_dir = config_dir / "does-not-exist"
        assert config._disk_signature() != config._disk_signature()


class TestSignatureCost:
    def test_the_signature_is_cheap_enough_to_take_per_request(self, config_dir):
        config = AgentConfig(base_dir=config_dir)
        config._ensure_loaded()

        start = time.perf_counter()
        for _ in range(50):
            config._disk_signature()
        per_call_ms = (time.perf_counter() - start) * 1000 / 50

        # Generous: the point is that it is stat() calls, not YAML
        # parsing, so it should be well under a millisecond here.
        assert per_call_ms < 20, f"signature took {per_call_ms:.1f}ms per call"
