"""Loading channels.yaml, including when it is not there.

A channel is the agent's public surface, so every path that fails to
read the file has to fail closed: no channels, rather than an
exception that becomes a 500 on every channel request.
"""

from __future__ import annotations

import pytest

from deep_agent.src.channels.loader import load_channels, reset_cache


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_cache()
    yield
    reset_cache()


def _write(path, text: str) -> None:
    path.write_text(text)


class TestLoadChannels:
    def test_no_file_means_no_channels(self, monkeypatch):
        import deep_agent.src.agent.config.paths as paths

        monkeypatch.setattr(paths, "config_file", lambda _name: None)
        assert load_channels().channels == {}

    def test_a_valid_file_is_read(self, monkeypatch, tmp_path):
        import deep_agent.src.agent.config.paths as paths

        target = tmp_path / "channels.yaml"
        _write(
            target,
            "channels:\n"
            "  slack:\n"
            "    type: slack\n"
            "    signing_secret_env: SLACK_SIGNING_SECRET\n",
        )
        monkeypatch.setattr(paths, "config_file", lambda _name: target)

        assert "slack" in load_channels().channels

    def test_the_result_is_cached_on_mtime(self, monkeypatch, tmp_path):
        import deep_agent.src.agent.config.paths as paths

        target = tmp_path / "channels.yaml"
        _write(
            target,
            "channels:\n"
            "  slack:\n"
            "    type: slack\n"
            "    signing_secret_env: SLACK_SIGNING_SECRET\n",
        )
        monkeypatch.setattr(paths, "config_file", lambda _name: target)

        first = load_channels()
        assert load_channels() is first

    def test_invalid_yaml_disables_every_channel(self, monkeypatch, tmp_path):
        import deep_agent.src.agent.config.paths as paths

        target = tmp_path / "channels.yaml"
        _write(target, "channels: [this is not a mapping")
        monkeypatch.setattr(paths, "config_file", lambda _name: target)

        assert load_channels().channels == {}

    def test_a_file_that_vanishes_mid_load_fails_closed(self, monkeypatch, tmp_path):
        """Deleted between `config_file` and `stat`.

        The raised FileNotFoundError used to turn every channel
        request into a 500, which is a worse answer than no channels.
        """
        import deep_agent.src.agent.config.paths as paths

        missing = tmp_path / "gone.yaml"
        monkeypatch.setattr(paths, "config_file", lambda _name: missing)

        assert load_channels().channels == {}
