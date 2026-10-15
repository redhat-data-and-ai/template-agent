"""Channel config: the defaults are the security posture."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deep_agent.src.channels.config import (
    ChannelConfig,
    ChannelsFileConfig,
    IdentityConfig,
)


def a_channel(**kwargs) -> ChannelConfig:
    return ChannelConfig(type="webhook", signing_secret_env="SECRET_ENV", **kwargs)


class TestDefaults:
    def test_channels_are_off_unless_turned_on(self):
        """An inbound surface nobody decided to expose is a bug."""
        assert ChannelsFileConfig().enabled is False
        assert ChannelsFileConfig().active() == {}

    def test_enabling_the_feature_with_no_channel_is_an_error(self):
        with pytest.raises(ValidationError):
            ChannelsFileConfig(enabled=True, channels={})

    def test_a_disabled_channel_does_not_count_as_one(self):
        with pytest.raises(ValidationError):
            ChannelsFileConfig(enabled=True, channels={"w": a_channel(enabled=False)})

    def test_a_disabled_channel_is_not_active(self):
        config = ChannelsFileConfig(
            enabled=True,
            channels={"on": a_channel(), "off": a_channel(enabled=False)},
        )
        assert set(config.active()) == {"on"}


class TestSecretHandling:
    def test_the_config_names_an_env_var_not_a_secret(self, monkeypatch):
        """A field that can hold a secret eventually holds one."""
        monkeypatch.setenv("SECRET_ENV", "shhh")
        assert a_channel().signing_secret == "shhh"

    def test_an_unset_env_var_reads_as_none_not_empty(self, monkeypatch):
        monkeypatch.delenv("SECRET_ENV", raising=False)
        assert a_channel().signing_secret is None

    def test_the_secret_is_read_per_use_so_rotation_needs_no_restart(self, monkeypatch):
        channel = a_channel()
        monkeypatch.setenv("SECRET_ENV", "first")
        assert channel.signing_secret == "first"
        monkeypatch.setenv("SECRET_ENV", "second")
        assert channel.signing_secret == "second"


class TestIdentity:
    def test_an_unmapped_sender_resolves_to_nothing_by_default(self):
        """Rejecting is safer than a shared fallback identity."""
        assert IdentityConfig().resolve("whoever") is None

    def test_a_mapped_sender_resolves(self):
        identity = IdentityConfig(map={"U1": "alice@example.com"})
        assert identity.resolve("U1") == "alice@example.com"

    def test_a_default_user_catches_the_rest(self):
        identity = IdentityConfig(default_user="bot@example.com")
        assert identity.resolve("anyone") == "bot@example.com"

    def test_an_explicit_mapping_wins_over_the_default(self):
        identity = IdentityConfig(
            map={"U1": "alice@example.com"}, default_user="bot@example.com"
        )
        assert identity.resolve("U1") == "alice@example.com"


class TestBounds:
    @pytest.mark.parametrize("age", [0, 3601])
    def test_the_replay_window_is_bounded(self, age):
        """Zero disables freshness; an hour is long enough to be a replay."""
        with pytest.raises(ValidationError):
            a_channel(max_age_seconds=age)

    def test_run_timeout_is_bounded(self):
        """Channel runs are unattended; nobody is watching to cancel one."""
        with pytest.raises(ValidationError):
            a_channel(timeout_seconds=99999)


class TestShippedExample:
    def test_the_shipped_config_is_valid_and_inert(self):
        import yaml

        from deep_agent.src.agent.config.paths import config_file

        path = config_file("channels")
        assert path is not None, "channels.yaml should ship as a worked example"

        config = ChannelsFileConfig(**(yaml.safe_load(path.read_text()) or {}))
        assert config.enabled is False
        assert config.active() == {}
