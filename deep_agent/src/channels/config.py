"""Channel configuration, loaded from `config/agent/integrations/channels.yaml`.

Adding a channel is a config edit, like everything else in these
templates. The models here are what `make validate-config` checks, so a
missing signing secret is a startup error rather than a 500 on the first
Slack event.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class IdentityConfig(BaseModel):
    """How a channel sender becomes a platform user.

    A channel carries no SSO token, so something has to decide whose
    data a run may touch. An explicit map is the only answer that keeps
    ownership and authz meaningful — "whoever posted in Slack" is not an
    identity the rest of the system can reason about.
    """

    # Channel-native sender id (Slack user ID, webhook `sender` field)
    # -> platform user id.
    map: dict[str, str] = Field(default_factory=dict)

    # Used when a sender is not in the map. Unset means unmapped senders
    # are rejected, which is the safe default: a shared fallback identity
    # makes every run look like the same person in the audit trail.
    default_user: str | None = None

    def resolve(self, sender: str) -> str | None:
        """Platform user for this sender, or None if they are not allowed."""
        return self.map.get(sender) or self.default_user


class ChannelConfig(BaseModel):
    """One inbound channel."""

    type: Literal["slack", "webhook"]
    enabled: bool = True

    # Name of the environment variable holding the signing secret — not
    # the secret itself. Secrets reach the pod through a Secret, and a
    # config file that can hold one will eventually hold one.
    signing_secret_env: str

    # Reject anything older than this. Slack recommends 5 minutes; the
    # window is what stops a captured request being replayed forever.
    max_age_seconds: int = Field(default=300, ge=1, le=3600)

    identity: IdentityConfig = Field(default_factory=IdentityConfig)

    # Assistant to run. Defaults to the main graph.
    assistant_id: str = "agent"

    # Hard ceiling on a channel-initiated run. Channels are unattended,
    # so nobody is watching to cancel one that goes wrong.
    timeout_seconds: int = Field(default=300, ge=5, le=3600)

    @property
    def signing_secret(self) -> str | None:
        """The secret, read at use time so rotation does not need a restart."""
        return os.environ.get(self.signing_secret_env) or None


class ChannelsFileConfig(BaseModel):
    """The whole channels.yaml."""

    # Off unless a channel is configured. An inbound surface that appears
    # by default is an inbound surface nobody decided to expose.
    enabled: bool = False
    channels: dict[str, ChannelConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_enabled_channels_are_usable(self) -> ChannelsFileConfig:
        if not self.enabled:
            return self
        active = [name for name, c in self.channels.items() if c.enabled]
        if not active:
            raise ValueError(
                "channels.enabled is true but no channel is enabled; "
                "either enable one or set channels.enabled to false"
            )
        return self

    def active(self) -> dict[str, ChannelConfig]:
        """Enabled channels, empty when the feature is off."""
        if not self.enabled:
            return {}
        return {name: c for name, c in self.channels.items() if c.enabled}
