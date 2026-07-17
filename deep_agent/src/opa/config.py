"""OPA authorization configuration.

Resolution order (highest wins):
  1. Environment variables: OPA_ENABLED, OPA_URL, OPA_TIMEOUT
  2. agent.yaml ``opa:`` section
  3. OpaFileConfig defaults (enabled: false)

YAML reference (agent.yaml):
    opa.enabled
    opa.url
    opa.timeout
"""

from __future__ import annotations

from deep_agent.src.settings import settings


def _yaml_config():
    from deep_agent.src.agent.config import agent_config
    return agent_config.get_opa_config()


def is_opa_enabled() -> bool:
    """Return whether OPA authorization is enabled.

    Env var OPA_ENABLED overrides agent.yaml when explicitly set.
    """
    if settings.OPA_ENABLED is not None:
        return settings.OPA_ENABLED
    return _yaml_config().enabled


def get_opa_url() -> str:
    """Return the OPA decision endpoint URL."""
    if settings.OPA_URL is not None:
        return settings.OPA_URL
    return _yaml_config().url


def get_opa_timeout() -> float:
    """Return the OPA request timeout in seconds."""
    if settings.OPA_TIMEOUT is not None:
        return settings.OPA_TIMEOUT
    return _yaml_config().timeout
