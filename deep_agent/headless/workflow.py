"""Choose one headless workflow and the behavior that belongs to it.

``local``
    The caller signs in with their own SSO. The question goes to
    ``http://127.0.0.1:8080/invoke``. Snowflake access is the caller's, so
    no JIT grant is requested.

``preprod`` and ``prod``
    The caller signs in as the service account for that environment. JIT
    grants Snowflake access, the question goes to the deployed agent, and
    JIT revokes that access afterward.
"""

from __future__ import annotations

from typing import cast
from urllib.parse import urlparse

from deep_agent.headless.settings import HeadlessSettings

LOCAL = "local"
DEPLOYED = ("preprod", "prod")
WORKFLOWS = (LOCAL, *DEPLOYED)
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


class WorkflowError(ValueError):
    """Raised when a workflow cannot be applied."""


def apply_workflow(settings: HeadlessSettings, workflow: str) -> HeadlessSettings:
    """Return settings for ``local``, ``preprod``, or ``prod``."""
    name = workflow.strip().lower()
    if name == LOCAL:
        return cast(
            HeadlessSettings,
            settings.model_copy(
                update={
                    "workflow": LOCAL,
                    "auth_mode": "sso",
                    "headless_target": "aif",
                    "request_jit": False,
                }
            ),
        )
    if name not in DEPLOYED:
        raise WorkflowError(
            f"Unknown workflow {workflow!r}. Use local, preprod, or prod."
        )
    if not _has_deployed_destination(settings):
        raise WorkflowError(
            "The preprod and prod workflows call the deployed agent. "
            "Set AGENT_BASE_URL, AGENT_ORG, and AGENT_NAME, "
            "or set AGENT_URL to that agent's URL."
        )
    target = "aif" if _uses_invoke_url(settings) else "local"
    return cast(
        HeadlessSettings,
        settings.model_copy(
            update={
                "workflow": name,
                "environment": name,
                "auth_mode": "password",
                "headless_target": target,
                "request_jit": True,
            }
        ),
    )


def _aif_configured(settings: HeadlessSettings) -> bool:
    """Return True when org and name can build an engine invoke URL."""
    return bool(
        settings.agent_base_url.strip()
        and settings.agent_org.strip()
        and settings.agent_name.strip()
    )


def _uses_invoke_url(settings: HeadlessSettings) -> bool:
    """Return True when the call posts to a URL ending in ``/invoke``."""
    if _remote_agent_url(settings):
        return True
    return _aif_configured(settings)


def _has_deployed_destination(settings: HeadlessSettings) -> bool:
    """Return True when the call has a deployed agent to reach."""
    return _uses_invoke_url(settings)


def _remote_agent_url(settings: HeadlessSettings) -> bool:
    """Return True when ``AGENT_URL`` is the chat URL from AI Factory."""
    host = (urlparse(settings.agent_url).hostname or "").lower()
    return bool(host) and host not in _LOOPBACK
