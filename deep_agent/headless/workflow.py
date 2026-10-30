"""Choose one headless workflow and the behavior that belongs to it.

``local``
    Browser SSO as the human user. Posts to this agent's ``/invoke`` route
    (``AGENT_URL`` loopback or ``http://127.0.0.1:<AGENT_PORT>/invoke``).
    No AI Factory registry.

``preprod`` and ``prod``
    Service-account password grant. Posts to a remote agent ``/invoke``:

    * **Direct (no AIF registry)** — set ``AGENT_URL`` to that agent's base
      or ``…/invoke`` URL (template consumers should use this).
    * **Optional AIF engine** — set ``AGENT_BASE_URL``, ``AGENT_ORG``, and
      ``AGENT_NAME`` when you intentionally want the registry path.

Tools are auto-approved on every path.
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


def _environment_for_local(settings: HeadlessSettings) -> str:
    """Pick the OIDC client secret env that matches the local agent's issuer.

    A stage token cannot be verified by a prod JWKS (and the reverse), so
    local headless must mint against the same realm the agent loads.
    """
    issuer = settings.sso_issuer_url.strip().lower()
    if "auth.stage.redhat.com" in issuer:
        return "preprod"
    if "auth.redhat.com" in issuer:
        return "prod"
    env = settings.environment.strip().lower()
    return env if env in {"dev", "preprod", "prod"} else "preprod"


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
                    "headless_target": "invoke",
                    "environment": _environment_for_local(settings),
                }
            ),
        )
    if name not in DEPLOYED:
        raise WorkflowError(
            f"Unknown workflow {workflow!r}. Use local, preprod, or prod."
        )
    if not _has_deployed_destination(settings):
        raise WorkflowError(
            "The preprod and prod workflows need a remote agent. "
            "Set AGENT_URL to that agent's URL (direct, no AIF registry), "
            "or set AGENT_BASE_URL, AGENT_ORG, and AGENT_NAME for the "
            "optional AI Factory engine path."
        )
    return cast(
        HeadlessSettings,
        settings.model_copy(
            update={
                "workflow": name,
                "environment": name,
                "auth_mode": "password",
                "headless_target": "invoke",
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


def _remote_agent_url(settings: HeadlessSettings) -> bool:
    """Return True when ``AGENT_URL`` points at a non-loopback host."""
    host = (urlparse(settings.agent_url).hostname or "").lower()
    return bool(host) and host not in _LOOPBACK


def _has_deployed_destination(settings: HeadlessSettings) -> bool:
    """Return True when a remote direct URL or AIF engine path is configured."""
    return _remote_agent_url(settings) or _aif_configured(settings)
