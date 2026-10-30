"""Environment settings for the headless runner."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


def _unwrap(value: str) -> str:
    """Strip wrapping quotes from an env value."""
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


class HeadlessSettings(BaseSettings):
    """Load headless options from the environment and ``.env``.

    The CLI selects a workflow, which sets the fields below:

    * ``local`` — browser SSO and this agent's local ``/invoke`` (no AIF).
    * ``preprod`` / ``prod`` — service-account password grant against a
      remote agent ``/invoke``. Prefer ``AGENT_URL`` (direct). Optionally
      set ``AGENT_BASE_URL`` + ``AGENT_ORG`` + ``AGENT_NAME`` for the AI
      Factory engine path. Tools are auto-approved on every path.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    workflow: str = "local"
    auth_mode: str = "sso"
    headless_target: str = "local"
    environment: str = "preprod"

    sso_client_id: str = ""
    sso_client_secret: str = ""
    sso_client_secret_dev: str = ""
    sso_client_secret_preprod: str = ""
    sso_client_secret_prod: str = ""
    sso_sa_username: str = ""
    sso_sa_password: str = ""
    sso_sa_password_dev: str = ""
    sso_sa_password_preprod: str = ""
    sso_sa_password_prod: str = ""
    sso_token_url_dev: str = ""
    sso_token_url_preprod: str = ""
    sso_token_url_prod: str = ""
    sso_callback_port: int = 8765
    sso_authorization_url: str = ""
    sso_token_url: str = ""
    sso_issuer_url: str = ""
    sso_scope: str = "openid profile email"

    agent_url: str = ""
    agent_port: int = 5002
    agent_base_url: str = ""
    agent_org: str = ""
    agent_name: str = ""
    agent_message: str = "What can you do?"
    agent_max_resumes: int = 40
    agent_timeout: int = 300
    headless_ssl_verify: bool = True

    def service_account(self) -> dict[str, str]:
        """Return the password-grant account selected by ``ENVIRONMENT``."""
        env = self.environment.strip().lower()
        passwords = {
            "dev": self.sso_sa_password_dev,
            "preprod": self.sso_sa_password_preprod,
            "prod": self.sso_sa_password_prod,
        }
        secrets = {
            "dev": self.sso_client_secret_dev,
            "preprod": self.sso_client_secret_preprod,
            "prod": self.sso_client_secret_prod,
        }
        token_urls = {
            "dev": self.sso_token_url_dev,
            "preprod": self.sso_token_url_preprod,
            "prod": self.sso_token_url_prod,
        }
        token_url = _unwrap(token_urls.get(env, "")) or self.token_url()
        return {
            "username": _unwrap(self.sso_sa_username),
            "password": _unwrap(passwords.get(env, ""))
            or _unwrap(self.sso_sa_password),
            "client_id": _unwrap(self.sso_client_id),
            "client_secret": _unwrap(secrets.get(env, ""))
            or _unwrap(self.sso_client_secret),
            "token_url": token_url,
        }

    def authorization_url(self) -> str:
        """Return the OIDC authorization endpoint."""
        if self.sso_authorization_url.strip():
            return self.sso_authorization_url.strip()
        issuer = self.sso_issuer_url.rstrip("/")
        if not issuer:
            return ""
        return f"{issuer}/protocol/openid-connect/auth"

    def token_url(self) -> str:
        """Return the OIDC token endpoint."""
        if self.sso_token_url.strip():
            return self.sso_token_url.strip()
        issuer = self.sso_issuer_url.rstrip("/")
        if not issuer:
            return ""
        return f"{issuer}/protocol/openid-connect/token"

    def callback_url(self) -> str:
        """Return the local redirect URI used by the browser login."""
        return f"http://localhost:{self.sso_callback_port}/auth/callback/oidc"

    def invoke_url(self) -> str:
        """Return the ``/invoke`` URL a headless run posts to.

        Resolution order:

        1. ``AGENT_URL`` — direct agent URL (``…/invoke`` appended if needed).
           Works for local loopback and for any deployed agent that exposes
           ``/invoke``. No AI Factory registry.
        2. Local workflow without ``AGENT_URL`` —
           ``http://127.0.0.1:<AGENT_PORT>/invoke``.
        3. Optional AIF engine path —
           ``{AGENT_BASE_URL}/api/engine/v1/agents/{ORG}/{NAME}/invoke``.
        """
        direct = self.agent_url.strip().rstrip("/")
        if direct:
            if direct.endswith("/invoke"):
                return direct
            return f"{direct}/invoke"
        if self.workflow == "local":
            return f"http://127.0.0.1:{self.agent_port}/invoke"
        base = self.agent_base_url.rstrip("/")
        return f"{base}/api/engine/v1/agents/{self.agent_org}/{self.agent_name}/invoke"

    def aif_invoke_url(self) -> str:
        """Alias for :meth:`invoke_url` (kept for older callers)."""
        return self.invoke_url()
