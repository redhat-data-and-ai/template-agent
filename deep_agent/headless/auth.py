"""Tokens for a headless call.

Local workflow uses browser SSO (human token). Deployed workflows use the
service-account password grant (CI / runner IPs are Snowflake-allowlisted).
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from deep_agent.headless.settings import HeadlessSettings


def _pkce_pair() -> tuple[str, str]:
    """Return ``(verifier, S256 challenge)``."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


class _Callback:
    """One-shot localhost catcher for the authorization code."""

    def __init__(self, port: int) -> None:
        self._port = port
        self.code: str | None = None
        self.error: str | None = None
        self._running = False
        self._server: HTTPServer | None = None

    def start(self) -> str | None:
        """Bind the callback port before the browser is opened.

        Returns an error message when the port is already taken. The chat UI
        uses this same redirect, so a busy port would show its error page.
        """
        holder = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                """Store the authorization code from the browser redirect."""
                parsed = urlparse(self.path)
                if not parsed.path.startswith("/auth/callback/oidc"):
                    self.send_response(404)
                    self.end_headers()
                    return
                params = parse_qs(parsed.query)
                if "code" in params:
                    holder.code = params["code"][0]
                    body = b"<html><body><h1>Signed in.</h1><p>You can close this tab.</p></body></html>"
                    status = 200
                else:
                    holder.error = params.get("error", ["unknown"])[0]
                    body = b"<html><body><h1>Sign-in failed.</h1></body></html>"
                    status = 400
                self.send_response(status)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body)
                holder._running = False

            def log_message(self, format: str, *args: object) -> None:
                return

        try:
            self._server = HTTPServer(("localhost", self._port), Handler)
        except OSError:
            return f"Port {self._port} is already in use. Stop the chat UI and run this command again."
        self._server.timeout = 1
        self._running = True
        threading.Thread(target=self._serve, daemon=True).start()
        return None

    def wait(self, timeout: int = 180) -> None:
        """Block until the browser returns or ``timeout`` seconds pass."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.code or self.error:
                return
            time.sleep(0.2)

    def stop(self) -> None:
        """Close the callback socket."""
        self._running = False
        if self._server is not None:
            self._server.shutdown()

    def _serve(self) -> None:
        try:
            while self._running and self._server is not None:
                self._server.handle_request()
        finally:
            if self._server is not None:
                self._server.server_close()
                self._server = None


def _token_error(response: httpx.Response) -> str:
    """Return a short token-endpoint error that does not include secrets."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    if not isinstance(body, dict):
        return f"HTTP {response.status_code}"
    error = body.get("error") or ""
    description = body.get("error_description") or ""
    detail = " ".join(part for part in (str(error), str(description)) if part)
    return detail or f"HTTP {response.status_code}"


def exchange_authorization_code(
    settings: HeadlessSettings,
    code: str,
    verifier: str,
) -> str | None:
    """Exchange a browser login code using the environment's client secret."""
    account = settings.service_account()
    if (
        not account["client_id"]
        or not account["client_secret"]
        or not account["token_url"]
    ):
        print(
            "SSO browser login needs SSO_CLIENT_ID, SSO_ISSUER_URL, and "
            "SSO_CLIENT_SECRET or SSO_CLIENT_SECRET_<ENVIRONMENT>.",
            file=sys.stderr,
        )
        return None
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": settings.callback_url(),
        "client_id": account["client_id"],
        "client_secret": account["client_secret"],
        "code_verifier": verifier,
    }
    try:
        response = httpx.post(account["token_url"], data=form, timeout=30.0)
    except httpx.HTTPError as exc:
        print(f"Token exchange failed: {exc}", file=sys.stderr)
        return None
    if response.status_code != 200:
        print(f"Token exchange failed: {_token_error(response)}", file=sys.stderr)
        return None
    token = response.json().get("access_token")
    if not isinstance(token, str) or not token:
        print("Token response had no access_token.", file=sys.stderr)
        return None
    return token


def _token_from_agent_callback(
    settings: HeadlessSettings,
    auth_url: str,
    state: str,
    verifier: str,
) -> str | None:
    """Sign in through the agent when it already owns the callback port.

    The registered redirect is ``http://localhost:8080/auth/callback/oidc``.
    When the agent is the process on that port, it stores the code and this
    command reads it back.
    """
    base = f"http://127.0.0.1:{settings.sso_callback_port}"
    try:
        ready = httpx.post(
            f"{base}/auth/callback/oidc/expect",
            json={"state": state},
            timeout=5.0,
        )
    except httpx.HTTPError:
        ready = None
    if ready is None or ready.status_code != 200:
        print(
            f"Port {settings.sso_callback_port} is already in use, and it does "
            "not accept the headless sign-in. Start this agent on that port.",
            file=sys.stderr,
        )
        return None
    print(f"Waiting for SSO at {settings.callback_url()}")
    print(auth_url)
    webbrowser.open(auth_url)
    deadline = time.time() + 180
    code: str | None = None
    error = "timed out"
    while time.time() < deadline:
        try:
            pending = httpx.get(f"{base}/auth/callback/oidc/pending", timeout=5.0)
        except httpx.HTTPError:
            time.sleep(0.2)
            continue
        if pending.status_code != 200:
            time.sleep(0.2)
            continue
        body = pending.json()
        if isinstance(body, dict) and isinstance(body.get("code"), str):
            code = body["code"]
            break
        if isinstance(body, dict) and body.get("error"):
            error = str(body["error"])
            break
        time.sleep(0.2)
    if not code:
        print(f"SSO did not return a code ({error}).", file=sys.stderr)
        return None
    return exchange_authorization_code(settings, code, verifier)


def browser_token(settings: HeadlessSettings) -> str | None:
    """Open a browser, catch the code, and exchange it for an access token."""
    authorize = settings.authorization_url()
    account = settings.service_account()
    if not account["client_id"] or not authorize or not account["token_url"]:
        print(
            "SSO browser login needs SSO_CLIENT_ID and SSO_ISSUER_URL.",
            file=sys.stderr,
        )
        return None
    if not account["client_secret"]:
        print(
            "SSO browser login needs SSO_CLIENT_SECRET or "
            "SSO_CLIENT_SECRET_<ENVIRONMENT>.",
            file=sys.stderr,
        )
        return None

    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(16)
    query = urlencode(
        {
            "response_type": "code",
            "client_id": account["client_id"],
            "redirect_uri": settings.callback_url(),
            "scope": settings.sso_scope,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    auth_url = f"{authorize}?{query}"
    callback = _Callback(settings.sso_callback_port)
    bind_error = callback.start()
    if bind_error:
        return _token_from_agent_callback(settings, auth_url, state, verifier)
    print(f"Waiting for SSO at {settings.callback_url()}")
    print(auth_url)
    webbrowser.open(auth_url)
    try:
        callback.wait()
    except KeyboardInterrupt:
        callback.stop()
        print("Sign-in cancelled.", file=sys.stderr)
        return None
    finally:
        callback.stop()

    if not callback.code:
        print(
            f"SSO did not return a code ({callback.error or 'timed out'}).",
            file=sys.stderr,
        )
        return None

    return exchange_authorization_code(settings, callback.code, verifier)


def client_credentials_token(settings: HeadlessSettings) -> str | None:
    """Exchange the SSO client id and secret for a service-account token."""
    token_url = settings.token_url()
    if not settings.sso_client_id or not settings.sso_client_secret or not token_url:
        print(
            "Client credentials need SSO_CLIENT_ID, SSO_CLIENT_SECRET, and SSO_ISSUER_URL.",
            file=sys.stderr,
        )
        return None
    form = {
        "grant_type": "client_credentials",
        "client_id": settings.sso_client_id,
        "client_secret": settings.sso_client_secret,
        "scope": settings.sso_scope,
    }
    try:
        response = httpx.post(token_url, data=form, timeout=30.0)
    except httpx.HTTPError as exc:
        print(f"Service account token failed: {exc}", file=sys.stderr)
        return None
    if response.status_code != 200:
        print(
            f"Service account token failed: {_token_error(response)}", file=sys.stderr
        )
        return None
    token = response.json().get("access_token")
    if not isinstance(token, str) or not token:
        print("Token response had no access_token.", file=sys.stderr)
        return None
    return token


def password_grant_token(settings: HeadlessSettings) -> str | None:
    """Exchange the service-account username and password for an access token."""
    account = settings.service_account()
    if not all(
        (
            account["username"],
            account["password"],
            account["client_id"],
            account["client_secret"],
            account["token_url"],
        )
    ):
        print(
            "Password grant needs SSO_SA_USERNAME, SSO_SA_PASSWORD_<ENVIRONMENT>, "
            "SSO_CLIENT_ID, and SSO_CLIENT_SECRET_<ENVIRONMENT>.",
            file=sys.stderr,
        )
        return None
    form = {
        "grant_type": "password",
        "client_id": account["client_id"],
        "client_secret": account["client_secret"],
        "username": account["username"],
        "password": account["password"],
        "scope": settings.sso_scope,
    }
    try:
        response = httpx.post(account["token_url"], data=form, timeout=30.0)
    except httpx.HTTPError as exc:
        print(f"Service account token failed: {exc}", file=sys.stderr)
        return None
    if response.status_code != 200:
        print(
            f"Service account token failed: {_token_error(response)}", file=sys.stderr
        )
        return None
    token = response.json().get("access_token")
    if not isinstance(token, str) or not token:
        print("Token response had no access_token.", file=sys.stderr)
        return None
    return token


def access_token(settings: HeadlessSettings) -> str | None:
    """Return a bearer token for the selected auth mode."""
    if settings.auth_mode == "password":
        return password_grant_token(settings)
    if settings.auth_mode == "client-creds":
        return client_credentials_token(settings)
    if settings.auth_mode == "sso":
        return browser_token(settings)
    print(
        f"Unknown AUTH_MODE {settings.auth_mode!r}. Use sso, password, or client-creds.",
        file=sys.stderr,
    )
    return None
