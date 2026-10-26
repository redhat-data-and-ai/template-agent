"""Grant and revoke temporary Snowflake access before a headless run.

Same DICE pipeline the Dataverse test suite uses. ``DICE_JIT_TRIGGER_TOKEN``
is a GitLab trigger token. ``ENVIRONMENT`` selects ``JIT_DVA_ENV``.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Any

import httpx

from deep_agent.headless.settings import HeadlessSettings, _unwrap

GITLAB_API = "https://gitlab.cee.redhat.com/api/v4"
_ENV_MAP = {
    "preprod": "preprod",
    "prod": "prod",
    "sandbox": "sandbox",
    "dev": "dev",
}
_POLL_INTERVAL = 30
_POLL_TIMEOUT = 360
_MAX_GRANT_ATTEMPTS = 2
_MAX_REVOKE_ATTEMPTS = 2
_TERMINAL = {"success", "failed", "canceled", "skipped"}


class JITAccessError(RuntimeError):
    """Raised when a JIT grant or revoke does not succeed."""


def _jit_env(environment: str) -> str:
    """Map ``ENVIRONMENT`` to the DICE ``JIT_DVA_ENV`` value."""
    env = environment.strip().lower() or "preprod"
    mapped = _ENV_MAP.get(env)
    if not mapped:
        raise JITAccessError(f"No DICE JIT_DVA_ENV mapping for ENVIRONMENT={env!r}")
    return mapped


def _safe_text(text: str, secret: str) -> str:
    """Return a short response excerpt that does not contain the trigger token."""
    cleaned = text.replace(secret, "[redacted]") if secret else text
    return cleaned[:300]


class JITSession:
    """Grant Snowflake access for one headless run and revoke it afterward."""

    def __init__(
        self,
        settings: HeadlessSettings,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        poll_interval: float = _POLL_INTERVAL,
        poll_timeout: float = _POLL_TIMEOUT,
    ) -> None:
        """Store the settings and the HTTP client used for GitLab."""
        self._settings = settings
        self._token = _unwrap(settings.dice_jit_trigger_token)
        self._project_id = _unwrap(settings.dice_project_id) or "106670"
        self._client = client
        self._owns_client = client is None
        self._sleep = sleep
        self._monotonic = monotonic
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._attempted = False

    def grant(self) -> bool:
        """Request JIT access. Return True when the agent run may start."""
        if not self._token:
            print(
                "DICE_JIT_TRIGGER_TOKEN is unset. Snowflake JIT access was not requested."
            )
            return True
        self._attempted = True
        try:
            self._run("grant", _MAX_GRANT_ATTEMPTS)
        except JITAccessError as exc:
            print(f"JIT grant failed: {exc}", file=sys.stderr)
            return False
        print("JIT Snowflake access granted.")
        return True

    def revoke(self) -> None:
        """Revoke access after a grant attempt. Do not raise."""
        if not self._attempted:
            return
        try:
            self._run("revoke", _MAX_REVOKE_ATTEMPTS)
        except JITAccessError as exc:
            print(
                "JIT revoke failed. The service account may still have "
                f"Snowflake access. Last error: {exc}",
                file=sys.stderr,
            )
            return
        print("JIT Snowflake access revoked.")

    def close(self) -> None:
        """Close the GitLab client when this session created it."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _run(self, action: str, attempts: int) -> None:
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            print(f"JIT {action} attempt {attempt}/{attempts}")
            try:
                pipeline_id = self._trigger(action)
                status = self._poll(pipeline_id)
            except (JITAccessError, httpx.HTTPError) as exc:
                last_error = exc
                print(f"JIT {action} attempt {attempt} failed: {exc}", file=sys.stderr)
                continue
            if status == "success":
                return
            last_error = JITAccessError(
                f"JIT {action} pipeline #{pipeline_id} ended with status {status!r}"
            )
            print(str(last_error), file=sys.stderr)
        raise JITAccessError(
            f"JIT {action} failed after {attempts} attempts: {last_error}"
        )

    def _trigger(self, action: str) -> int:
        jit_env = _jit_env(self._settings.environment)
        response = self._http().post(
            f"{GITLAB_API}/projects/{self._project_id}/trigger/pipeline",
            data={
                "token": self._token,
                "ref": "main",
                "variables[JIT_DVA_ACTION]": action,
                "variables[JIT_DVA_ENV]": jit_env,
            },
        )
        if response.status_code != 201:
            detail = _safe_text(response.text, self._token)
            raise JITAccessError(
                f"DICE trigger ({action}) returned HTTP {response.status_code}: {detail}"
            )
        payload: Any = response.json()
        pipeline_id = payload.get("id") if isinstance(payload, dict) else None
        if not pipeline_id:
            raise JITAccessError(
                f"DICE trigger ({action}) response missing pipeline id"
            )
        print(f"DICE JIT {action} pipeline #{pipeline_id} for {jit_env}")
        return int(pipeline_id)

    def _poll(self, pipeline_id: int) -> str:
        deadline = self._monotonic() + self._poll_timeout
        last_status = "unknown"
        while self._monotonic() < deadline:
            response = self._http().get(
                f"{GITLAB_API}/projects/{self._project_id}/pipelines/{pipeline_id}"
            )
            if response.status_code == 200:
                payload: Any = response.json()
                if isinstance(payload, dict):
                    last_status = str(payload.get("status") or "unknown")
                if last_status in _TERMINAL:
                    return last_status
            self._sleep(self._poll_interval)
        raise JITAccessError(
            f"DICE pipeline #{pipeline_id} did not finish within "
            f"{int(self._poll_timeout)}s (last status: {last_status})"
        )

    def _http(self) -> httpx.Client:
        if self._client is None:
            # GitLab's internal certificate is not in the default trust store.
            self._client = httpx.Client(timeout=30.0, verify=False)  # nosec B501
        return self._client
