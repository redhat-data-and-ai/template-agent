"""Invoke the local agent or the AI Factory agent and approve tool pauses."""

from __future__ import annotations

import sys
from typing import Any

import httpx

from deep_agent.headless.interrupts import (
    aif_resume_body,
    count_actions,
    find_mcp_auth,
    local_resume_body,
    local_start_body,
)
from deep_agent.headless.jit_access import JITSession
from deep_agent.headless.settings import HeadlessSettings
from deep_agent.headless.stream import (
    end_status,
    interrupts_from_events,
    message_texts,
    parse_sse_lines,
    thread_id_from_events,
)


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _http_error(response: httpx.Response) -> str:
    """Return a short failure string that does not include the bearer token."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    if not isinstance(body, dict):
        return f"HTTP {response.status_code}"
    detail = body.get("message") or body.get("detail") or body.get("error") or ""
    if detail:
        return f"HTTP {response.status_code}: {detail}"
    return f"HTTP {response.status_code}"


def _read_stream(
    client: httpx.Client,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> tuple[str | None, list[tuple[str, Any]]]:
    """POST a streaming run and return ``(header thread id, events)``."""
    with client.stream("POST", url, json=body, headers=headers) as response:
        if response.status_code != 200:
            response.read()
            raise httpx.HTTPStatusError(
                _http_error(response),
                request=response.request,
                response=response,
            )
        lines = list(response.iter_lines())
        header_thread = response.headers.get("X-Thread-ID")
    return header_thread, parse_sse_lines(lines)


def _print_turn(events: list[tuple[str, Any]]) -> None:
    for text in message_texts(events):
        print(text)


class HeadlessInvoker:
    """Call one agent and keep approving HITL until it finishes."""

    def __init__(self, settings: HeadlessSettings, token: str) -> None:
        """Store the call target and the bearer token."""
        self._settings = settings
        self._headers = _headers(token)

    def run(self) -> int:
        """Run the message. Grant Snowflake JIT only for a deployed workflow."""
        if not self._settings.request_jit:
            return self._dispatch()
        session = JITSession(self._settings)
        try:
            if not session.grant():
                return 1
            return self._dispatch()
        finally:
            session.revoke()
            session.close()

    def _dispatch(self) -> int:
        """Call the selected target. Return a process exit code."""
        if self._settings.headless_target == "aif":
            return self._run_aif()
        if self._settings.headless_target == "local":
            return self._run_local()
        print(
            f"Unknown HEADLESS_TARGET {self._settings.headless_target!r}. Use local or aif.",
            file=sys.stderr,
        )
        return 1

    def _run_local(self) -> int:
        base = self._settings.agent_url.rstrip("/")
        timeout = httpx.Timeout(self._settings.agent_timeout)
        try:
            with httpx.Client(
                timeout=timeout, verify=self._settings.headless_ssl_verify
            ) as client:
                created = client.post(f"{base}/threads", json={}, headers=self._headers)
                if created.status_code not in (200, 201):
                    print(
                        f"Create thread failed: {_http_error(created)}", file=sys.stderr
                    )
                    return 1
                thread_id = str(created.json()["thread_id"])
                print(f"Thread {thread_id}")
                url = f"{base}/threads/{thread_id}/runs/stream"
                body: dict[str, Any] = local_start_body(self._settings.agent_message)
                for attempt in range(self._settings.agent_max_resumes + 1):
                    label = "Resume" if attempt else "Invoke"
                    print(f"{label} local {base} ({attempt + 1})")
                    _header, events = _read_stream(client, url, body, self._headers)
                    _print_turn(events)
                    code = self._after_turn(thread_id, events)
                    if code is not None:
                        return code
                    body = local_resume_body(
                        count_actions(interrupts_from_events(events))
                    )
        except httpx.HTTPError as exc:
            print(f"Local invoke failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"Stopped after {self._settings.agent_max_resumes} resumes.",
            file=sys.stderr,
        )
        return 3

    def _run_aif(self) -> int:
        url = self._settings.aif_invoke_url()
        if not url.endswith("/invoke"):
            print(
                "AI Factory target needs the chat URL in AGENT_URL, "
                "or AGENT_BASE_URL, AGENT_ORG, and AGENT_NAME.",
                file=sys.stderr,
            )
            return 1
        timeout = httpx.Timeout(self._settings.agent_timeout)
        body: dict[str, Any] = {"message": self._settings.agent_message}
        try:
            with httpx.Client(
                timeout=timeout, verify=self._settings.headless_ssl_verify
            ) as client:
                thread_id: str | None = None
                for attempt in range(self._settings.agent_max_resumes + 1):
                    label = "Resume" if attempt else "Invoke"
                    print(f"{label} {url} ({attempt + 1})")
                    header_thread, events = _read_stream(
                        client, url, body, self._headers
                    )
                    thread_id = (
                        thread_id_from_events(events, header_thread) or thread_id
                    )
                    if thread_id:
                        print(f"Thread {thread_id}")
                    _print_turn(events)
                    code = self._after_turn(thread_id, events)
                    if code is not None:
                        return code
                    if not thread_id:
                        print("Cannot resume: no thread id.", file=sys.stderr)
                        return 1
                    body = aif_resume_body(
                        thread_id,
                        count_actions(interrupts_from_events(events)),
                    )
        except httpx.HTTPError as exc:
            print(f"AI Factory invoke failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"Stopped after {self._settings.agent_max_resumes} resumes.",
            file=sys.stderr,
        )
        return 3

    def _after_turn(
        self, thread_id: str | None, events: list[tuple[str, Any]]
    ) -> int | None:
        """Return an exit code when the turn is finished, else ``None`` to resume."""
        interrupts = interrupts_from_events(events)
        if end_status(events) == "success" or not interrupts:
            print("Agent completed.")
            return 0
        mcp_auth = find_mcp_auth(interrupts)
        if mcp_auth:
            name = mcp_auth.get("mcp_name") or "mcp"
            print(f"MCP authentication required for {name}.", file=sys.stderr)
            print(
                mcp_auth.get("message") or "Connect that MCP in a browser first.",
                file=sys.stderr,
            )
            return 2
        if not thread_id:
            print("Cannot resume: no thread id.", file=sys.stderr)
            return 1
        print(f"Approving {count_actions(interrupts)} action(s).")
        return None
