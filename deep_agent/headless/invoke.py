"""Invoke an agent ``/invoke`` (or threads API) and approve tool pauses."""

from __future__ import annotations

import sys
from typing import Any

import httpx

from deep_agent.headless.interrupts import (
    count_actions,
    find_mcp_auth,
    invoke_resume_body,
    local_resume_body,
    local_start_body,
)
from deep_agent.headless.settings import HeadlessSettings
from deep_agent.headless.stream import (
    end_status,
    interrupts_from_events,
    iter_sse_events,
    message_texts,
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
    """POST a streaming run, print assistant text as it arrives, return events."""
    with client.stream("POST", url, json=body, headers=headers) as response:
        if response.status_code != 200:
            response.read()
            raise httpx.HTTPStatusError(
                _http_error(response),
                request=response.request,
                response=response,
            )
        header_thread = response.headers.get("X-Thread-ID")
        if header_thread:
            print(f"Thread {header_thread}", flush=True)
        events: list[tuple[str, Any]] = []
        for event in iter_sse_events(response.iter_lines()):
            events.append(event)
            for text in message_texts([event]):
                print(text, flush=True)
            if event[0] == "end" and isinstance(event[1], dict):
                status = event[1].get("status")
                if status == "interrupted":
                    print("… tool pause (auto-approving)", flush=True)
                elif status and status != "success":
                    print(f"… status={status}", flush=True)
    return header_thread, events


class HeadlessInvoker:
    """Call one agent and keep approving HITL until it finishes."""

    def __init__(self, settings: HeadlessSettings, token: str) -> None:
        """Store the call target and the bearer token."""
        self._settings = settings
        self._headers = _headers(token)

    def run(self) -> int:
        """Run the message against the selected target."""
        target = self._settings.headless_target
        # ``aif`` is a legacy alias for the ``/invoke`` request shape.
        if target in {"invoke", "aif"}:
            return self._run_invoke()
        if target in {"threads", "local"}:
            return self._run_threads()
        print(
            f"Unknown HEADLESS_TARGET {target!r}. Use invoke or threads.",
            file=sys.stderr,
        )
        return 1

    def _run_threads(self) -> int:
        """Call the LangGraph threads + ``/runs/stream`` API."""
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
                    print(f"{label} threads {base} ({attempt + 1})", flush=True)
                    _header, events = _read_stream(client, url, body, self._headers)
                    code = self._after_turn(thread_id, events)
                    if code is not None:
                        return code
                    body = local_resume_body(
                        count_actions(interrupts_from_events(events))
                    )
        except httpx.HTTPError as exc:
            print(f"Threads invoke failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"Stopped after {self._settings.agent_max_resumes} resumes.",
            file=sys.stderr,
        )
        return 3

    def _run_invoke(self) -> int:
        """POST the agent's ``/invoke`` route (direct or optional AIF engine URL)."""
        url = self._settings.invoke_url()
        if not url.endswith("/invoke"):
            print(
                "Invoke target needs AGENT_URL (direct agent /invoke), "
                "or AGENT_BASE_URL + AGENT_ORG + AGENT_NAME (optional AIF).",
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
                    print(f"{label} {url} ({attempt + 1})", flush=True)
                    header_thread, events = _read_stream(
                        client, url, body, self._headers
                    )
                    thread_id = (
                        thread_id_from_events(events, header_thread) or thread_id
                    )
                    if thread_id and not header_thread:
                        print(f"Thread {thread_id}", flush=True)
                    code = self._after_turn(thread_id, events)
                    if code is not None:
                        return code
                    if not thread_id:
                        print("Cannot resume: no thread id.", file=sys.stderr)
                        return 1
                    body = invoke_resume_body(
                        thread_id,
                        count_actions(interrupts_from_events(events)),
                    )
        except httpx.HTTPError as exc:
            print(f"Invoke failed: {exc}", file=sys.stderr)
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
        status = end_status(events)
        interrupts = interrupts_from_events(events)
        if status == "error":
            print("Agent run failed (status=error).", file=sys.stderr)
            return 1
        if status == "success" or not interrupts:
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
        print(f"Approving {count_actions(interrupts)} action(s).", flush=True)
        return None
