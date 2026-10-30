"""Headless HTTP API on every deployment of this agent.

``POST /invoke`` is part of the agent service (local *and* deployed). Any
caller with a bearer token can run the agent without the chat UI:

* Body: ``{"message": "..."}`` (first turn) or a resume payload.
* Streams SSE from ``/runs/stream`` and **auto-approves** tool HITL pauses
  (chat still requires a human). MCP login interrupts are not auto-approved.

The optional ``python -m deep_agent.headless`` CLI is only a client of this
same route. Deployed agents are reachable at ``{AGENT_URL}/invoke``.

Local SSO helpers (``/auth/callback/oidc*``) support browser login when the
CLI shares this process's callback port.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from deep_agent.headless.interrupts import (
    count_actions,
    find_mcp_auth,
    local_resume_body,
)
from deep_agent.headless.stream import (
    end_status,
    interrupts_from_events,
    parse_sse_lines,
)

# DVA Core chains many HITL pauses (skill reads, task, subagent tools, SQL).
_MAX_APPROVALS = 40
_HEARTBEAT = b": heartbeat"
_THINKING = b": thinking"

router = APIRouter(tags=["invoke"])


async def _open_run(
    client: httpx.AsyncClient,
    base: str,
    thread_id: str,
    run: dict[str, Any],
    headers: dict[str, str],
) -> httpx.Response:
    """POST one ``/runs/stream`` call and return the streaming response."""
    sent = client.build_request(
        "POST",
        f"{base}/threads/{thread_id}/runs/stream",
        json=run,
        headers=headers,
    )
    return await client.send(sent, stream=True)


_lock = threading.Lock()
_expected_state: str | None = None
_code: str | None = None
_error: str | None = None

_SIGNED_IN = (
    "<html><body><h1>Signed in.</h1><p>You can close this tab.</p></body></html>"
)
_FAILED = "<html><body><h1>Sign-in failed.</h1></body></html>"


def run_body(body: dict[str, Any]) -> dict[str, Any] | None:
    """Turn a headless payload into a ``/runs/stream`` body."""
    if "command" in body or "input" in body:
        run: dict[str, Any] = {
            "assistant_id": body.get("assistant_id") or "agent",
            "stream_mode": body.get("stream_mode") or "updates",
        }
        if "input" in body:
            run["input"] = body["input"]
        if "command" in body:
            run["command"] = body["command"]
        return run
    message = body.get("message")
    if isinstance(message, str) and message.strip():
        return {
            "assistant_id": "agent",
            "input": {"messages": [{"role": "human", "content": message}]},
            "stream_mode": "updates",
        }
    return None


def approval_resume(events: list[tuple[str, Any]]) -> dict[str, Any] | None:
    """Return an approve body when ``/invoke`` is waiting on a tool.

    A finished run and an MCP login interrupt are left alone. Chat does not
    use this path.
    """
    interrupts = interrupts_from_events(events)
    if not interrupts or find_mcp_auth(interrupts):
        return None
    if end_status(events) == "success":
        return None
    count = count_actions(interrupts)
    if count < 1:
        return None
    return local_resume_body(count)


def rewrite_keepalive(
    pending: bytearray, chunk: bytes, *, final: bool = False
) -> bytes:
    """Replace the stream keep-alive comment with ``thinking``."""
    pending.extend(chunk)
    out = bytearray()
    while True:
        index = pending.find(_HEARTBEAT)
        if index == -1:
            break
        out.extend(pending[:index])
        out.extend(_THINKING)
        del pending[: index + len(_HEARTBEAT)]
    if final:
        out.extend(pending)
        pending.clear()
        return bytes(out)
    keep = len(_HEARTBEAT) - 1
    if len(pending) > keep:
        out.extend(pending[:-keep])
        del pending[:-keep]
    return bytes(out)


def _forward_headers(request: Request) -> dict[str, str]:
    """Copy the caller token onto the in-process run request."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    authorization = request.headers.get("authorization")
    if authorization:
        headers["Authorization"] = authorization
    return headers


@router.post("/auth/callback/oidc/expect")
async def expect_sso(request: Request) -> JSONResponse:
    """Remember the login state the headless command is waiting for."""
    global _expected_state, _code, _error
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"detail": "JSON object required"}, status_code=400)
    state = body.get("state") if isinstance(body, dict) else None
    if not isinstance(state, str) or not state:
        return JSONResponse({"detail": "state required"}, status_code=400)
    with _lock:
        _expected_state = state
        _code = None
        _error = None
    return JSONResponse({"ok": True})


@router.get("/auth/callback/oidc")
async def sso_callback(request: Request) -> HTMLResponse:
    """Store the authorization code from the browser redirect."""
    global _code, _error
    state = request.query_params.get("state")
    with _lock:
        expected = _expected_state
        if not expected or state != expected:
            return HTMLResponse(_FAILED, status_code=400)
        code = request.query_params.get("code")
        if code:
            _code = code
            _error = None
            return HTMLResponse(_SIGNED_IN)
        _error = request.query_params.get("error") or "unknown"
        return HTMLResponse(_FAILED, status_code=400)


@router.get("/auth/callback/oidc/pending")
async def sso_pending() -> dict[str, str | None]:
    """Return the code once, then forget it."""
    global _code, _error
    with _lock:
        code = _code
        error = _error
        if code or error:
            _code = None
            _error = None
    return {"code": code, "error": error}


@router.post("/invoke", response_model=None)
async def invoke(request: Request) -> JSONResponse | StreamingResponse:
    """Headless entrypoint for local and deployed agents.

    Forward the caller's ``Authorization`` header, create a thread if needed,
    stream the run, and auto-approve tool pauses until the turn finishes.
    """
    try:
        payload = await request.json()
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        return JSONResponse({"detail": "JSON object required"}, status_code=400)
    run = run_body(payload)
    if run is None:
        return JSONResponse(
            {"detail": "message or resume body required"}, status_code=400
        )

    base = str(request.base_url).rstrip("/")
    headers = _forward_headers(request)
    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
    thread_id = payload.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        try:
            created = await client.post(f"{base}/threads", json={}, headers=headers)
        except httpx.HTTPError:
            await client.aclose()
            return JSONResponse({"detail": "Agent is unreachable"}, status_code=502)
        if created.status_code not in (200, 201):
            await client.aclose()
            return JSONResponse(
                {"detail": "Could not create a thread"},
                status_code=created.status_code,
            )
        created_body = created.json()
        thread_id = str(created_body.get("thread_id") or "")
        if not thread_id:
            await client.aclose()
            return JSONResponse({"detail": "Thread was not created"}, status_code=502)

    try:
        response = await _open_run(client, base, thread_id, run, headers)
    except httpx.HTTPError:
        await client.aclose()
        return JSONResponse({"detail": "Agent run failed"}, status_code=502)

    async def chunks() -> AsyncIterator[bytes]:
        """Stream the run, approving each tool pause on this headless URL."""
        current = response
        try:
            for attempt in range(_MAX_APPROVALS + 1):
                text_parts: list[str] = []
                pending = bytearray()
                try:
                    async for chunk in current.aiter_bytes():
                        text_parts.append(chunk.decode("utf-8", errors="replace"))
                        rewritten = rewrite_keepalive(pending, chunk)
                        if rewritten:
                            yield rewritten
                    tail = rewrite_keepalive(pending, b"", final=True)
                    if tail:
                        yield tail
                finally:
                    await current.aclose()
                if current.status_code not in (200, 201) or attempt >= _MAX_APPROVALS:
                    return
                events = parse_sse_lines("".join(text_parts).splitlines())
                resume = approval_resume(events)
                if resume is None:
                    return
                current = await _open_run(client, base, thread_id, resume, headers)
        except httpx.HTTPError:
            yield b'event: error\ndata: {"detail":"Agent run failed"}\n\n'
        finally:
            await client.aclose()

    media = response.headers.get("content-type", "text/event-stream")
    return StreamingResponse(
        chunks(),
        status_code=response.status_code,
        media_type=media.split(";")[0],
        headers={"X-Thread-ID": thread_id},
    )
