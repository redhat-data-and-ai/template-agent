"""Record/replay for LLM and other HTTP traffic.

Agent tests that touch a model are otherwise flaky, slow and billed. The
existing `mock_llm` fixture avoids that by replacing the model object,
but then nothing exercises the real path — prompt assembly, provider
payload shape, response parsing and error handling all go untested.

This intercepts one layer lower, at the `httpx` transport every provider
SDK uses, so the code under test is the real code and only the network
is fake.

Two modes:

* **replay** (default) — serve recorded responses. An unrecorded request
  is an error, never a silent fall-through to the network. A cassette
  that quietly lets calls escape is worse than no cassette.
* **record** — `CASSETTE_MODE=record`, make the real call and save it.
  Credentials are stripped before anything is written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx

CASSETTE_DIR = Path(__file__).resolve().parent.parent / "cassettes"

# Headers never written to disk. A recorded Authorization header is a
# committed credential.
_REDACTED_HEADERS = {
    "authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "cookie",
    "set-cookie",
    "x-refresh-token",
    "x-token",
}

# Body fields that commonly carry secrets in provider payloads.
_SECRET_BODY_KEYS = re.compile(
    r"(api_?key|client_secret|refresh_token|access_token|password)",
    re.IGNORECASE,
)

_PLACEHOLDER = "[REDACTED]"


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: (_PLACEHOLDER if k.lower() in _REDACTED_HEADERS else v)
        for k, v in headers.items()
    }


def _redact_body(raw: bytes) -> str:
    """Redact secret-shaped fields, preserving shape for matching."""
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text

    def scrub(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                k: (_PLACEHOLDER if _SECRET_BODY_KEYS.search(k) else scrub(v))
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [scrub(v) for v in node]
        return node

    return json.dumps(scrub(parsed), sort_keys=True)


def request_key(method: str, url: str, body: bytes) -> str:
    """Stable identity for a request.

    Includes the body so two different prompts to the same endpoint do
    not collide, but hashes it so cassettes stay readable.
    """
    digest = hashlib.sha256(body or b"").hexdigest()[:16]
    return f"{method.upper()} {url} {digest}"


class CassetteMissError(AssertionError):
    """Raised when replay has no recording for a request."""


class Cassette:
    """One test's recorded interactions."""

    def __init__(self, path: Path, mode: str) -> None:
        self.path = path
        self.mode = mode
        self._entries: dict[str, list[dict[str, Any]]] = {}
        self._cursor: dict[str, int] = {}
        if path.is_file():
            self._entries = json.loads(path.read_text())

    @property
    def recorded_count(self) -> int:
        return sum(len(v) for v in self._entries.values())

    def next_response(self, key: str) -> dict[str, Any] | None:
        """Pop the next recorded response for a key, in call order.

        Repeated identical requests replay in sequence, so a retry gets
        the retried response rather than the first one again.
        """
        bucket = self._entries.get(key)
        if not bucket:
            return None
        index = self._cursor.get(key, 0)
        self._cursor[key] = index + 1
        return bucket[min(index, len(bucket) - 1)]

    def record(self, key: str, response: dict[str, Any]) -> None:
        self._entries.setdefault(key, []).append(response)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, indent=2, sort_keys=True))


class CassetteTransport(httpx.AsyncBaseTransport):
    """httpx transport backed by a cassette."""

    def __init__(self, cassette: Cassette) -> None:
        self._cassette = cassette
        self._inner: httpx.AsyncBaseTransport | None = (
            httpx.AsyncHTTPTransport() if cassette.mode == "record" else None
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = request.content or b""
        key = request_key(request.method, str(request.url), body)

        if self._cassette.mode == "replay":
            entry = self._cassette.next_response(key)
            if entry is None:
                raise CassetteMissError(
                    f"No recording for {key}\n"
                    f"  cassette: {self._cassette.path}\n"
                    f"  re-record with: CASSETTE_MODE=record pytest <test>\n"
                    f"  (replay never falls through to the network)"
                )
            # `content-encoding` is dropped on the way out. The body
            # recorded is `response.text`, already decoded, so
            # replaying it under a `content-encoding: gzip` header
            # makes httpx try to gunzip plain text and fail.
            headers = {
                k: v
                for k, v in (entry.get("headers") or {}).items()
                if k.lower() not in ("content-encoding", "content-length")
            }
            return httpx.Response(
                status_code=entry["status"],
                headers=headers,
                content=entry["body"].encode(),
                request=request,
            )

        assert self._inner is not None
        response = await self._inner.handle_async_request(request)
        await response.aread()
        self._cassette.record(
            key,
            {
                "status": response.status_code,
                "headers": _redact_headers(dict(response.headers)),
                "body": response.text,
                "request_headers": _redact_headers(dict(request.headers)),
                "request_body": _redact_body(body),
            },
        )
        return response


def cassette_mode() -> str:
    """`record` only when explicitly asked; `replay` otherwise."""
    return "record" if os.getenv("CASSETTE_MODE") == "record" else "replay"
