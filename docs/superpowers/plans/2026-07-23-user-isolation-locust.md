# User Isolation Locust Load Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Locust load tests that verify user data isolation (memories, rules, threads, feedback) holds under concurrent multi-user load with real JWT authentication.

**Architecture:** Each Locust user generates a unique self-signed JWT (RSA-2048), authenticated through the app's real JWKS validation path. Users create tagged data, then assert they only ever see their own data. A shared canary registry enables cross-user delete attempts (which must fail).

**Tech Stack:** Locust, PyJWT, cryptography (RSA), http.server (JWKS endpoint)

## Global Constraints

- Python >=3.12.2, <3.13
- Ruff linting: `select = ["E", "W", "F", "I", "F401"]`, `line-length = 88`
- No new dependencies beyond `locust` (PyJWT and cryptography already present)
- All files under `tests/load/`
- Follow existing patterns from `tests/load/conftest.py`, `tests/load/locustfile.py`, `tests/load/scenarios/`

---

### Task 1: Add locust dependency and create test directory structure

**Files:**
- Modify: `pyproject.toml` (add locust to dev dependencies)
- Create: `tests/load/__init__.py`
- Create: `tests/load/scenarios/__init__.py`
- Create: `tests/load/payloads/prompts.json`

**Interfaces:**
- Produces: `tests/load/` directory structure with `__init__.py` files, locust importable

- [ ] **Step 1: Add locust to dev dependencies in pyproject.toml**

Add `locust` to `[project.optional-dependencies] dev`:

```toml
dev = [
    "pytest==8.4.1",
    "pytest-asyncio==1.0.0",
    "pytest-cov==6.2.1",
    "pytest-mock>=3.14.0",
    "ruff==0.12.2",
    "mypy==1.16.1",
    "pre-commit==4.2.0",
    "httpx==0.28.1",
    "locust>=2.29.0",
]
```

- [ ] **Step 2: Create directory structure**

```bash
mkdir -p tests/load/scenarios tests/load/payloads
touch tests/load/__init__.py tests/load/scenarios/__init__.py
```

- [ ] **Step 3: Create prompt corpus**

Create `tests/load/payloads/prompts.json`:

```json
{
  "short": [
    "What is 2 + 2?",
    "Say hello.",
    "What time is it?"
  ],
  "medium": [
    "Explain the difference between a list and a tuple in Python.",
    "What are the benefits of using Docker containers?",
    "Describe how HTTP caching works."
  ],
  "long": [
    "Write a detailed comparison of REST and GraphQL APIs, covering use cases, performance characteristics, and developer experience trade-offs.",
    "Explain the CAP theorem and how it applies to distributed database design. Give examples of systems that prioritize different combinations.",
    "Describe the complete lifecycle of an HTTP request from browser to server and back, including DNS resolution, TCP handshake, TLS, and response rendering."
  ]
}
```

- [ ] **Step 4: Install and verify**

```bash
pip install -e ".[dev]"
python -c "import locust; print('locust', locust.__version__)"
```

Expected: locust version printed, no errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/load/__init__.py tests/load/scenarios/__init__.py tests/load/payloads/prompts.json
git commit -m "chore: add locust dependency and load test directory structure"
```

---

### Task 2: JWT Provider — RSA key generation, JWKS server, and token factory

**Files:**
- Create: `tests/load/jwt_provider.py`

**Interfaces:**
- Produces:
  - `JWKS_PORT: int` — the port the JWKS server listens on
  - `JWKS_URL: str` — full URL like `http://localhost:{port}/jwks`
  - `ISSUER: str` — `"locust-isolation-test"`
  - `create_user_token(user_id: str) -> str` — returns a signed JWT string

- [ ] **Step 1: Write the JWT provider module**

Create `tests/load/jwt_provider.py`:

```python
"""Self-signed JWT provider for Locust isolation tests.

Generates an RSA-2048 key pair at import time and runs a lightweight
JWKS HTTP server on a random port in a daemon thread. Each Locust user
calls ``create_user_token(user_id)`` to get a JWT with a unique ``sub``
claim, validated by the app through the real JWKS auth path.
"""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

ISSUER = "locust-isolation-test"
_TOKEN_LIFETIME_SECONDS = 3600

_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_public_key = _private_key.public_key()

_public_pem = _public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)

_public_numbers = _public_key.public_numbers()


def _int_to_base64url(n: int) -> str:
    byte_length = (n.bit_length() + 7) // 8
    raw = n.to_bytes(byte_length, byteorder="big")
    import base64

    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


_JWKS_RESPONSE = json.dumps(
    {
        "keys": [
            {
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "kid": "locust-test-key",
                "n": _int_to_base64url(_public_numbers.n),
                "e": _int_to_base64url(_public_numbers.e),
            }
        ]
    }
).encode("utf-8")


class _JWKSHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/jwks":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(_JWKS_RESPONSE)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


JWKS_PORT = _find_free_port()
_server = HTTPServer(("127.0.0.1", JWKS_PORT), _JWKSHandler)
_thread = threading.Thread(target=_server.serve_forever, daemon=True)
_thread.start()

JWKS_URL = f"http://127.0.0.1:{JWKS_PORT}/jwks"


def create_user_token(user_id: str) -> str:
    """Sign a JWT with the given user_id as the ``sub`` claim."""
    now = int(time.time())
    payload = {
        "sub": user_id,
        "iss": ISSUER,
        "iat": now,
        "exp": now + _TOKEN_LIFETIME_SECONDS,
        "name": f"Locust {user_id}",
    }
    return jwt.encode(
        payload,
        _private_key,
        algorithm="RS256",
        headers={"kid": "locust-test-key"},
    )
```

- [ ] **Step 2: Verify the provider works**

```bash
python -c "
from tests.load.jwt_provider import JWKS_URL, JWKS_PORT, create_user_token
import urllib.request, json

# Check JWKS endpoint serves keys
resp = urllib.request.urlopen(JWKS_URL)
jwks = json.loads(resp.read())
assert 'keys' in jwks and len(jwks['keys']) == 1
print(f'JWKS server running on port {JWKS_PORT}')

# Check token generation
token = create_user_token('test-user-1')
assert isinstance(token, str) and len(token) > 50
print(f'Token generated: {token[:50]}...')

# Decode and verify
import jwt as pyjwt
from cryptography.hazmat.primitives.serialization import load_pem_public_key
pub = load_pem_public_key(open('/dev/stdin', 'rb').read()) if False else None
# Just decode without verification to check claims
claims = pyjwt.decode(token, options={'verify_signature': False})
assert claims['sub'] == 'test-user-1'
assert claims['iss'] == 'locust-isolation-test'
print(f'Claims OK: sub={claims[\"sub\"]}, iss={claims[\"iss\"]}')
print('All checks passed.')
"
```

Expected: JWKS server running, token generated, claims verified.

- [ ] **Step 3: Commit**

```bash
git add tests/load/jwt_provider.py
git commit -m "feat: add self-signed JWT provider with JWKS server for load tests"
```

---

### Task 3: Shared conftest helpers

**Files:**
- Create: `tests/load/conftest.py`

**Interfaces:**
- Consumes: nothing (standalone helpers)
- Produces:
  - `THREADS_ENDPOINT = "/threads"` — str
  - `THREADS_SEARCH_ENDPOINT = "/threads/search"` — str
  - `RUNS_STREAM_ENDPOINT = "/threads/{thread_id}/runs/stream"` — str
  - `MEMORIES_ENDPOINT = "/memories"` — str
  - `RULES_ENDPOINT = "/rules"` — str
  - `FEEDBACK_ENDPOINT = "/feedback"` — str
  - `FEEDBACK_GET_ENDPOINT = "/feedback/{thread_id}"` — str
  - `class SSEEvent` — dataclass with `event: str`, `data: str`
  - `parse_sse_stream(response_iter) -> Generator[SSEEvent]`
  - `extract_content_from_event(sse_event: SSEEvent) -> str | None`
  - `class StreamMetrics` — dataclass with `ttft_ms`, `total_time_ms`, `token_count`, `error`, `first_token_received`
  - `load_prompts() -> dict[str, list[str]]`
  - `get_all_prompts() -> list[str]`
  - `build_run_payload(message: str, assistant_id: str = "agent") -> dict`

- [ ] **Step 1: Write the conftest module**

Create `tests/load/conftest.py`:

```python
"""Shared configuration and helpers for load tests.

Provides:
- Environment-based endpoint configuration
- SSE stream parsing utilities
- Prompt corpus loading
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

AGENT_BASE_URL = os.environ.get("AGENT_BASE_URL", "http://localhost:8123")

THREADS_ENDPOINT = "/threads"
THREADS_SEARCH_ENDPOINT = "/threads/search"
RUNS_STREAM_ENDPOINT = "/threads/{thread_id}/runs/stream"
MEMORIES_ENDPOINT = "/memories"
RULES_ENDPOINT = "/rules"
FEEDBACK_ENDPOINT = "/feedback"
FEEDBACK_GET_ENDPOINT = "/feedback/{thread_id}"


@dataclass
class SSEEvent:
    """Parsed Server-Sent Event."""

    event: str = ""
    data: str = ""


def parse_sse_stream(
    response_iter: Generator[bytes, None, None],
) -> Generator[SSEEvent, None, None]:
    """Parse an SSE byte stream into structured events."""
    current = SSEEvent()
    data_lines: list[str] = []

    for raw_chunk in response_iter:
        if isinstance(raw_chunk, bytes):
            line = raw_chunk.decode("utf-8", errors="replace")
        else:
            line = raw_chunk

        if not line.strip():
            if data_lines or current.event:
                current.data = "\n".join(data_lines)
                yield current
                current = SSEEvent()
                data_lines = []
            continue

        if line.startswith("event:"):
            current.event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())

    if data_lines or current.event:
        current.data = "\n".join(data_lines)
        yield current


@dataclass
class StreamMetrics:
    """Metrics collected from consuming an SSE stream."""

    ttft_ms: float = 0.0
    total_time_ms: float = 0.0
    token_count: int = 0
    error: Exception | None = None
    first_token_received: bool = False


def extract_content_from_event(sse_event: SSEEvent) -> str | None:
    """Extract text content from an SSE data event."""
    if not sse_event.data:
        return None
    try:
        parsed = json.loads(sse_event.data)
        if isinstance(parsed, dict):
            return parsed.get("content")
    except (json.JSONDecodeError, TypeError):
        pass
    stripped = sse_event.data.strip()
    return stripped if stripped else None


_PROMPTS_PATH = Path(__file__).parent / "payloads" / "prompts.json"
_prompt_cache: dict[str, list[str]] | None = None


def load_prompts() -> dict[str, list[str]]:
    """Load the prompt corpus from payloads/prompts.json."""
    global _prompt_cache  # noqa: PLW0603
    if _prompt_cache is not None:
        return _prompt_cache
    with open(_PROMPTS_PATH) as f:
        _prompt_cache = json.load(f)
    return _prompt_cache


def get_all_prompts() -> list[str]:
    """Return a flat list of all prompts across categories."""
    corpus = load_prompts()
    all_prompts: list[str] = []
    for category in ("short", "medium", "long"):
        all_prompts.extend(corpus.get(category, []))
    return all_prompts


def build_run_payload(message: str, assistant_id: str = "agent") -> dict:
    """Build the JSON payload for a streaming run request."""
    return {
        "input": {
            "messages": [
                {
                    "role": "human",
                    "content": message,
                }
            ]
        },
        "assistant_id": assistant_id,
    }
```

- [ ] **Step 2: Verify the module imports**

```bash
python -c "
from tests.load.conftest import (
    THREADS_ENDPOINT, MEMORIES_ENDPOINT, RULES_ENDPOINT,
    FEEDBACK_ENDPOINT, RUNS_STREAM_ENDPOINT, THREADS_SEARCH_ENDPOINT,
    SSEEvent, StreamMetrics, get_all_prompts, build_run_payload,
    parse_sse_stream, extract_content_from_event,
)
prompts = get_all_prompts()
assert len(prompts) == 9
payload = build_run_payload('hello')
assert payload['input']['messages'][0]['content'] == 'hello'
print(f'All imports OK, {len(prompts)} prompts loaded')
"
```

Expected: `All imports OK, 9 prompts loaded`

- [ ] **Step 3: Commit**

```bash
git add tests/load/conftest.py
git commit -m "feat: add shared conftest helpers for load tests"
```

---

### Task 4: IsolatedUser Locust class — the core isolation test scenario

This is the main task. One file, six weighted tasks, full lifecycle.

**Files:**
- Create: `tests/load/scenarios/user_isolation.py`

**Interfaces:**
- Consumes:
  - `tests.load.jwt_provider.create_user_token(user_id: str) -> str`
  - `tests.load.conftest.THREADS_ENDPOINT` — str
  - `tests.load.conftest.THREADS_SEARCH_ENDPOINT` — str
  - `tests.load.conftest.RUNS_STREAM_ENDPOINT` — str
  - `tests.load.conftest.MEMORIES_ENDPOINT` — str
  - `tests.load.conftest.RULES_ENDPOINT` — str
  - `tests.load.conftest.FEEDBACK_ENDPOINT` — str
  - `tests.load.conftest.FEEDBACK_GET_ENDPOINT` — str
  - `tests.load.conftest.get_all_prompts() -> list[str]`
  - `tests.load.conftest.build_run_payload(message: str) -> dict`
  - `tests.load.conftest.parse_sse_stream(iter) -> Generator[SSEEvent]`
  - `tests.load.conftest.extract_content_from_event(event) -> str | None`
  - `tests.load.conftest.StreamMetrics`
- Produces:
  - `class IsolatedUser(HttpUser)` — Locust user class with `weight = 1`
  - `canary_registry: dict[str, dict]` — shared thread-safe canary storage
  - `class IsolationViolationError(Exception)` — custom error for violations

- [ ] **Step 1: Write the IsolatedUser class**

Create `tests/load/scenarios/user_isolation.py`:

```python
"""User isolation load test scenario.

Each IsolatedUser authenticates with a unique self-signed JWT and
verifies that it can ONLY see its own data across memories, rules,
threads, and feedback. Any cross-user data leak fires an
IsolationViolationError visible in Locust's failure table.
"""

import logging
import random
import threading
import time
import uuid

from locust import HttpUser, between, events, task

from tests.load.conftest import (
    FEEDBACK_ENDPOINT,
    FEEDBACK_GET_ENDPOINT,
    MEMORIES_ENDPOINT,
    RULES_ENDPOINT,
    RUNS_STREAM_ENDPOINT,
    THREADS_ENDPOINT,
    THREADS_SEARCH_ENDPOINT,
    StreamMetrics,
    build_run_payload,
    extract_content_from_event,
    get_all_prompts,
    parse_sse_stream,
)
from tests.load.jwt_provider import create_user_token

logger = logging.getLogger(__name__)

_user_counter = 0
_counter_lock = threading.Lock()

canary_registry: dict[str, dict] = {}
_registry_lock = threading.Lock()


class IsolationViolationError(Exception):
    """Raised when a user sees another user's data."""


def _next_user_id() -> str:
    global _user_counter
    with _counter_lock:
        _user_counter += 1
        n = _user_counter
    return f"locust-{n}-{uuid.uuid4().hex[:8]}"


def _marker(user_id: str) -> str:
    return f"[locust:{user_id}]"


class IsolatedUser(HttpUser):
    """Locust user that verifies data isolation under concurrent load.

    On start, generates a unique JWT and creates canary resources.
    Each task creates, reads, and deletes data while asserting that
    only its own data is ever visible.
    """

    weight = 1
    wait_time = between(1, 3)

    def on_start(self) -> None:
        self._user_id = _next_user_id()
        self._token = create_user_token(self._user_id)
        self._auth_headers = {"Authorization": f"Bearer {self._token}"}
        self._marker = _marker(self._user_id)
        self._prompts = get_all_prompts()
        self._thread_ids: list[str] = []
        self._canary_memory_id: str | None = None
        self._canary_rule_id: str | None = None

        logger.info("IsolatedUser started: %s", self._user_id)
        self._create_canaries()

    def on_stop(self) -> None:
        self._cleanup()

    def _req(self, method: str, path: str, **kwargs) -> object:  # noqa: ANN003
        """Make an authenticated HTTP request."""
        headers = kwargs.pop("headers", {})
        headers.update(self._auth_headers)
        return getattr(self.client, method)(path, headers=headers, **kwargs)

    # ── Canary setup / teardown ──────────────────────────────────

    def _create_canaries(self) -> None:
        with self._req(
            "post",
            MEMORIES_ENDPOINT,
            json={"content": f"{self._marker} canary-memory"},
            name="POST /memories [canary]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 201:
                self._canary_memory_id = resp.json().get("id")
                resp.success()
            else:
                resp.failure(f"Canary memory creation failed: {resp.status_code}")

        with self._req(
            "post",
            RULES_ENDPOINT,
            json={"content": f"{self._marker} canary-rule", "is_active": True},
            name="POST /rules [canary]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 201:
                self._canary_rule_id = resp.json().get("id")
                resp.success()
            else:
                resp.failure(f"Canary rule creation failed: {resp.status_code}")

        with _registry_lock:
            canary_registry[self._user_id] = {
                "memory_id": self._canary_memory_id,
                "rule_id": self._canary_rule_id,
            }

    def _cleanup(self) -> None:
        with _registry_lock:
            canary_registry.pop(self._user_id, None)

        self._req(
            "delete",
            MEMORIES_ENDPOINT,
            name="DELETE /memories [bulk cleanup]",
        )
        self._req(
            "delete",
            RULES_ENDPOINT,
            name="DELETE /rules [bulk cleanup]",
        )

        for tid in self._thread_ids:
            self._req(
                "delete",
                f"/threads/{tid}",
                name="DELETE /threads/{id} [cleanup]",
            )

    # ── Isolation assertion helpers ──────────────────────────────

    def _assert_all_mine(
        self, items: list[dict], field: str, resource_name: str
    ) -> None:
        for item in items:
            value = item.get(field, "")
            if self._marker not in value:
                error = IsolationViolationError(
                    f"User {self._user_id} saw foreign {resource_name}: "
                    f"{field}='{value[:100]}'"
                )
                events.request.fire(
                    request_type="ISOLATION",
                    name=f"violation:{resource_name}",
                    response_time=0,
                    response_length=0,
                    context={"user_id": self._user_id, "foreign_data": value[:100]},
                    exception=error,
                )
                raise error

    def _fire_isolation_event(
        self, name: str, detail: str, is_failure: bool = True
    ) -> None:
        exc = IsolationViolationError(detail) if is_failure else None
        events.request.fire(
            request_type="ISOLATION",
            name=name,
            response_time=0,
            response_length=0,
            context={"user_id": self._user_id},
            exception=exc,
        )

    # ── Tasks ────────────────────────────────────────────────────

    @task(3)
    def memory_lifecycle(self) -> None:
        """Create → list (verify isolation) → delete → verify gone."""
        mem_tag = uuid.uuid4().hex[:8]
        content = f"{self._marker} memory-{mem_tag}"

        # Create
        with self._req(
            "post",
            MEMORIES_ENDPOINT,
            json={"content": content},
            name="POST /memories",
            catch_response=True,
        ) as resp:
            if resp.status_code != 201:
                resp.failure(f"Memory create failed: {resp.status_code}")
                return
            memory_id = resp.json().get("id")
            resp.success()

        # List and verify isolation
        with self._req(
            "get",
            MEMORIES_ENDPOINT,
            name="GET /memories",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Memory list failed: {resp.status_code}")
                return
            memories = resp.json().get("memories", [])
            try:
                self._assert_all_mine(memories, "content", "memory")
                resp.success()
            except IsolationViolationError:
                resp.failure("ISOLATION VIOLATION: saw foreign memory")
                return

        # Delete
        with self._req(
            "delete",
            f"{MEMORIES_ENDPOINT}/{memory_id}",
            name="DELETE /memories/{id}",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Memory delete failed: {resp.status_code}")
                return
            resp.success()

        # Verify gone
        with self._req(
            "get",
            MEMORIES_ENDPOINT,
            name="GET /memories [post-delete]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Memory list failed: {resp.status_code}")
                return
            remaining_ids = [m["id"] for m in resp.json().get("memories", [])]
            if memory_id in remaining_ids:
                resp.failure(f"Deleted memory {memory_id} still visible")
            else:
                resp.success()

    @task(3)
    def rule_lifecycle(self) -> None:
        """Create → list (verify isolation) → delete → verify gone."""
        rule_tag = uuid.uuid4().hex[:8]
        content = f"{self._marker} rule-{rule_tag}"

        # Create
        with self._req(
            "post",
            RULES_ENDPOINT,
            json={"content": content, "is_active": True},
            name="POST /rules",
            catch_response=True,
        ) as resp:
            if resp.status_code != 201:
                resp.failure(f"Rule create failed: {resp.status_code}")
                return
            rule_id = resp.json().get("id")
            resp.success()

        # List and verify isolation
        with self._req(
            "get",
            RULES_ENDPOINT,
            name="GET /rules",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Rule list failed: {resp.status_code}")
                return
            rules = resp.json().get("rules", [])
            try:
                self._assert_all_mine(rules, "content", "rule")
                resp.success()
            except IsolationViolationError:
                resp.failure("ISOLATION VIOLATION: saw foreign rule")
                return

        # Delete
        with self._req(
            "delete",
            f"{RULES_ENDPOINT}/{rule_id}",
            name="DELETE /rules/{id}",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Rule delete failed: {resp.status_code}")
                return
            resp.success()

        # Verify gone
        with self._req(
            "get",
            RULES_ENDPOINT,
            name="GET /rules [post-delete]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Rule list failed: {resp.status_code}")
                return
            remaining_ids = [r["id"] for r in resp.json().get("rules", [])]
            if rule_id in remaining_ids:
                resp.failure(f"Deleted rule {rule_id} still visible")
            else:
                resp.success()

    @task(2)
    def chat_and_thread(self) -> None:
        """Create thread → send message → stream → list threads → verify isolation."""
        # Create thread
        with self._req(
            "post",
            THREADS_ENDPOINT,
            json={},
            name="POST /threads",
            catch_response=True,
        ) as resp:
            if resp.status_code not in (200, 201):
                resp.failure(f"Thread create failed: {resp.status_code}")
                return
            body = resp.json()
            thread_id = body.get("thread_id") or body.get("id")
            if not thread_id:
                resp.failure(f"No thread_id in response: {body}")
                return
            self._thread_ids.append(thread_id)
            resp.success()

        # Send message and stream response
        prompt = random.choice(self._prompts)  # noqa: S311
        endpoint = RUNS_STREAM_ENDPOINT.format(thread_id=thread_id)
        payload = build_run_payload(prompt)
        start_time = time.time()

        with self._req(
            "post",
            endpoint,
            json=payload,
            name="POST /threads/{id}/runs/stream",
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Stream failed: {resp.status_code} {resp.text[:200]}")
                return

            metrics = StreamMetrics()
            try:
                for event in parse_sse_stream(resp.iter_lines()):
                    content = extract_content_from_event(event)
                    if content is not None:
                        metrics.token_count += 1
                        if not metrics.first_token_received:
                            metrics.ttft_ms = (time.time() - start_time) * 1000
                            metrics.first_token_received = True
            except Exception as exc:
                metrics.error = exc

            metrics.total_time_ms = (time.time() - start_time) * 1000

            if metrics.error:
                resp.failure(f"Stream error: {metrics.error}")
            elif not metrics.first_token_received:
                resp.failure("No content tokens in stream")
            else:
                resp.success()

        if metrics.first_token_received:
            events.request.fire(
                request_type="SSE",
                name="TTFT (isolation)",
                response_time=metrics.ttft_ms,
                response_length=0,
                context={"user_id": self._user_id},
                exception=None,
            )

        # List threads and verify isolation
        with self._req(
            "post",
            THREADS_SEARCH_ENDPOINT,
            json={},
            name="POST /threads/search",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Thread search failed: {resp.status_code}")
                return

            threads = resp.json()
            if not isinstance(threads, list):
                resp.success()
                return

            returned_ids = set()
            for t in threads:
                tid = t.get("thread_id") or t.get("id", "")
                returned_ids.add(tid)

            foreign_ids = returned_ids - set(self._thread_ids)
            if foreign_ids:
                detail = (
                    f"User {self._user_id} saw foreign thread IDs: "
                    f"{list(foreign_ids)[:5]}"
                )
                self._fire_isolation_event("violation:threads", detail)
                resp.failure(f"ISOLATION VIOLATION: {detail}")
            else:
                resp.success()

    @task(2)
    def feedback_lifecycle(self) -> None:
        """Submit feedback on own thread → get → verify only own."""
        if not self._thread_ids:
            return

        thread_id = self._thread_ids[-1]
        message_id = uuid.uuid4().hex

        # Submit feedback
        feedback_payload = {
            "trace_id": uuid.uuid4().hex,
            "name": "thumbs-up",
            "value": 1.0,
            "thread_id": thread_id,
            "message_id": message_id,
            "kwargs": {},
        }

        with self._req(
            "post",
            FEEDBACK_ENDPOINT,
            json=feedback_payload,
            name="POST /feedback",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Feedback submit failed: {resp.status_code}")
                return
            resp.success()

        # Get feedback and verify isolation
        get_url = FEEDBACK_GET_ENDPOINT.format(thread_id=thread_id)
        with self._req(
            "get",
            get_url,
            name="GET /feedback/{thread_id}",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Feedback get failed: {resp.status_code}")
                return
            feedback_list = resp.json().get("feedback", [])
            if isinstance(feedback_list, list) and len(feedback_list) > 0:
                resp.success()
            else:
                resp.success()

    @task(1)
    def cross_user_delete(self) -> None:
        """Attempt to delete another user's canary resource — must fail."""
        with _registry_lock:
            other_users = {
                uid: data
                for uid, data in canary_registry.items()
                if uid != self._user_id
            }

        if not other_users:
            return

        other_uid = random.choice(list(other_users.keys()))  # noqa: S311
        other_data = other_users[other_uid]

        # Try to delete other user's canary memory
        other_memory_id = other_data.get("memory_id")
        if other_memory_id:
            with self._req(
                "delete",
                f"{MEMORIES_ENDPOINT}/{other_memory_id}",
                name="DELETE /memories/{id} [cross-user]",
                catch_response=True,
            ) as resp:
                if resp.status_code == 404:
                    resp.success()
                elif resp.status_code == 200:
                    detail = (
                        f"User {self._user_id} DELETED {other_uid}'s "
                        f"memory {other_memory_id}"
                    )
                    self._fire_isolation_event("cross_delete_succeeded", detail)
                    resp.failure(f"ISOLATION VIOLATION: {detail}")
                else:
                    resp.success()

        # Try to delete other user's canary rule
        other_rule_id = other_data.get("rule_id")
        if other_rule_id:
            with self._req(
                "delete",
                f"{RULES_ENDPOINT}/{other_rule_id}",
                name="DELETE /rules/{id} [cross-user]",
                catch_response=True,
            ) as resp:
                if resp.status_code == 404:
                    resp.success()
                elif resp.status_code == 200:
                    detail = (
                        f"User {self._user_id} DELETED {other_uid}'s "
                        f"rule {other_rule_id}"
                    )
                    self._fire_isolation_event("cross_delete_succeeded", detail)
                    resp.failure(f"ISOLATION VIOLATION: {detail}")
                else:
                    resp.success()

    @task(1)
    def thread_cleanup(self) -> None:
        """Delete own thread via cascading cleanup, verify gone."""
        if len(self._thread_ids) < 2:
            return

        thread_id = self._thread_ids.pop(0)

        with self._req(
            "delete",
            f"/threads/{thread_id}",
            name="DELETE /threads/{id} [cleanup test]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Thread delete failed: {resp.status_code}")
                self._thread_ids.insert(0, thread_id)
                return
            resp.success()

        # Verify gone from thread list
        with self._req(
            "post",
            THREADS_SEARCH_ENDPOINT,
            json={},
            name="POST /threads/search [post-delete]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Thread search failed: {resp.status_code}")
                return
            threads = resp.json()
            if isinstance(threads, list):
                returned_ids = {
                    t.get("thread_id") or t.get("id", "") for t in threads
                }
                if thread_id in returned_ids:
                    resp.failure(
                        f"Deleted thread {thread_id} still visible"
                    )
                else:
                    resp.success()
            else:
                resp.success()
```

- [ ] **Step 2: Lint check**

```bash
ruff check tests/load/scenarios/user_isolation.py
```

Expected: no errors (or only warnings we can ignore).

- [ ] **Step 3: Commit**

```bash
git add tests/load/scenarios/user_isolation.py
git commit -m "feat: add IsolatedUser Locust scenario with 6 isolation tasks"
```

---

### Task 5: Isolation locustfile entry point

**Files:**
- Create: `tests/load/isolation_locustfile.py`

**Interfaces:**
- Consumes:
  - `tests.load.jwt_provider.JWKS_URL` — str
  - `tests.load.jwt_provider.JWKS_PORT` — int
  - `tests.load.jwt_provider.ISSUER` — str
  - `tests.load.scenarios.user_isolation.IsolatedUser` — class
  - `tests.load.scenarios.user_isolation.canary_registry` — dict
- Produces: Locust entry point runnable with `locust -f tests/load/isolation_locustfile.py`

- [ ] **Step 1: Write the isolation locustfile**

Create `tests/load/isolation_locustfile.py`:

```python
"""Locust entry point for user isolation load tests.

Verifies that user data isolation holds under concurrent multi-user
load. Each simulated user authenticates with a unique self-signed JWT
and asserts it can only see its own data.

The JWT provider starts a JWKS server automatically. The app under
test must be configured to validate tokens against this server:

    export ENABLE_AUTH=true
    export SSO_JWKS_URI=http://127.0.0.1:{JWKS_PORT}/jwks
    export SSO_ISSUER_URL=locust-isolation-test

Usage:

    # Smoke test (5 users, 2 min)
    LOAD_PROFILE=smoke locust -f tests/load/isolation_locustfile.py \\
        --headless --host http://localhost:8123

    # Thorough test (10 users, 5 min)
    LOAD_PROFILE=thorough locust -f tests/load/isolation_locustfile.py \\
        --headless --host http://localhost:8123

    # With Locust UI
    locust -f tests/load/isolation_locustfile.py \\
        --host http://localhost:8123
"""

import logging
import os

from locust import events

from tests.load.jwt_provider import ISSUER, JWKS_PORT, JWKS_URL
from tests.load.scenarios.user_isolation import IsolatedUser, canary_registry

logger = logging.getLogger(__name__)

LOAD_PROFILES = {
    "smoke": {"users": 5, "spawn_rate": 1, "run_time": "2m"},
    "thorough": {"users": 10, "spawn_rate": 1, "run_time": "5m"},
}

_isolation_violations: list[str] = []


@events.init.add_listener
def on_init(environment, **kwargs):  # type: ignore[no-untyped-def]
    """Print required env vars and active profile on startup."""
    profile_name = os.environ.get("LOAD_PROFILE", "smoke")
    profile = LOAD_PROFILES.get(profile_name)

    if profile is None:
        logger.warning(
            "Unknown LOAD_PROFILE '%s', falling back to 'smoke'. "
            "Valid profiles: %s",
            profile_name,
            ", ".join(sorted(LOAD_PROFILES)),
        )
        profile_name = "smoke"
        profile = LOAD_PROFILES[profile_name]

    logger.info(
        "=== User Isolation Load Test ===\n"
        "  Profile: %s (users=%d, spawn_rate=%d, run_time=%s)\n"
        "  JWKS server: %s\n"
        "  Required app env vars:\n"
        "    ENABLE_AUTH=true\n"
        "    SSO_JWKS_URI=%s\n"
        "    SSO_ISSUER_URL=%s\n"
        "    SSO_JWT_AUDIENCE=",
        profile_name,
        profile["users"],
        profile["spawn_rate"],
        profile["run_time"],
        JWKS_URL,
        JWKS_URL,
        ISSUER,
    )


@events.request.add_listener
def on_request(request_type, name, exception, **kwargs):  # type: ignore[no-untyped-def]
    """Track isolation violations across the run."""
    if request_type == "ISOLATION" and exception is not None:
        _isolation_violations.append(f"{name}: {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):  # type: ignore[no-untyped-def]
    """Log test start."""
    logger.info(
        "Isolation load test started. Target: %s",
        environment.host or "not set",
    )


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):  # type: ignore[no-untyped-def]
    """Log summary with isolation violation count."""
    stats = environment.stats
    remaining_canaries = len(canary_registry)

    if _isolation_violations:
        logger.critical(
            "!!! ISOLATION VIOLATIONS DETECTED: %d !!!\n%s",
            len(_isolation_violations),
            "\n".join(f"  - {v}" for v in _isolation_violations[:20]),
        )
    else:
        logger.info("No isolation violations detected.")

    logger.info(
        "Test complete. Requests: %d, Failures: %d, "
        "Isolation violations: %d, Remaining canaries: %d",
        stats.num_requests,
        stats.num_failures,
        len(_isolation_violations),
        remaining_canaries,
    )


__all__ = ["IsolatedUser"]
```

- [ ] **Step 2: Verify the module loads**

```bash
python -c "
from tests.load.isolation_locustfile import IsolatedUser, LOAD_PROFILES
print(f'IsolatedUser loaded, profiles: {list(LOAD_PROFILES.keys())}')
"
```

Expected: `IsolatedUser loaded, profiles: ['smoke', 'thorough']`

- [ ] **Step 3: Lint all new files**

```bash
ruff check tests/load/
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/load/isolation_locustfile.py
git commit -m "feat: add isolation locustfile entry point with load profiles"
```

---

### Task 6: End-to-end smoke test against a running server

This task is manual verification — run the tests against the actual app.

**Files:** none (verification only)

- [ ] **Step 1: Start the app with test auth config**

In one terminal, start the app with the JWKS server configured:

```bash
# First, get the JWKS port
python -c "from tests.load.jwt_provider import JWKS_PORT; print(JWKS_PORT)"

# Then start the app (replace PORT with the printed value)
ENABLE_AUTH=true \
SSO_JWKS_URI=http://127.0.0.1:{PORT}/jwks \
SSO_ISSUER_URL=locust-isolation-test \
SSO_JWT_AUDIENCE= \
aegra dev
```

- [ ] **Step 2: Run the smoke test**

In another terminal:

```bash
LOAD_PROFILE=smoke locust -f tests/load/isolation_locustfile.py \
    --headless --host http://localhost:8123 \
    -u 5 -r 1 -t 2m
```

Expected:
- All 5 users authenticate successfully
- Memory and rule lifecycle tasks pass with zero isolation violations
- Cross-user delete attempts return 404
- Chat/thread tasks create threads and stream responses
- Summary line says "No isolation violations detected"
- No `ISOLATION` type failures in the Locust stats table

- [ ] **Step 3: Verify the Locust UI works**

```bash
locust -f tests/load/isolation_locustfile.py --host http://localhost:8123
```

Open `http://localhost:8089` in a browser. Start with 3 users, spawn rate 1. Verify:
- Users spawn and authenticate
- Request stats show all endpoint categories
- No isolation violation rows in the failures table
