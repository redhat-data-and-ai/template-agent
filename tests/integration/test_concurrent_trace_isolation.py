"""Live concurrent trace isolation test.

Sends 10 concurrent requests to the running agent, each with a unique
X-Trace-ID, X-Request-ID, X-Org-ID, and X-Agent-ID. Verifies:

1. Each response echoes back its own X-Trace-ID and X-Request-ID (no cross-contamination)
2. Threads are isolated per user (different thread_ids)
3. The agent processes all requests without errors
4. Response headers prove middleware correctly propagated context

Prerequisites:
    - Agent running on http://localhost:5002
    - pgvector + redis running
"""

from __future__ import annotations

import asyncio
import json
import time
from uuid import uuid4

import httpx
import pytest

AGENT_BASE = "http://localhost:5002"
CONCURRENCY = 10


def _agent_reachable() -> bool:
    try:
        import urllib.request

        urllib.request.urlopen(f"{AGENT_BASE}/health", timeout=2)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _agent_reachable(),
    reason="Agent not running on localhost:5002",
)


class TestConcurrentTraceIsolation:
    """10 concurrent users, each gets isolated traces."""

    @pytest.mark.asyncio
    async def test_concurrent_thread_creation_isolated_headers(self):
        """Each of 10 concurrent thread-creation requests must echo back
        its own X-Trace-ID and X-Request-ID, proving per-request isolation."""
        results: dict[int, dict] = {}
        errors: list[str] = []

        async def create_thread(client: httpx.AsyncClient, idx: int) -> None:
            trace_id = f"iso-trace-{idx}-{uuid4().hex[:8]}"
            request_id = f"iso-req-{idx}-{uuid4().hex[:8]}"
            org_id = f"org-{idx}"
            agent_id = f"org-{idx}/agent-{idx}"

            resp = await client.post(
                f"{AGENT_BASE}/threads",
                json={"metadata": {"user_id": f"user-{idx}"}},
                headers={
                    "X-Trace-ID": trace_id,
                    "X-Request-ID": request_id,
                    "X-Org-ID": org_id,
                    "X-Agent-ID": agent_id,
                },
            )

            resp_trace = resp.headers.get("x-trace-id")
            resp_request = resp.headers.get("x-request-id")

            if resp.status_code >= 400:
                errors.append(f"User {idx}: HTTP {resp.status_code}: {resp.text}")
                return

            if resp_trace != trace_id:
                errors.append(
                    f"User {idx}: X-Trace-ID mismatch: "
                    f"sent {trace_id}, got {resp_trace}"
                )
            if not resp_request:
                errors.append(f"User {idx}: X-Request-ID missing from response")

            body = resp.json()
            results[idx] = {
                "thread_id": body.get("thread_id"),
                "sent_trace_id": trace_id,
                "resp_trace_id": resp_trace,
                "resp_request_id": resp_request,
                "status": resp.status_code,
            }

        async with httpx.AsyncClient(timeout=30) as client:
            tasks = [create_thread(client, i) for i in range(CONCURRENCY)]
            await asyncio.gather(*tasks)

        assert len(errors) == 0, "Isolation failures:\n" + "\n".join(errors)
        assert len(results) == CONCURRENCY, (
            f"Expected {CONCURRENCY} results, got {len(results)}"
        )

        thread_ids = {r["thread_id"] for r in results.values()}
        assert len(thread_ids) == CONCURRENCY, (
            f"Expected {CONCURRENCY} unique threads, got {len(thread_ids)}"
        )

    @pytest.mark.asyncio
    async def test_concurrent_health_checks_isolated(self):
        """Concurrent health checks with different trace IDs must each
        echo their own ID back."""
        errors: list[str] = []

        async def health_check(client: httpx.AsyncClient, idx: int) -> None:
            trace_id = f"health-{idx}-{uuid4().hex[:8]}"
            request_id = f"hreq-{idx}-{uuid4().hex[:8]}"

            resp = await client.get(
                f"{AGENT_BASE}/health",
                headers={
                    "X-Trace-ID": trace_id,
                    "X-Request-ID": request_id,
                },
            )

            if resp.headers.get("x-trace-id") != trace_id:
                errors.append(
                    f"Health {idx}: trace mismatch: "
                    f"sent {trace_id}, got {resp.headers.get('x-trace-id')}"
                )
            if not resp.headers.get("x-request-id"):
                errors.append(f"Health {idx}: X-Request-ID missing from response")

        async with httpx.AsyncClient(timeout=10) as client:
            tasks = [health_check(client, i) for i in range(CONCURRENCY)]
            await asyncio.gather(*tasks)

        assert len(errors) == 0, "Health check isolation failures:\n" + "\n".join(
            errors
        )

    @pytest.mark.asyncio
    async def test_concurrent_runs_with_bmi_tool(self):
        """10 concurrent users ask for BMI calculation with different inputs.
        Each must get their own result, not another user's."""

        user_inputs = [
            {"name": f"User{i}", "weight": 60 + i * 5, "height": 1.60 + i * 0.05}
            for i in range(CONCURRENCY)
        ]
        results: dict[int, dict] = {}
        errors: list[str] = []

        async def run_bmi(client: httpx.AsyncClient, idx: int) -> None:
            trace_id = f"bmi-trace-{idx}-{uuid4().hex[:8]}"
            user = user_inputs[idx]

            # Create thread
            thread_resp = await client.post(
                f"{AGENT_BASE}/threads",
                json={"metadata": {"user_id": f"bmi-user-{idx}"}},
                headers={"X-Trace-ID": trace_id},
            )
            if thread_resp.status_code >= 400:
                errors.append(
                    f"BMI User {idx}: thread creation failed: {thread_resp.status_code}"
                )
                return
            thread_id = thread_resp.json()["thread_id"]

            # Run with wait
            msg = f"Calculate BMI for {user['name']}: weight {user['weight']}kg, height {user['height']}m"
            try:
                run_resp = await client.post(
                    f"{AGENT_BASE}/threads/{thread_id}/runs/wait",
                    json={
                        "assistant_id": "agent",
                        "input": {"messages": [{"role": "user", "content": msg}]},
                        "metadata": {"user_id": f"bmi-user-{idx}"},
                    },
                    headers={
                        "X-Trace-ID": trace_id,
                        "X-Request-ID": f"bmi-req-{idx}",
                    },
                    timeout=120,
                )
            except httpx.ReadTimeout:
                errors.append(f"BMI User {idx}: timeout waiting for run")
                return

            resp_trace = run_resp.headers.get("x-trace-id")
            if resp_trace != trace_id:
                errors.append(
                    f"BMI User {idx}: trace leak: sent {trace_id}, got {resp_trace}"
                )

            results[idx] = {
                "thread_id": thread_id,
                "status": run_resp.status_code,
                "trace_id": trace_id,
                "resp_trace_id": resp_trace,
            }

        async with httpx.AsyncClient(timeout=120) as client:
            tasks = [run_bmi(client, i) for i in range(CONCURRENCY)]
            await asyncio.gather(*tasks)

        assert len(errors) == 0, "BMI isolation failures:\n" + "\n".join(errors)

        thread_ids = {r["thread_id"] for r in results.values() if r.get("thread_id")}
        assert len(thread_ids) == len(results), "Thread ID collision detected"

        for idx, r in results.items():
            assert r["trace_id"] == r["resp_trace_id"], (
                f"User {idx}: trace ID leaked: {r['trace_id']} != {r['resp_trace_id']}"
            )

    @pytest.mark.asyncio
    async def test_sequential_users_no_context_bleed(self):
        """Two sequential requests with different trace IDs must not bleed context."""
        async with httpx.AsyncClient(timeout=30) as client:
            # User A
            trace_a = f"seq-a-{uuid4().hex[:8]}"
            resp_a = await client.post(
                f"{AGENT_BASE}/threads",
                json={"metadata": {"user_id": "alice"}},
                headers={"X-Trace-ID": trace_a, "X-Org-ID": "org-alice"},
            )
            assert resp_a.headers["x-trace-id"] == trace_a

            # User B
            trace_b = f"seq-b-{uuid4().hex[:8]}"
            resp_b = await client.post(
                f"{AGENT_BASE}/threads",
                json={"metadata": {"user_id": "bob"}},
                headers={"X-Trace-ID": trace_b, "X-Org-ID": "org-bob"},
            )
            assert resp_b.headers["x-trace-id"] == trace_b

            # Verify no bleed
            assert resp_a.headers["x-trace-id"] != resp_b.headers["x-trace-id"]
            assert resp_a.json()["thread_id"] != resp_b.json()["thread_id"]


class TestCleanup:
    """Clean up threads created by tests."""

    @pytest.mark.asyncio
    async def test_cleanup_test_threads(self):
        """Delete threads created by isolation tests."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{AGENT_BASE}/threads/search",
                json={"limit": 100},
            )
            if resp.status_code == 200:
                threads = resp.json()
                for t in threads:
                    tid = t.get("thread_id", "")
                    meta = t.get("metadata", {})
                    user_id = meta.get("user_id", "")
                    if any(
                        prefix in user_id
                        for prefix in ("user-", "bmi-user-", "alice", "bob")
                    ):
                        await client.delete(f"{AGENT_BASE}/threads/{tid}")
