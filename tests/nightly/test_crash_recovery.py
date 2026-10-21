"""Kill-and-restart: does an interrupted run actually resume?

Aegra's feature matrix claims lease-based crash recovery — a worker that
dies mid-run has its lease expire and the run re-executed from its last
checkpoint. We depend on that for durability, and a claim in a
dependency's docs is not a test of our deployment.

Nightly, not per-PR: it needs a real stack with Redis and Postgres, and
a run long enough to kill in the middle.

Against the local Kind cluster (`make kind`):

    export AGENT_URL=http://agent.localhost:8081
    export WORKER_KILL_COMMAND="kubectl --context kind-template-agent \
        -n template-agent delete pod \
        $(kubectl --context kind-template-agent -n template-agent get pod \
          -l component=agent -o name | head -1) --force --grace-period=0"
    make test-nightly

Two things the endpoint has to satisfy, both learned the hard way. It must
not be a `kubectl port-forward`: a forward binds to one pod and dies with
it, so every assertion after the kill fails with a transport error that
looks exactly like a recovery failure. And the deployment needs more than
one replica, so the Service still has an endpoint while the killed pod is
replaced. The Kind overlay runs two, as preprod and prod do.

Skips cleanly when no stack is reachable, so running the suite on a
laptop is not a failure.
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

pytestmark = [pytest.mark.nightly, pytest.mark.slow]

AGENT_URL = os.getenv("AGENT_URL", "http://localhost:5002")
ASSISTANT_ID = os.getenv("AGENT_ASSISTANT_ID", "agent")
# Auth is off in Kind; preprod and prod need a bearer token.
AGENT_TOKEN = os.getenv("AGENT_TOKEN", "")
# Asks the Kind mock model to take its time, per request, so this tier
# does not have to slow the cluster down for everyone while it runs.
SLOW_MARKER = os.getenv("MOCK_LLM_SLOW_MARKER", "__slow__")
_HEADERS = {"Authorization": f"Bearer {AGENT_TOKEN}"} if AGENT_TOKEN else {}


def _stack_is_up() -> bool:
    try:
        probe = httpx.get(f"{AGENT_URL}/health", headers=_HEADERS, timeout=3)
        return probe.status_code == 200
    except Exception:
        return False


pytestmark.append(
    pytest.mark.skipif(
        not _stack_is_up(),
        reason=f"no agent reachable at {AGENT_URL}; start one with `make local`",
    )
)


@pytest.fixture
def client() -> httpx.Client:
    with httpx.Client(base_url=AGENT_URL, headers=_HEADERS, timeout=30) as c:
        yield c


def _kill_the_worker(kill_command: str) -> None:
    """Run the operator's kill command, and insist that it worked.

    `os.system`'s exit status was discarded, so a kill that failed —
    a wrong label, no permission — let the run finish normally and
    the test passed with no recovery exercised at all.
    """
    status = os.system(kill_command)  # noqa: S605 — operator-supplied, nightly only
    assert status == 0, f"kill command failed (exit {status}): {kill_command}"


def _wait_for_a_live_worker(client: httpx.Client, timeout: float = 180) -> None:
    """Poll until a replacement pod answers."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if client.get("/health").status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise AssertionError("no worker came back after the kill")


def _create_thread(client: httpx.Client) -> str:
    response = client.post("/threads", json={})
    response.raise_for_status()
    return response.json()["thread_id"]


class TestLeaseRecovery:
    def test_interrupted_run_is_resumed(self, client: httpx.Client):
        """Start a run, kill its worker, assert it finishes anyway.

        Killing the worker is deployment-specific — on Kind or OpenShift
        this is a pod delete. The env var keeps the assertion here and
        the mechanism outside.
        """
        kill_command = os.getenv("WORKER_KILL_COMMAND")
        if not kill_command:
            pytest.skip(
                "set WORKER_KILL_COMMAND to the command that kills a worker, "
                "e.g. 'oc delete pod -l component=agent --force'"
            )

        thread_id = _create_thread(client)
        run = client.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": ASSISTANT_ID,
                # The marker asks the mock model for a slow response, so
                # the run is still going when the kill lands two seconds
                # from now. Against a real model, drop it — any real
                # model is slow enough. See the mock-llm component.
                "input": {
                    "messages": [
                        {"type": "human", "content": f"count to ten {SLOW_MARKER}"}
                    ]
                },
            },
        )
        run.raise_for_status()
        run_id = run.json()["run_id"]

        # Let it get past scheduling, then take the worker out.
        time.sleep(2)

        # The run must still be going, or the kill lands on nothing and
        # the assertion below passes without recovery ever happening.
        # This is the difference between a durability test and a test
        # that a fast model finishes quickly.
        before = client.get(f"/threads/{thread_id}/runs/{run_id}")
        status_before = (
            before.json().get("status") if before.status_code == 200 else None
        )
        assert status_before in ("pending", "running"), (
            f"run was already {status_before!r} before the kill — nothing was "
            "interrupted. Slow the model down (MOCK_LLM_DELAY_MS) so the run "
            "is genuinely in flight."
        )

        _kill_the_worker(kill_command)

        # The kill takes endpoints out of the Service; a connection error
        # here is the pod dying, not the run failing. Keep polling.
        deadline = time.time() + 180
        status = None
        while time.time() < deadline:
            try:
                probe = client.get(f"/threads/{thread_id}/runs/{run_id}")
            except httpx.HTTPError:
                time.sleep(5)
                continue
            if probe.status_code == 200:
                status = probe.json().get("status")
                if status in {"success", "error", "interrupted"}:
                    break
            time.sleep(5)

        assert status == "success", (
            f"run did not recover after the worker was killed (status={status}). "
            "Aegra's lease reaper should re-execute from the last checkpoint."
        )

    def test_thread_state_survives_a_restart(self, client: httpx.Client):
        """Checkpoints are in Postgres, so state must outlive the process."""
        kill_command = os.getenv("WORKER_KILL_COMMAND")
        if not kill_command:
            pytest.skip(
                "set WORKER_KILL_COMMAND to the command that kills a worker, "
                "e.g. 'oc delete pod -l component=agent --force'"
            )

        thread_id = _create_thread(client)

        client.post(
            f"/threads/{thread_id}/runs/wait",
            json={
                "assistant_id": ASSISTANT_ID,
                "input": {"messages": [{"type": "human", "content": "remember: 42"}]},
            },
        ).raise_for_status()

        # Restart before reading, or this asserts nothing: an
        # in-memory checkpointer answers `/history` perfectly well
        # for as long as the process that wrote it is alive, which
        # is exactly the case the test is meant to rule out.
        _kill_the_worker(kill_command)
        _wait_for_a_live_worker(client)

        history = client.get(f"/threads/{thread_id}/history")
        history.raise_for_status()

        assert history.json(), "thread history empty — checkpoints not persisted"
