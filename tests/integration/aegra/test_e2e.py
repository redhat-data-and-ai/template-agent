"""End-to-end test: Full aegra deployment (MR-32).

Tests the complete LangGraph Platform API contract by verifying
health, thread creation, agent invocation, and state retrieval.

Requires: ``langgraph dev`` or ``langgraph up`` running on LANGGRAPH_API_URL.
Mark: ``pytest -m e2e`` to run these tests separately.
"""

import os

import httpx
import pytest

pytestmark = pytest.mark.e2e

LANGGRAPH_API_URL = os.environ.get("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
ASSISTANT_ID = "agent"


def _api_url(path: str) -> str:
    return f"{LANGGRAPH_API_URL}{path}"


@pytest.fixture()
def client():
    with httpx.Client(base_url=LANGGRAPH_API_URL, timeout=60) as c:
        yield c


class TestAegraHealthEndpoint:
    """Verify the LangGraph Platform health endpoint."""

    def test_health_ok(self, client):
        resp = client.get("/ok")
        assert resp.status_code == 200

    def test_info_endpoint(self, client):
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data


class TestAegraAssistants:
    """Verify assistants are registered correctly."""

    def test_list_assistants(self, client):
        resp = client.post("/assistants/search", json={})
        assert resp.status_code == 200
        assistants = resp.json()
        assert len(assistants) >= 1

    def test_agent_assistant_exists(self, client):
        resp = client.get(f"/assistants/{ASSISTANT_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["assistant_id"] == ASSISTANT_ID


class TestAegraThreadLifecycle:
    """Verify thread creation, retrieval, and deletion."""

    def test_create_thread(self, client):
        resp = client.post("/threads", json={})
        assert resp.status_code == 200
        thread = resp.json()
        assert "thread_id" in thread

    def test_create_and_get_thread(self, client):
        create_resp = client.post("/threads", json={})
        thread_id = create_resp.json()["thread_id"]

        get_resp = client.get(f"/threads/{thread_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["thread_id"] == thread_id

    def test_delete_thread(self, client):
        create_resp = client.post("/threads", json={})
        thread_id = create_resp.json()["thread_id"]

        del_resp = client.delete(f"/threads/{thread_id}")
        assert del_resp.status_code == 200


class TestAegraAgentInvocation:
    """Test agent invocation via the LangGraph API.

    These tests exercise the actual agent graph — they require
    valid Google credentials and a running mock MCP server.
    """

    @pytest.mark.slow
    def test_invoke_returns_response(self, client):
        thread_resp = client.post("/threads", json={})
        thread_id = thread_resp.json()["thread_id"]

        resp = client.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": ASSISTANT_ID,
                "input": {
                    "messages": [
                        {"role": "human", "content": "Hello, what can you do?"}
                    ]
                },
            },
        )
        assert resp.status_code in (200, 201, 202)

    @pytest.mark.slow
    def test_stream_returns_events(self, client):
        thread_resp = client.post("/threads", json={})
        thread_id = thread_resp.json()["thread_id"]

        with client.stream(
            "POST",
            f"/threads/{thread_id}/runs/stream",
            json={
                "assistant_id": ASSISTANT_ID,
                "input": {"messages": [{"role": "human", "content": "Say hello"}]},
                "stream_mode": "updates",
            },
        ) as resp:
            assert resp.status_code == 200
            events = list(resp.iter_lines())
            assert len(events) > 0
