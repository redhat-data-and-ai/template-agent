"""Tests for X-Request-ID, X-Org-ID, X-Agent-ID extraction and log binding."""

from __future__ import annotations

import json
import uuid

import pytest

from deep_agent.utils.pylogger import (
    _agent_id_var,
    _org_id_var,
    _request_id_var,
    bind_request_context,
    clear_request_context,
)


@pytest.fixture(autouse=True)
def _clean_context():
    clear_request_context()
    yield
    clear_request_context()


# ---------------------------------------------------------------------------
# Context-var helpers
# ---------------------------------------------------------------------------


class TestBindRequestContext:
    def test_bind_request_id(self):
        bind_request_context(request_id="rid-1")
        assert _request_id_var.get() == "rid-1"

    def test_bind_org_and_agent_id(self):
        bind_request_context(org_id="acme", agent_id="acme/bot")
        assert _org_id_var.get() == "acme"
        assert _agent_id_var.get() == "acme/bot"

    def test_clear_resets_all(self):
        bind_request_context(request_id="x", org_id="y", agent_id="z")
        clear_request_context()
        assert _request_id_var.get() is None
        assert _org_id_var.get() is None
        assert _agent_id_var.get() is None

    def test_backward_compat_trace_id(self):
        """Existing trace_id / user_id / thread_id params still work."""
        bind_request_context(trace_id="t1", user_id="u1", thread_id="th1")
        from deep_agent.utils.pylogger import (
            _thread_id_var,
            _trace_id_var,
            _user_id_var,
        )

        assert _trace_id_var.get() == "t1"
        assert _user_id_var.get() == "u1"
        assert _thread_id_var.get() == "th1"


# ---------------------------------------------------------------------------
# Structlog processor test
# ---------------------------------------------------------------------------


def test_request_id_injected_into_log_event():
    """Verify _inject_request_context adds request_id/org_id/agent_id to event dict."""
    from deep_agent.utils.pylogger import _inject_request_context

    bind_request_context(request_id="log-rid", org_id="myorg", agent_id="myorg/agent-x")
    event: dict = {"event": "test_event"}
    result = _inject_request_context(None, "info", event)
    clear_request_context()

    assert result["request_id"] == "log-rid"
    assert result["org_id"] == "myorg"
    assert result["agent_id"] == "myorg/agent-x"
    assert result.get("service") is not None


# ---------------------------------------------------------------------------
# Middleware tests (RequestContextMiddleware in http_app)
# ---------------------------------------------------------------------------


class TestRequestContextMiddleware:
    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient

        from deep_agent.aegra.http_app import app

        return TestClient(app)

    def test_generates_request_id_when_absent(self, client):
        r = client.get("/health")
        rid = r.headers.get("x-request-id")
        assert rid is not None
        uuid.UUID(rid)

    def test_preserves_incoming_request_id(self, client):
        r = client.get("/health", headers={"X-Request-ID": "agent-42"})
        assert r.headers["x-request-id"] == "agent-42"

    def test_preserves_trace_id(self, client):
        r = client.get("/health", headers={"X-Trace-ID": "trace-abc"})
        assert r.headers["x-trace-id"] == "trace-abc"

    def test_both_ids_returned(self, client):
        r = client.get(
            "/health",
            headers={"X-Request-ID": "req-1", "X-Trace-ID": "trace-1"},
        )
        assert r.headers["x-request-id"] == "req-1"
        assert r.headers["x-trace-id"] == "trace-1"
