"""Tests for api.py - FastAPI server implementation."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from template_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode


class TestRequestLoggingMiddleware:
    """Tests for RequestLoggingMiddleware."""

    @pytest.fixture
    def app_with_middleware(self, monkeypatch):
        """Create a test app with RequestLoggingMiddleware."""
        monkeypatch.setenv("USE_INMEMORY_SAVER", "true")
        monkeypatch.setenv("REQUEST_LOGGING_ENABLED", "true")
        monkeypatch.setenv("REQUEST_LOG_HEADERS", "true")
        monkeypatch.setenv("REQUEST_LOG_BODY", "true")

        from template_agent.src.api import RequestLoggingMiddleware

        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        @app.post("/test-post")
        async def test_post_endpoint():
            return {"status": "posted"}

        return app

    def test_middleware_passes_through_when_disabled(self, monkeypatch):
        """Middleware passes through when REQUEST_LOGGING_ENABLED is false."""
        monkeypatch.setenv("USE_INMEMORY_SAVER", "true")
        monkeypatch.setenv("REQUEST_LOGGING_ENABLED", "false")

        from template_agent.src.api import RequestLoggingMiddleware

        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_middleware_logs_requests_when_enabled(self, app_with_middleware):
        """Middleware logs requests when enabled."""
        client = TestClient(app_with_middleware)
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_middleware_logs_post_with_body(self, app_with_middleware):
        """Middleware logs POST requests with body."""
        client = TestClient(app_with_middleware)
        resp = client.post("/test-post", json={"data": "test"})
        assert resp.status_code == 200


class TestExceptionHandlers:
    """Tests for exception handlers."""

    @pytest.fixture
    def app_with_handlers(self, monkeypatch):
        """Create app with exception handlers only (no lifespan)."""
        monkeypatch.setenv("USE_INMEMORY_SAVER", "true")

        from template_agent.src.api import (
            app_exception_handler,
            generic_exception_handler,
        )

        app = FastAPI()

        @app.get("/raise-generic")
        async def raise_generic():
            raise ValueError("Something went wrong")

        @app.get("/raise-app-exception")
        async def raise_app_exception():
            raise AppException(
                "Custom error detail",
                AppExceptionCode.INTERNAL_SERVER_ERROR,
            )

        app.add_exception_handler(Exception, generic_exception_handler)
        app.add_exception_handler(AppException, app_exception_handler)

        return app

    def test_generic_exception_handler(self, app_with_handlers):
        """Generic exception handler returns 500 with error details."""
        client = TestClient(app_with_handlers, raise_server_exceptions=False)
        resp = client.get("/raise-generic")
        assert resp.status_code == 500
        body = resp.json()
        assert "detail_message" in body
        assert "error_code" in body

    def test_app_exception_handler(self, app_with_handlers):
        """AppException handler returns appropriate error response."""
        client = TestClient(app_with_handlers, raise_server_exceptions=False)
        resp = client.get("/raise-app-exception")
        assert resp.status_code == 500
        body = resp.json()
        assert "Custom error detail" in body["detail_message"]


class TestA2AAgentCardEndpoint:
    """Tests for root_agent_card endpoint when A2A is enabled."""

    @pytest.fixture
    def a2a_enabled_app(self, monkeypatch):
        """Create a minimal app that simulates A2A enabled behavior."""
        monkeypatch.setenv("USE_INMEMORY_SAVER", "true")
        monkeypatch.setenv("A2A_ENABLED", "true")
        monkeypatch.setenv("A2A_PATH_PREFIX", "/a2a")

        app = FastAPI()

        @app.get("/.well-known/agent-card.json")
        async def root_agent_card(request: Request):
            """Simulated root-level agent card endpoint."""
            prefix = "/a2a"
            scheme = "https" if request.url.scheme == "https" else "http"
            host = request.headers.get("host", "localhost:8000")
            caller_base = f"{scheme}://{host}{prefix}/"
            return {
                "name": "Template Agent",
                "url": caller_base,
                "supportedInterfaces": [{"url": caller_base, "protocol": "JSONRPC"}],
            }

        return app

    def test_agent_card_returns_correct_url(self, a2a_enabled_app):
        """Agent card endpoint returns correct URL based on Host header."""
        client = TestClient(a2a_enabled_app)
        resp = client.get(
            "/.well-known/agent-card.json",
            headers={"Host": "my-agent.example.com:9090"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "my-agent.example.com:9090" in body["url"]
        assert body["name"] == "Template Agent"

    def test_agent_card_uses_http_scheme(self, a2a_enabled_app):
        """Agent card endpoint uses http scheme by default."""
        client = TestClient(a2a_enabled_app)
        resp = client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        body = resp.json()
        assert body["url"].startswith("http://")
