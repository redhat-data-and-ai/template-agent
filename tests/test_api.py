"""Tests for api.py - FastAPI server implementation."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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


