"""Tests for A2A middleware - version defaulting and auth handling."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import jwt
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from template_agent.src.a2a.context import a2a_request_ctx
from template_agent.src.a2a.middleware import (
    A2AAuthMiddleware,
    A2AVersionDefaultMiddleware,
    _extract_bearer,
    _validate_jwt,
)
from template_agent.src.settings import Settings

JWT_SECRET = "test-secret-key-for-hs256"


def _settings(**overrides) -> Settings:
    defaults = {
        "USE_INMEMORY_SAVER": True,
        "A2A_ENABLED": True,
        "A2A_AUTH_REQUIRED": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestExtractBearer:
    """Tests for _extract_bearer helper function."""

    def test_extracts_bearer_token(self):
        """Extracts token from 'Bearer <token>' format."""
        assert _extract_bearer("Bearer my-token") == "my-token"

    def test_extracts_bearer_token_lowercase(self):
        """Extracts token when 'bearer' is lowercase."""
        assert _extract_bearer("bearer my-token") == "my-token"

    def test_extracts_bearer_token_mixed_case(self):
        """Extracts token when 'BEARER' is uppercase."""
        assert _extract_bearer("BEARER my-token") == "my-token"

    def test_returns_none_for_empty(self):
        """Returns None for empty string."""
        assert _extract_bearer("") is None

    def test_returns_none_for_none(self):
        """Returns None for None input."""
        assert _extract_bearer(None) is None

    def test_returns_none_for_basic_auth(self):
        """Returns None for Basic auth scheme."""
        assert _extract_bearer("Basic base64string") is None

    def test_returns_none_for_bearer_only(self):
        """Returns None when just 'Bearer' without token."""
        assert _extract_bearer("Bearer") is None

    def test_returns_none_for_bearer_with_empty_token(self):
        """Returns None when token is whitespace only."""
        assert _extract_bearer("Bearer   ") is None


class TestValidateJwt:
    """Tests for _validate_jwt function."""

    def _make_token(self, secret: str = JWT_SECRET, **extra_claims) -> str:
        payload = {
            "sub": "test-agent",
            "iat": int(time.time()),
            "exp": int(time.time()) + 300,
        }
        payload.update(extra_claims)
        return jwt.encode(payload, secret, algorithm="HS256")

    def test_valid_jwt_with_secret(self):
        """Valid JWT with correct secret passes."""
        cfg = _settings(A2A_JWT_SECRET=JWT_SECRET)
        token = self._make_token()
        assert _validate_jwt(token, cfg) is True

    def test_invalid_jwt_with_wrong_secret(self):
        """JWT with wrong secret fails."""
        cfg = _settings(A2A_JWT_SECRET=JWT_SECRET)
        token = self._make_token(secret="wrong-secret")
        assert _validate_jwt(token, cfg) is False

    def test_expired_jwt(self):
        """Expired JWT fails validation."""
        cfg = _settings(A2A_JWT_SECRET=JWT_SECRET)
        token = self._make_token(exp=int(time.time()) - 60)
        assert _validate_jwt(token, cfg) is False

    def test_malformed_jwt(self):
        """Malformed JWT fails validation."""
        cfg = _settings(A2A_JWT_SECRET=JWT_SECRET)
        assert _validate_jwt("not.a.valid.jwt", cfg) is False

    def test_presence_only_when_no_config(self):
        """Presence-only check passes when no secret/JWKS configured."""
        cfg = _settings()
        assert _validate_jwt("any-opaque-token", cfg) is True

    def test_jwt_with_audience_validation(self):
        """JWT with correct audience passes."""
        cfg = _settings(
            A2A_JWT_SECRET=JWT_SECRET,
            A2A_JWT_AUDIENCE="expected-audience",
        )
        token = self._make_token(aud="expected-audience")
        assert _validate_jwt(token, cfg) is True

    def test_jwt_with_wrong_audience(self):
        """JWT with wrong audience fails."""
        cfg = _settings(
            A2A_JWT_SECRET=JWT_SECRET,
            A2A_JWT_AUDIENCE="expected-audience",
        )
        token = self._make_token(aud="wrong-audience")
        assert _validate_jwt(token, cfg) is False

    def test_jwt_with_issuer_validation(self):
        """JWT with correct issuer passes."""
        cfg = _settings(
            A2A_JWT_SECRET=JWT_SECRET,
            A2A_JWT_ISSUER="expected-issuer",
        )
        token = self._make_token(iss="expected-issuer")
        assert _validate_jwt(token, cfg) is True

    def test_jwt_with_wrong_issuer(self):
        """JWT with wrong issuer fails."""
        cfg = _settings(
            A2A_JWT_SECRET=JWT_SECRET,
            A2A_JWT_ISSUER="expected-issuer",
        )
        token = self._make_token(iss="wrong-issuer")
        assert _validate_jwt(token, cfg) is False

    def test_jwks_validation_failure(self):
        """JWKS validation failure returns False."""
        cfg = _settings(A2A_JWT_JWKS_URL="https://example.com/.well-known/jwks.json")

        with patch("jwt.PyJWKClient") as MockJWKClient:
            mock_client = MockJWKClient.return_value
            mock_client.get_signing_key_from_jwt.side_effect = jwt.PyJWTError(
                "JWKS error"
            )
            assert _validate_jwt("some-token", cfg) is False


class TestA2AVersionDefaultMiddleware:
    """Tests for A2AVersionDefaultMiddleware."""

    @pytest.fixture
    def app_with_middleware(self):
        """Create test app with version default middleware."""

        async def echo_headers(request: Request) -> PlainTextResponse:
            headers_dict = dict(request.scope.get("headers", []))
            a2a_version = headers_dict.get(b"a2a-version", b"not-set").decode()
            return PlainTextResponse(f"a2a-version={a2a_version}")

        app = Starlette(routes=[Route("/", echo_headers, methods=["POST", "GET"])])
        app.add_middleware(A2AVersionDefaultMiddleware)
        return app

    def test_injects_version_for_v1_method(self, app_with_middleware):
        """Injects A2A-Version: 1.0 for v1.0 methods."""
        client = TestClient(app_with_middleware)
        resp = client.post(
            "/",
            json={"method": "SendMessage", "params": {}},
        )
        assert resp.status_code == 200
        assert "a2a-version=1.0" in resp.text

    def test_does_not_inject_when_header_present(self, app_with_middleware):
        """Does not override existing A2A-Version header."""
        client = TestClient(app_with_middleware)
        resp = client.post(
            "/",
            json={"method": "SendMessage", "params": {}},
            headers={"A2A-Version": "0.3"},
        )
        assert resp.status_code == 200
        assert "a2a-version=0.3" in resp.text

    def test_does_not_inject_for_v03_method(self, app_with_middleware):
        """Does not inject for v0.3 method names."""
        client = TestClient(app_with_middleware)
        resp = client.post(
            "/",
            json={"method": "message/send", "params": {}},
        )
        assert resp.status_code == 200
        assert "a2a-version=not-set" in resp.text

    def test_does_not_inject_for_get_request(self, app_with_middleware):
        """Does not modify GET requests."""
        client = TestClient(app_with_middleware)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "a2a-version=not-set" in resp.text

    def test_handles_invalid_json_gracefully(self, app_with_middleware):
        """Handles invalid JSON body gracefully."""
        client = TestClient(app_with_middleware)
        resp = client.post(
            "/",
            content="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200

    def test_all_v1_methods_recognized(self, app_with_middleware):
        """All v1.0 method names trigger version injection."""
        v1_methods = [
            "SendMessage",
            "SendStreamingMessage",
            "GetTask",
            "CancelTask",
            "ListTasks",
            "SubscribeToTask",
            "CreateTaskPushNotificationConfig",
            "GetTaskPushNotificationConfig",
            "ListTaskPushNotificationConfigs",
            "DeleteTaskPushNotificationConfig",
            "GetExtendedAgentCard",
        ]
        client = TestClient(app_with_middleware)
        for method in v1_methods:
            resp = client.post("/", json={"method": method})
            assert "a2a-version=1.0" in resp.text, f"Failed for method {method}"


class TestA2AAuthMiddlewareContextReset:
    """Tests for A2A auth middleware context cleanup."""

    @pytest.fixture
    def app_with_auth(self):
        """Create test app with auth middleware."""
        cfg = _settings(A2A_AUTH_REQUIRED=False)

        async def check_context(request: Request) -> PlainTextResponse:
            ctx = a2a_request_ctx.get()
            return PlainTextResponse(
                f"corr={ctx.correlation_id} agent={ctx.calling_agent_id}"
            )

        app = Starlette(routes=[Route("/", check_context, methods=["POST"])])
        app.add_middleware(A2AAuthMiddleware, cfg=cfg)
        return app

    def test_context_reset_after_request(self, app_with_auth):
        """Context is reset after request completes."""
        client = TestClient(app_with_auth)

        resp1 = client.post(
            "/",
            headers={
                "X-Correlation-ID": "corr-1",
                "X-Calling-Agent-ID": "agent-1",
            },
        )
        assert "corr=corr-1" in resp1.text
        assert "agent=agent-1" in resp1.text

        resp2 = client.post(
            "/",
            headers={
                "X-Correlation-ID": "corr-2",
                "X-Calling-Agent-ID": "agent-2",
            },
        )
        assert "corr=corr-2" in resp2.text
        assert "agent=agent-2" in resp2.text


class TestA2AAuthMiddlewareInvalidToken:
    """Tests for A2A auth middleware with invalid tokens."""

    def test_rejects_invalid_jwt_when_auth_required(self):
        """Invalid JWT is rejected when auth is required and secret configured."""
        cfg = _settings(A2A_AUTH_REQUIRED=True, A2A_JWT_SECRET=JWT_SECRET)

        async def handler(request: Request) -> PlainTextResponse:
            return PlainTextResponse("ok")

        app = Starlette(routes=[Route("/", handler, methods=["POST"])])
        app.add_middleware(A2AAuthMiddleware, cfg=cfg)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/", headers={"Authorization": "Bearer invalid.jwt.token"})
        assert resp.status_code == 401
        assert "invalid_token" in resp.headers.get("www-authenticate", "")
