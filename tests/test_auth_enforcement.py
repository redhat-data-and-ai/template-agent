"""Tests for A2A auth middleware enforcement (Phase 5)."""

from __future__ import annotations

import time

import jwt
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from template_agent.src.a2a.context import a2a_request_ctx
from template_agent.src.a2a.middleware import A2AAuthMiddleware
from template_agent.src.settings import Settings

JWT_SECRET = "test-secret-key-for-hs256"


def _settings(**overrides) -> Settings:
    defaults = {
        "USE_INMEMORY_SAVER": True,
        "A2A_ENABLED": True,
        "A2A_AUTH_REQUIRED": True,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_token(secret: str = JWT_SECRET, **extra_claims) -> str:
    payload = {
        "sub": "test-agent",
        "iat": int(time.time()),
        "exp": int(time.time()) + 300,
    }
    payload.update(extra_claims)
    return jwt.encode(payload, secret, algorithm="HS256")


async def _echo(request: Request) -> PlainTextResponse:
    """Handler that echoes back the ContextVar contents."""
    ctx = a2a_request_ctx.get()
    return PlainTextResponse(
        f"token={ctx.access_token or 'none'}"
        f" agent={ctx.calling_agent_id or 'none'}"
        f" corr={ctx.correlation_id or 'none'}"
    )


async def _card(request: Request) -> PlainTextResponse:
    return PlainTextResponse("card")


def _build_app(cfg: Settings) -> Starlette:
    app = Starlette(
        routes=[
            Route("/", _echo, methods=["POST"]),
            Route("/.well-known/agent-card.json", _card, methods=["GET"]),
        ],
    )
    app.add_middleware(A2AAuthMiddleware, cfg=cfg)
    return app


class TestAuthEnforcement:
    def test_reject_missing_token(self):
        app = _build_app(_settings(A2A_AUTH_REQUIRED=True))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/")
        assert resp.status_code == 401
        assert "Bearer" in resp.headers.get("www-authenticate", "")

    def test_accept_valid_token_presence_only(self):
        """When no JWT_SECRET/JWKS configured, presence-only check passes."""
        cfg = _settings(A2A_AUTH_REQUIRED=True)
        app = _build_app(cfg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/", headers={"Authorization": "Bearer some-opaque-token"})
        assert resp.status_code == 200
        assert "token=some-opaque-token" in resp.text

    def test_accept_valid_jwt(self):
        cfg = _settings(A2A_AUTH_REQUIRED=True, A2A_JWT_SECRET=JWT_SECRET)
        app = _build_app(cfg)
        client = TestClient(app, raise_server_exceptions=False)
        token = _make_token()
        resp = client.post("/", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_reject_invalid_jwt(self):
        cfg = _settings(A2A_AUTH_REQUIRED=True, A2A_JWT_SECRET=JWT_SECRET)
        app = _build_app(cfg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/", headers={"Authorization": "Bearer bad.jwt.token"})
        assert resp.status_code == 401
        assert "invalid_token" in resp.headers.get("www-authenticate", "")

    def test_reject_expired_jwt(self):
        cfg = _settings(A2A_AUTH_REQUIRED=True, A2A_JWT_SECRET=JWT_SECRET)
        app = _build_app(cfg)
        client = TestClient(app, raise_server_exceptions=False)
        token = _make_token(exp=int(time.time()) - 60)
        resp = client.post("/", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_agent_card_is_unauthenticated(self):
        cfg = _settings(A2A_AUTH_REQUIRED=True, A2A_JWT_SECRET=JWT_SECRET)
        app = _build_app(cfg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        assert resp.text == "card"

    def test_no_auth_required(self):
        cfg = _settings(A2A_AUTH_REQUIRED=False)
        app = _build_app(cfg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/")
        assert resp.status_code == 200

    def test_identity_and_correlation_headers(self):
        cfg = _settings(A2A_AUTH_REQUIRED=False)
        app = _build_app(cfg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/",
            headers={
                "X-Calling-Agent-ID": "upstream-123",
                "X-Correlation-ID": "corr-abc",
            },
        )
        assert resp.status_code == 200
        assert "agent=upstream-123" in resp.text
        assert "corr=corr-abc" in resp.text

    def test_correlation_id_generated_when_missing(self):
        cfg = _settings(A2A_AUTH_REQUIRED=False)
        app = _build_app(cfg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/")
        assert resp.status_code == 200
        # Should have a generated correlation id (UUID format), not "none"
        assert "corr=none" not in resp.text
