"""The channel endpoint, end to end bar the actual run.

These assertions are about the order of checks as much as the outcomes:
verify before parse, reject unmapped senders, ack before working.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_agent.aegra.channel_routes import router
from deep_agent.src.channels.config import (
    ChannelConfig,
    ChannelsFileConfig,
    IdentityConfig,
)
from deep_agent.src.channels.signatures import sign_webhook

SECRET = "s3cr3t"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SIGNING_SECRET", SECRET)

    config = ChannelsFileConfig(
        enabled=True,
        channels={
            "webhook": ChannelConfig(
                type="webhook",
                signing_secret_env="WEBHOOK_SIGNING_SECRET",
                identity=IdentityConfig(map={"ci": "automation@example.com"}),
            )
        },
    )
    with patch("deep_agent.src.channels.loader.load_channels", return_value=config):
        app = FastAPI()
        app.include_router(router)
        yield TestClient(app)


def post(client, payload: dict, *, secret: str = SECRET, timestamp: str | None = None):
    import time

    body = json.dumps(payload).encode()
    ts = timestamp or str(int(time.time()))
    return client.post(
        "/channels/webhook",
        content=body,
        headers={
            "X-Timestamp": ts,
            "X-Signature": sign_webhook(body=body, timestamp=ts, secret=secret),
            "Content-Type": "application/json",
        },
    )


class TestSignatureGate:
    def test_a_signed_request_is_accepted(self, client):
        with patch("deep_agent.aegra.channel_routes._run_and_deliver", new=AsyncMock()):
            response = post(client, {"sender": "ci", "text": "hello"})
        assert response.status_code == 202

    def test_an_unsigned_request_is_rejected(self, client):
        response = client.post("/channels/webhook", json={"sender": "ci", "text": "x"})
        assert response.status_code == 401

    def test_a_wrongly_signed_request_is_rejected(self, client):
        response = post(client, {"sender": "ci", "text": "x"}, secret="wrong")
        assert response.status_code == 401

    def test_a_stale_request_is_rejected(self, client):
        response = post(client, {"sender": "ci", "text": "x"}, timestamp="1")
        assert response.status_code == 401

    def test_the_rejection_reason_is_not_disclosed(self, client):
        """Stale and mismatched must look identical to a caller.

        The difference is a tuning signal for anyone probing the
        endpoint.
        """
        stale = post(client, {"sender": "ci", "text": "x"}, timestamp="1")
        wrong = post(client, {"sender": "ci", "text": "x"}, secret="wrong")
        assert stale.json() == wrong.json()

    def test_a_bad_body_is_rejected_before_it_is_parsed(self, client):
        """An unsigned malformed body must 401, not 400.

        A 400 would prove the handler parsed something it had not
        authenticated.
        """
        response = client.post(
            "/channels/webhook",
            content=b"not json at all",
            headers={"X-Timestamp": "1", "X-Signature": "deadbeef"},
        )
        assert response.status_code == 401


class TestIdentity:
    def test_an_unmapped_sender_is_refused(self, client):
        response = post(client, {"sender": "stranger", "text": "hello"})
        assert response.status_code == 403

    def test_a_mapped_sender_runs_as_their_platform_user(self, client):
        seen = {}

        async def capture(message, channel, user_id):
            seen["user_id"] = user_id

        with patch("deep_agent.aegra.channel_routes._run_and_deliver", new=capture):
            post(client, {"sender": "ci", "text": "hello"})

        assert seen["user_id"] == "automation@example.com"


class TestUnknownChannels:
    def test_an_unconfigured_channel_is_404(self, client):
        response = client.post("/channels/nope", json={})
        assert response.status_code == 404


class TestListing:
    def test_listing_exposes_no_secrets_or_identities(self, client):
        body = client.get("/channels/").json()
        rendered = json.dumps(body)
        assert "webhook" in rendered
        assert SECRET not in rendered
        assert "automation@example.com" not in rendered


class TestDisabledByDefault:
    def test_nothing_is_exposed_until_a_channel_is_configured(self):
        """The shipped config has `enabled: false`, and that must mean 404."""
        with patch(
            "deep_agent.src.channels.loader.load_channels",
            return_value=ChannelsFileConfig(),
        ):
            app = FastAPI()
            app.include_router(router)
            assert TestClient(app).post("/channels/webhook", json={}).status_code == 404
