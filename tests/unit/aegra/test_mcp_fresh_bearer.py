"""The access token is attached per request, not per graph build.

Dataverse issues a 300-second access token. Discovery already renews
with a 600-second buffer, so every turn *starts* on a fresh one — but
a turn that runs longer than five minutes still outlives it, and the
model alone stalls 20-60s at a time. Baked into the client headers at
build time, the token goes stale mid-turn and the server answers 401,
which the agent reports as "not connected" for an integration the
user connected four minutes earlier.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from deep_agent.aegra.mcp import (
    _bearer_refresher,
    _build_server_config,
    _FreshBearer,
)

ENTRY = {"url": "https://warehouse.example/mcp", "auth": True}


async def _run(auth: httpx.Auth, header: str | None = None) -> httpx.Request:
    request = httpx.Request("POST", "https://warehouse.example/mcp")
    if header:
        request.headers["Authorization"] = header
    flow = auth.async_auth_flow(request)
    out = await anext(flow)
    await flow.aclose()
    return out


class TestFreshBearer:
    async def test_it_sends_the_current_token_not_the_built_one(self):
        async def resolve():
            return "new-token"

        out = await _run(_FreshBearer(resolve, "stale-token"))
        assert out.headers["Authorization"] == "Bearer new-token"

    async def test_a_failed_refresh_still_sends_what_we_have(self):
        """Seconds may remain on it, and 401 beats an unauthenticated call."""

        async def resolve():
            raise RuntimeError("provider down")

        out = await _run(_FreshBearer(resolve, "stale-token"))
        assert out.headers["Authorization"] == "Bearer stale-token"

    async def test_no_token_anywhere_sends_no_header(self):
        async def resolve():
            return None

        out = await _run(_FreshBearer(resolve, None))
        assert "Authorization" not in out.headers


class TestBearerRefresher:
    def test_per_user_oauth_gets_one(self):
        assert _bearer_refresher("warehouse", ENTRY, "dcr", "user-1") is not None
        assert _bearer_refresher("warehouse", ENTRY, "oauth", "user-1") is not None

    def test_sso_does_not(self):
        """An SSO bearer belongs to the request that carried it."""
        assert _bearer_refresher("warehouse", ENTRY, "sso", "user-1") is None

    def test_an_anonymous_run_does_not(self):
        assert _bearer_refresher("warehouse", ENTRY, "dcr", None) is None


class TestServerConfig:
    def test_a_refreshable_server_gets_a_client_factory(self):
        async def resolve():
            return "t"

        cfg = _build_server_config(ENTRY, "built-token", resolve)
        assert "httpx_client_factory" in cfg
        # The static header stays for transports that take no factory.
        assert cfg["headers"]["Authorization"] == "Bearer built-token"

    def test_without_a_refresher_nothing_changes(self):
        cfg = _build_server_config(ENTRY, "built-token")
        assert "httpx_client_factory" not in cfg
        assert cfg["headers"]["Authorization"] == "Bearer built-token"

    def test_an_unauthenticated_server_gets_no_auth(self):
        async def resolve():
            return "t"

        cfg = _build_server_config({**ENTRY, "auth": False}, None, resolve)
        assert "Authorization" not in cfg["headers"]

    def test_the_sdks_own_auth_is_not_overridden(self):
        async def resolve():
            return "t"

        cfg = _build_server_config(ENTRY, "built-token", resolve)
        theirs = httpx.BasicAuth("u", "p")
        client = cfg["httpx_client_factory"](auth=theirs)
        assert client.auth is theirs

    def test_ours_is_used_when_the_sdk_passes_none(self):
        async def resolve():
            return "t"

        cfg = _build_server_config(ENTRY, "built-token", resolve)
        client = cfg["httpx_client_factory"](auth=None)
        assert isinstance(client.auth, _FreshBearer)

    def test_production_never_gets_verification_turned_off(self):
        """A dev-only `ssl_verify: false` must not reach a cluster."""
        entry = {**ENTRY, "ssl_verify": False}

        with patch("deep_agent.aegra.mcp.mcp_httpx_verify", return_value=False):
            from deep_agent.src import settings as settings_mod

            with patch.object(
                type(settings_mod.settings), "is_production", property(lambda _: True)
            ):
                cfg = _build_server_config(entry, "built-token")

        # verify forced back on, so no insecure client is handed out
        assert "httpx_client_factory" not in cfg


@pytest.mark.parametrize("mode", ["oauth", "dcr"])
async def test_the_refresher_asks_the_resolver(mode):
    seen = {}

    class _Resolver:
        async def resolve(self, user_id, name, entry):
            seen.update(user_id=user_id, name=name)
            return "resolved"

    with patch(
        "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
        return_value=_Resolver(),
    ):
        fn = _bearer_refresher("warehouse", ENTRY, mode, "user-1")
        assert await fn() == "resolved"
    assert seen == {"user_id": "user-1", "name": "warehouse"}
