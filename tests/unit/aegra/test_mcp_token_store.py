"""Unit tests for MCP OAuth token storage in Redis."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.fernet import Fernet

from deep_agent.aegra.mcp_crypto import reset_mcp_crypto_cache
from deep_agent.aegra.mcp_token_store import McpTokenStore


@pytest.fixture(autouse=True)
def _clear_crypto_cache():
    reset_mcp_crypto_cache()
    yield
    reset_mcp_crypto_cache()


@pytest.fixture
def fernet_key(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("MCP_TOKEN_ENCRYPTION_KEY", key)
    return key


@pytest.fixture
def store():
    return McpTokenStore("postgresql://unused")


@pytest.mark.asyncio
class TestMcpTokenStoreRedis:
    async def test_upsert_and_get_token_round_trip(self, store, fernet_key):
        expires_at = datetime.now(UTC) + timedelta(hours=1)
        stored_payload: dict[str, str] = {}

        def fake_set_persistent(key: str, value: str) -> bool:
            stored_payload["key"] = key
            stored_payload["value"] = value
            return True

        def fake_get(key: str) -> str | None:
            if key == stored_payload.get("key"):
                return stored_payload.get("value")
            return None

        with (
            patch(
                "deep_agent.aegra.mcp_token_store.cache_set_persistent",
                fake_set_persistent,
            ),
            patch("deep_agent.aegra.mcp_token_store.cache_get", fake_get),
        ):
            saved = await store.upsert_token(
                agent_name="default",
                user_id="user-1",
                mcp_name="oauth-mcp",
                access_token="access-secret",
                refresh_token="refresh-secret",
                expires_at=expires_at,
                scopes=["read", "write"],
            )
            loaded = await store.get_token("default", "user-1", "oauth-mcp")

        assert saved.access_token == "access-secret"
        assert saved.refresh_token == "refresh-secret"
        assert saved.scopes == ["read", "write"]
        assert loaded is not None
        assert loaded.access_token == "access-secret"
        assert loaded.refresh_token == "refresh-secret"
        assert loaded.expires_at == expires_at
        assert loaded.scopes == ["read", "write"]

        payload = json.loads(stored_payload["value"])
        assert payload["access_token"] != "access-secret"
        assert payload["refresh_token"] != "refresh-secret"

    async def test_get_token_returns_none_on_miss(self, store):
        with patch("deep_agent.aegra.mcp_token_store.cache_get", return_value=None):
            assert await store.get_token("default", "user-1", "oauth-mcp") is None

    async def test_upsert_token_raises_when_redis_unavailable(self, store, fernet_key):
        with patch(
            "deep_agent.aegra.mcp_token_store.cache_set_persistent", return_value=False
        ):
            with pytest.raises(RuntimeError, match="Failed to persist MCP OAuth token"):
                await store.upsert_token(
                    agent_name="default",
                    user_id="user-1",
                    mcp_name="oauth-mcp",
                    access_token="access-secret",
                )

    async def test_payload_to_token_returns_none_on_decryption_failure(
        self, store, fernet_key
    ):
        """Key rotation: _payload_to_token returns None when decrypt_secret raises."""
        payload = {
            "access_token": "gAAAAABinvalid_ciphertext",
            "refresh_token": "gAAAAABinvalid_ciphertext",
            "expires_at": None,
            "scopes": None,
            "updated_at": None,
        }

        stored_value = json.dumps(payload)

        with patch(
            "deep_agent.aegra.mcp_token_store.cache_get",
            return_value=stored_value,
        ):
            result = await store.get_token("default", "user-1", "oauth-mcp")

        assert result is None

    async def test_get_client_returns_none_on_secret_decryption_failure(
        self, store, fernet_key
    ):
        """Key rotation: get_client returns None when client_secret can't be decrypted."""
        fake_row = {
            "agent_name": "default",
            "mcp_name": "dcr-mcp",
            "client_id": "cid-123",
            "client_secret": "gAAAAABinvalid_ciphertext",
            "registration_data": None,
            "updated_at": None,
        }

        with (
            patch.object(store, "ensure_tables"),
            patch(
                "deep_agent.aegra.mcp_token_store.decrypt_secret",
                side_effect=RuntimeError("MCP token decryption failed"),
            ),
            patch("deep_agent.aegra.mcp_token_store.psycopg") as mock_psycopg,
        ):
            mock_cur = AsyncMock()
            mock_cur.fetchone.return_value = fake_row
            mock_conn = AsyncMock()
            mock_conn.execute.return_value = mock_cur
            mock_psycopg.AsyncConnection.connect = AsyncMock(return_value=mock_conn)
            mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.__aexit__ = AsyncMock(return_value=False)

            result = await store.get_client("default", "dcr-mcp")

        assert result is None

    async def test_payload_to_token_returns_none_on_invalid_fernet_key(
        self, store, fernet_key
    ):
        """Malformed Fernet key: decrypt_secret raises ValueError, not RuntimeError."""
        with (
            patch(
                "deep_agent.aegra.mcp_token_store.cache_get",
                return_value=json.dumps({"access_token": "x"}),
            ),
            patch(
                "deep_agent.aegra.mcp_token_store.decrypt_secret",
                side_effect=ValueError("Fernet key is not valid base64"),
            ),
        ):
            result = await store.get_token("default", "user-1", "oauth-mcp")
        assert result is None

    async def test_get_client_returns_none_on_invalid_fernet_key(
        self, store, fernet_key
    ):
        """Malformed Fernet key: get_client handles ValueError from decrypt_secret."""
        fake_row = {
            "agent_name": "default",
            "mcp_name": "dcr-mcp",
            "client_id": "cid-123",
            "client_secret": "bad-base64",
            "registration_data": None,
            "updated_at": None,
        }
        with (
            patch.object(store, "ensure_tables"),
            patch(
                "deep_agent.aegra.mcp_token_store.decrypt_secret",
                side_effect=ValueError("Fernet key is not valid base64"),
            ),
            patch("deep_agent.aegra.mcp_token_store.psycopg") as mock_psycopg,
        ):
            mock_cur = AsyncMock()
            mock_cur.fetchone.return_value = fake_row
            mock_conn = AsyncMock()
            mock_conn.execute.return_value = mock_cur
            mock_psycopg.AsyncConnection.connect = AsyncMock(return_value=mock_conn)
            mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.__aexit__ = AsyncMock(return_value=False)
            result = await store.get_client("default", "dcr-mcp")
        assert result is None

    async def test_delete_token(self, store):
        deleted_keys: list[str] = []

        def fake_delete(key: str) -> bool:
            deleted_keys.append(key)
            return True

        with patch("deep_agent.aegra.mcp_token_store.cache_delete", fake_delete):
            assert await store.delete_token("default", "user-1", "oauth-mcp") is True

        assert deleted_keys == ["mcp_oauth_token:default:user-1:oauth-mcp"]
