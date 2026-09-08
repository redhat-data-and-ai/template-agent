"""Tests for mcp_token_store — client_secret redaction and full store coverage."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from deep_agent.aegra.mcp_crypto import reset_mcp_crypto_cache
import deep_agent.aegra.mcp_token_store as _mod
from deep_agent.aegra.mcp_token_store import (
    McpOAuthClient,
    McpOAuthToken,
    McpTokenStore,
    _strip_client_secret,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    reset_mcp_crypto_cache()
    _mod._TABLES_ENSURED = False
    yield
    reset_mcp_crypto_cache()
    _mod._TABLES_ENSURED = False


@pytest.fixture
def fernet_key(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("MCP_TOKEN_ENCRYPTION_KEY", key)
    return key


@pytest.fixture
def store():
    return McpTokenStore("postgresql://unused")


class TestStripClientSecret:
    """Pure-function tests for _strip_client_secret."""

    def test_removes_client_secret_key(self):
        data = {"client_id": "cid", "client_secret": "super-secret", "extra": 1}
        result = _strip_client_secret(data)
        assert "client_secret" not in result
        assert result["client_id"] == "cid"
        assert result["extra"] == 1

    def test_does_not_mutate_original(self):
        data = {"client_secret": "original", "client_id": "cid"}
        _strip_client_secret(data)
        assert data["client_secret"] == "original"

    def test_returns_none_for_none(self):
        assert _strip_client_secret(None) is None

    def test_returns_same_dict_when_no_secret(self):
        data = {"client_id": "cid"}
        result = _strip_client_secret(data)
        assert result is data  # same object, no copy needed

    def test_removes_empty_string_secret(self):
        """Even an empty client_secret should be stripped."""
        data = {"client_id": "cid", "client_secret": "", "extra": 1}
        result = _strip_client_secret(data)
        assert "client_secret" not in result

    def test_removes_none_valued_secret(self):
        """client_secret key present but None-valued should still be stripped."""
        data = {"client_id": "cid", "client_secret": None}
        result = _strip_client_secret(data)
        assert "client_secret" not in result

    def test_handles_empty_dict(self):
        data = {}
        result = _strip_client_secret(data)
        assert result is data  # no copy needed


@pytest.mark.asyncio
class TestUpsertClientRedaction:
    """Integration-style: upsert_client strips client_secret from registration_data (AC4)."""

    async def test_upsert_client_strips_secret_from_registration_data(
        self, store, fernet_key
    ):
        executed_params: list[tuple] = []

        mock_conn = AsyncMock()

        async def capture_execute(sql, params=None):
            if params:
                executed_params.append(params)
            return AsyncMock()

        mock_conn.execute = capture_execute
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(store, "ensure_tables"),
            patch("deep_agent.aegra.mcp_token_store.psycopg") as mock_psycopg,
        ):
            mock_psycopg.AsyncConnection.connect = AsyncMock(return_value=mock_conn)

            dcr_response = {
                "client_id": "cid-123",
                "client_secret": "plaintext-secret",
                "client_name": "my-agent",
                "redirect_uris": ["http://localhost/callback"],
            }
            result = await store.upsert_client(
                agent_name="default",
                mcp_name="dcr-mcp",
                client_id="cid-123",
                client_secret="plaintext-secret",
                registration_data=dcr_response,
            )

        # The Jsonb-wrapped param is the 5th positional arg
        assert len(executed_params) == 1
        jsonb_param = executed_params[0][4]
        persisted_data = jsonb_param.obj  # psycopg Jsonb wraps .obj
        assert "client_secret" not in persisted_data
        assert "plaintext-secret" not in str(persisted_data)

        # Return value also reflects stripping
        assert "client_secret" not in result.registration_data

    @pytest.mark.parametrize(
        "mcp_name,dcr_response",
        [
            (
                "jira-mcp",
                {
                    "client_id": "jira-cid",
                    "client_secret": "jira-secret",
                    "client_name": "Agent-jira",
                    "redirect_uris": ["http://localhost/callback"],
                    "grant_types": ["authorization_code", "refresh_token"],
                    "token_endpoint_auth_method": "client_secret_basic",
                },
            ),
            (
                "gitlab-mcp",
                {
                    "client_id": "gl-cid",
                    "client_secret": "gl-secret",
                    "application_type": "web",
                },
            ),
            (
                "minimal-mcp",
                {"client_id": "min-cid", "client_secret": "s"},
            ),
        ],
        ids=["jira-full", "gitlab-style", "minimal"],
    )
    async def test_upsert_strips_secret_any_provider(
        self, store, fernet_key, mcp_name, dcr_response
    ):
        """client_secret is stripped regardless of MCP provider response shape."""
        mock_conn = AsyncMock()
        executed_params: list[tuple] = []

        async def capture_execute(sql, params=None):
            if params:
                executed_params.append(params)
            return AsyncMock()

        mock_conn.execute = capture_execute
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(store, "ensure_tables"),
            patch("deep_agent.aegra.mcp_token_store.psycopg") as mock_psycopg,
        ):
            mock_psycopg.AsyncConnection.connect = AsyncMock(return_value=mock_conn)
            result = await store.upsert_client(
                agent_name="default",
                mcp_name=mcp_name,
                client_id=dcr_response["client_id"],
                client_secret=dcr_response["client_secret"],
                registration_data=dcr_response,
            )

        persisted_data = executed_params[0][4].obj
        assert "client_secret" not in persisted_data
        assert "client_secret" not in result.registration_data

    async def test_upsert_client_no_registration_data(self, store, fernet_key):
        """upsert_client works fine when registration_data is None."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(store, "ensure_tables"),
            patch("deep_agent.aegra.mcp_token_store.psycopg") as mock_psycopg,
        ):
            mock_psycopg.AsyncConnection.connect = AsyncMock(return_value=mock_conn)
            result = await store.upsert_client(
                agent_name="default",
                mcp_name="dcr-mcp",
                client_id="cid-123",
                client_secret="secret",
                registration_data=None,
            )

        assert result.registration_data is None


class TestTokenKey:
    """Tests for McpTokenStore._token_key static method."""

    def test_builds_expected_key(self):
        key = McpTokenStore._token_key("agent-1", "user-42", "jira-mcp")
        assert key == "mcp_oauth_token:agent-1:user-42:jira-mcp"

    def test_handles_special_chars(self):
        key = McpTokenStore._token_key("a:b", "u/v", "m@x")
        assert key == "mcp_oauth_token:a:b:u/v:m@x"


class TestSerializeDatetime:
    """Tests for _serialize_datetime / _deserialize_datetime round-trip."""

    def test_serialize_none_returns_none(self):
        assert McpTokenStore._serialize_datetime(None) is None

    def test_serialize_aware_datetime(self):
        dt = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)
        result = McpTokenStore._serialize_datetime(dt)
        assert "2026-09-08" in result
        assert result.endswith("+00:00")

    def test_serialize_naive_datetime_gets_utc(self):
        dt = datetime(2026, 1, 1, 0, 0, 0)
        result = McpTokenStore._serialize_datetime(dt)
        assert result.endswith("+00:00")

    def test_deserialize_none_returns_none(self):
        assert McpTokenStore._deserialize_datetime(None) is None

    def test_deserialize_empty_string_returns_none(self):
        assert McpTokenStore._deserialize_datetime("") is None

    def test_deserialize_aware_iso(self):
        result = McpTokenStore._deserialize_datetime("2026-09-08T12:00:00+00:00")
        assert result.year == 2026
        assert result.tzinfo is not None

    def test_deserialize_naive_iso_gets_utc(self):
        result = McpTokenStore._deserialize_datetime("2026-09-08T12:00:00")
        assert result.tzinfo == UTC

    def test_round_trip(self):
        dt = datetime(2026, 6, 15, 8, 30, 45, tzinfo=UTC)
        serialized = McpTokenStore._serialize_datetime(dt)
        deserialized = McpTokenStore._deserialize_datetime(serialized)
        assert deserialized == dt


class TestTokenPayloadConversion:
    """Tests for _token_to_payload and _payload_to_token."""

    def test_token_to_payload_encrypts_secrets(self, store, fernet_key):
        payload = store._token_to_payload(
            access_token="access-123",
            refresh_token="refresh-456",
            expires_at=datetime(2026, 12, 31, tzinfo=UTC),
            scopes=["read", "write"],
        )
        assert payload["access_token"] != "access-123"
        assert payload["refresh_token"] != "refresh-456"
        assert payload["scopes"] == ["read", "write"]
        assert payload["expires_at"] is not None
        assert payload["updated_at"] is not None

    def test_token_to_payload_none_refresh(self, store, fernet_key):
        payload = store._token_to_payload(
            access_token="access-123",
            refresh_token=None,
            expires_at=None,
            scopes=None,
        )
        assert payload["refresh_token"] is None
        assert payload["expires_at"] is None
        assert payload["scopes"] is None

    def test_payload_to_token_decrypts(self, store, fernet_key):
        payload = store._token_to_payload(
            access_token="my-access",
            refresh_token="my-refresh",
            expires_at=datetime(2026, 12, 31, tzinfo=UTC),
            scopes=["read"],
        )
        token = store._payload_to_token("agent", "user1", "mcp-x", payload)
        assert isinstance(token, McpOAuthToken)
        assert token.access_token == "my-access"
        assert token.refresh_token == "my-refresh"
        assert token.scopes == ["read"]
        assert token.agent_name == "agent"
        assert token.user_id == "user1"
        assert token.mcp_name == "mcp-x"

    def test_payload_to_token_returns_none_on_bad_decrypt(self, store, fernet_key):
        payload = {
            "access_token": "not-a-fernet-token",
            "refresh_token": None,
            "expires_at": None,
            "scopes": None,
            "updated_at": None,
        }
        result = store._payload_to_token("a", "u", "m", payload)
        assert result is None


class TestExpiresAtFromTokenResponse:
    """Tests for McpTokenStore.expires_at_from_token_response."""

    def test_returns_future_datetime(self):
        before = datetime.now(UTC)
        result = McpTokenStore.expires_at_from_token_response({"expires_in": 3600})
        after = datetime.now(UTC)
        assert result is not None
        assert (
            before + timedelta(seconds=3599) < result < after + timedelta(seconds=3601)
        )

    def test_returns_none_when_missing(self):
        assert McpTokenStore.expires_at_from_token_response({}) is None

    def test_returns_none_for_invalid_value(self):
        assert (
            McpTokenStore.expires_at_from_token_response({"expires_in": "bad"}) is None
        )

    def test_handles_string_number(self):
        result = McpTokenStore.expires_at_from_token_response({"expires_in": "60"})
        assert result is not None


@pytest.mark.asyncio
class TestEnsureTables:
    """Tests for ensure_tables — migration + create SQL."""

    async def test_ensure_tables_executes_migration_sql(self, store):
        from deep_agent.aegra.mcp_token_store import (
            CREATE_OAUTH_CLIENTS_TABLE,
            MIGRATE_OAUTH_TABLES,
        )
        import deep_agent.aegra.mcp_token_store as mod

        mod._TABLES_ENSURED = False

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with patch("deep_agent.aegra.mcp_token_store.psycopg") as mock_psycopg:
            mock_psycopg.AsyncConnection.connect = AsyncMock(return_value=mock_conn)
            await store.ensure_tables()

        calls = [c.args[0] for c in mock_conn.execute.await_args_list]
        assert calls == [
            MIGRATE_OAUTH_TABLES,
            CREATE_OAUTH_CLIENTS_TABLE,
        ]
        assert mock_conn.commit.await_count == 1

    async def test_ensure_tables_marks_as_ensured(self, store):
        import deep_agent.aegra.mcp_token_store as mod

        mod._TABLES_ENSURED = False

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with patch("deep_agent.aegra.mcp_token_store.psycopg") as mock_psycopg:
            mock_psycopg.AsyncConnection.connect = AsyncMock(return_value=mock_conn)
            await store.ensure_tables()
            assert mod._TABLES_ENSURED is True

    async def test_logs_only_once(self, store):
        """Second call still runs SQL but does not log again."""
        import deep_agent.aegra.mcp_token_store as mod

        mod._TABLES_ENSURED = False

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with patch("deep_agent.aegra.mcp_token_store.psycopg") as mock_psycopg:
            mock_psycopg.AsyncConnection.connect = AsyncMock(return_value=mock_conn)
            await store.ensure_tables()
            assert mod._TABLES_ENSURED is True
            await store.ensure_tables()
            assert mod._TABLES_ENSURED is True


@pytest.mark.asyncio
class TestGetClient:
    """Tests for get_client — Postgres read path."""

    async def test_returns_none_when_not_found(self, store, fernet_key):
        mock_cur = AsyncMock()
        mock_cur.fetchone = AsyncMock(return_value=None)

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_cur)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(store, "ensure_tables"),
            patch("deep_agent.aegra.mcp_token_store.psycopg") as mock_psycopg,
        ):
            mock_psycopg.AsyncConnection.connect = AsyncMock(return_value=mock_conn)
            result = await store.get_client("agent", "mcp-x")

        assert result is None

    async def test_returns_client_when_found(self, store, fernet_key):
        from deep_agent.aegra.mcp_crypto import encrypt_secret

        enc = encrypt_secret("the-secret")
        row = {
            "agent_name": "agent",
            "mcp_name": "mcp-x",
            "client_id": "cid",
            "client_secret": enc,
            "registration_data": {"client_id": "cid"},
            "updated_at": datetime.now(UTC),
        }
        mock_cur = AsyncMock()
        mock_cur.fetchone = AsyncMock(return_value=row)

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_cur)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(store, "ensure_tables"),
            patch("deep_agent.aegra.mcp_token_store.psycopg") as mock_psycopg,
        ):
            mock_psycopg.AsyncConnection.connect = AsyncMock(return_value=mock_conn)
            result = await store.get_client("agent", "mcp-x")

        assert isinstance(result, McpOAuthClient)
        assert result.client_id == "cid"
        assert result.client_secret == "the-secret"

    async def test_returns_none_on_decrypt_failure(self, store, fernet_key):
        row = {
            "agent_name": "agent",
            "mcp_name": "mcp-x",
            "client_id": "cid",
            "client_secret": "corrupted-not-fernet",
            "registration_data": None,
            "updated_at": datetime.now(UTC),
        }
        mock_cur = AsyncMock()
        mock_cur.fetchone = AsyncMock(return_value=row)

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_cur)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(store, "ensure_tables"),
            patch("deep_agent.aegra.mcp_token_store.psycopg") as mock_psycopg,
        ):
            mock_psycopg.AsyncConnection.connect = AsyncMock(return_value=mock_conn)
            result = await store.get_client("agent", "mcp-x")

        assert result is None


@pytest.mark.asyncio
class TestGetToken:
    """Tests for get_token — Redis read path."""

    async def test_returns_none_when_no_redis_entry(self, store, fernet_key):
        with patch("deep_agent.aegra.mcp_token_store.cache_get", return_value=None):
            result = await store.get_token("agent", "user1", "mcp-x")
        assert result is None

    async def test_returns_none_on_corrupt_json(self, store, fernet_key):
        with patch(
            "deep_agent.aegra.mcp_token_store.cache_get", return_value="not-json{"
        ):
            result = await store.get_token("agent", "user1", "mcp-x")
        assert result is None

    async def test_returns_none_on_non_dict_payload(self, store, fernet_key):
        with patch(
            "deep_agent.aegra.mcp_token_store.cache_get", return_value='"just a string"'
        ):
            result = await store.get_token("agent", "user1", "mcp-x")
        assert result is None

    async def test_returns_token_on_valid_payload(self, store, fernet_key):
        payload = store._token_to_payload("access", "refresh", None, ["read"])
        raw = json.dumps(payload)
        with patch("deep_agent.aegra.mcp_token_store.cache_get", return_value=raw):
            result = await store.get_token("agent", "user1", "mcp-x")
        assert isinstance(result, McpOAuthToken)
        assert result.access_token == "access"
        assert result.refresh_token == "refresh"


@pytest.mark.asyncio
class TestUpsertToken:
    """Tests for upsert_token — Redis write path."""

    async def test_upsert_token_stores_and_returns(self, store, fernet_key):
        with patch(
            "deep_agent.aegra.mcp_token_store.cache_set_persistent", return_value=True
        ):
            result = await store.upsert_token(
                agent_name="agent",
                user_id="user1",
                mcp_name="mcp-x",
                access_token="access-tok",
                refresh_token="refresh-tok",
                expires_at=datetime(2026, 12, 31, tzinfo=UTC),
                scopes=["read"],
            )
        assert isinstance(result, McpOAuthToken)
        assert result.access_token == "access-tok"
        assert result.refresh_token == "refresh-tok"
        assert result.scopes == ["read"]

    async def test_upsert_token_raises_on_redis_failure(self, store, fernet_key):
        with (
            patch(
                "deep_agent.aegra.mcp_token_store.cache_set_persistent",
                return_value=False,
            ),
            pytest.raises(RuntimeError, match="Failed to persist"),
        ):
            await store.upsert_token(
                agent_name="agent",
                user_id="user1",
                mcp_name="mcp-x",
                access_token="access-tok",
            )


@pytest.mark.asyncio
class TestDeleteToken:
    """Tests for delete_token — Redis delete path."""

    async def test_delete_returns_true(self, store):
        with patch("deep_agent.aegra.mcp_token_store.cache_delete", return_value=True):
            result = await store.delete_token("agent", "user1", "mcp-x")
        assert result is True

    async def test_delete_returns_false(self, store):
        with patch("deep_agent.aegra.mcp_token_store.cache_delete", return_value=False):
            result = await store.delete_token("agent", "user1", "mcp-x")
        assert result is False
