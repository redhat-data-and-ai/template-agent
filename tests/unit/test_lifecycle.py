"""Unit tests for lifecycle state persistence module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.aegra.lifecycle import (
    POD_ID,
    RunExecutionContext,
    _derive_fernet_key,
    build_execution_context,
    decrypt_token,
    encrypt_token,
    persist_inflight_runs,
    resume_interrupted_runs,
)

# ── TestLifecycleSettings ────────────────────────────────────────


class TestLifecycleSettings:
    """Verify that lifecycle settings exist and have correct defaults."""

    def test_default_values(self):
        from deep_agent.src.settings import Settings

        s = Settings()
        assert s.LIFECYCLE_PERSISTENCE_ENABLED is True
        assert s.LIFECYCLE_LEASE_SECONDS == 300
        assert s.LIFECYCLE_MAX_RESUME_BATCH == 10
        assert s.LIFECYCLE_RESUME_ON_STARTUP is True

    def test_override_via_constructor(self):
        from deep_agent.src.settings import Settings

        s = Settings(
            LIFECYCLE_PERSISTENCE_ENABLED=False,
            LIFECYCLE_LEASE_SECONDS=60,
            LIFECYCLE_MAX_RESUME_BATCH=5,
            LIFECYCLE_RESUME_ON_STARTUP=False,
        )
        assert s.LIFECYCLE_PERSISTENCE_ENABLED is False
        assert s.LIFECYCLE_LEASE_SECONDS == 60
        assert s.LIFECYCLE_MAX_RESUME_BATCH == 5
        assert s.LIFECYCLE_RESUME_ON_STARTUP is False


# ── TestTokenEncryption ──────────────────────────────────────────


class TestTokenEncryption:
    """Verify Fernet-based token encryption round-trip."""

    def test_round_trip(self):
        secret = "test-secret-key-for-lifecycle"
        token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.test"
        encrypted = encrypt_token(token, secret)
        decrypted = decrypt_token(encrypted, secret)
        assert decrypted == token

    def test_different_secrets_fail(self):
        from cryptography.fernet import InvalidToken

        token = "my-secret-token"
        encrypted = encrypt_token(token, "secret-a")
        with pytest.raises(InvalidToken):
            decrypt_token(encrypted, "secret-b")

    def test_derive_key_deterministic(self):
        key1 = _derive_fernet_key("same-secret")
        key2 = _derive_fernet_key("same-secret")
        assert key1 == key2

    def test_derive_key_different_for_different_secrets(self):
        key1 = _derive_fernet_key("secret-1")
        key2 = _derive_fernet_key("secret-2")
        assert key1 != key2

    def test_encryption_secret_uses_sso_client_secret(self):
        """Verify _get_encryption_secret reads SSO_CLIENT_SECRET from settings."""
        from deep_agent.aegra.lifecycle import _get_encryption_secret

        mock_settings = MagicMock()
        mock_settings.SSO_CLIENT_SECRET = "my-sso-secret"
        with patch("deep_agent.src.settings.settings", mock_settings):
            assert _get_encryption_secret() == "my-sso-secret"


# ── TestRunExecutionContext ──────────────────────────────────────


class TestRunExecutionContext:
    """Verify RunExecutionContext Pydantic model."""

    def test_minimal_construction(self):
        ctx = RunExecutionContext(model_name="gpt-4", config_hash="abc123")
        assert ctx.model_name == "gpt-4"
        assert ctx.config_hash == "abc123"
        assert ctx.assistant_id is None
        assert ctx.user_id is None
        assert ctx.encrypted_refresh_token is None
        assert ctx.mcp_server_names == []
        assert ctx.orchestrator_config_snapshot == {}

    def test_full_construction_and_serialization(self):
        ctx = RunExecutionContext(
            model_name="gemini-pro",
            config_hash="def456",
            assistant_id="asst-001",
            user_id="user-42",
            encrypted_refresh_token="gAAAAA...",
            mcp_server_names=["tools", "search"],
            orchestrator_config_snapshot={"name": "orch", "model": "gemini-pro"},
        )
        data = ctx.model_dump()
        assert data["model_name"] == "gemini-pro"
        assert data["mcp_server_names"] == ["tools", "search"]
        assert data["orchestrator_config_snapshot"]["name"] == "orch"


# ── TestPersistInflightRuns ──────────────────────────────────────


class TestPersistInflightRuns:
    """Verify persist_inflight_runs shutdown behavior."""

    def test_no_inflight_runs(self):
        """No active runs in memory."""
        with patch(
            "deep_agent.aegra.lifecycle._get_active_runs",
            return_value={},
        ):
            count = persist_inflight_runs()
        assert count == 0

    def test_uses_active_runs_dict(self):
        """persist must read from in-memory active_runs dict and expire leases."""
        with (
            patch(
                "deep_agent.aegra.lifecycle._get_active_runs",
                return_value={"run-1": MagicMock(), "run-2": MagicMock()},
            ),
            patch(
                "deep_agent.aegra.lifecycle._expire_run_leases_batch",
                return_value=2,
            ) as mock_expire,
        ):
            count = persist_inflight_runs()

        assert count == 2
        mock_expire.assert_called_once()
        call_args = mock_expire.call_args[0][0]
        assert sorted(call_args) == ["run-1", "run-2"]

    def test_empty_active_runs_returns_zero(self):
        """When active_runs dict is empty, returns 0."""
        with patch(
            "deep_agent.aegra.lifecycle._get_active_runs",
            return_value={},
        ):
            count = persist_inflight_runs()

        assert count == 0

    def test_partial_failure(self):
        with (
            patch(
                "deep_agent.aegra.lifecycle._get_active_runs",
                return_value={"run-1": MagicMock(), "run-2": MagicMock()},
            ),
            patch(
                "deep_agent.aegra.lifecycle._expire_run_leases_batch",
                return_value=1,
            ),
        ):
            count = persist_inflight_runs()

        assert count == 1


# ── TestUpdateRunStatusPostgres ──────────────────────────────────


class TestUpdateRunStatusPostgres:
    """Verify _update_run_status_postgres uses claimed_by."""

    def test_with_claimed_by(self):
        from deep_agent.aegra.lifecycle import _update_run_status_postgres

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect", return_value=mock_conn):
            result = _update_run_status_postgres(
                "run-1", "interrupted", claimed_by="pod-abc"
            )

        assert result is True
        sql_call = mock_cursor.execute.call_args
        sql_str = sql_call[0][0]
        assert "claimed_by" in sql_str
        params = sql_call[0][1]
        assert "pod-abc" in params

    def test_without_claimed_by(self):
        from deep_agent.aegra.lifecycle import _update_run_status_postgres

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect", return_value=mock_conn):
            result = _update_run_status_postgres("run-1", "interrupted")

        assert result is True
        sql_call = mock_cursor.execute.call_args
        sql_str = sql_call[0][0]
        assert "claimed_by" not in sql_str


# ── TestResumeInterruptedRuns ────────────────────────────────────


class TestResumeInterruptedRuns:
    """Verify resume_interrupted_runs startup behavior."""

    @staticmethod
    def _make_async_conn_mock(cursor_mock):
        """Build an async connection mock matching psycopg.AsyncConnection."""
        cursor_ctx = MagicMock()
        cursor_ctx.__aenter__ = AsyncMock(return_value=cursor_mock)
        cursor_ctx.__aexit__ = AsyncMock(return_value=False)

        conn = MagicMock()
        conn.cursor.return_value = cursor_ctx
        conn.commit = AsyncMock()
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        return conn

    async def test_no_interrupted_runs(self):
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])
        mock_conn = self._make_async_conn_mock(mock_cursor)

        with patch("psycopg.AsyncConnection.connect", return_value=mock_conn):
            resume_fn = AsyncMock()
            results = await resume_interrupted_runs("postgresql://test", resume_fn)

        assert results == {}
        resume_fn.assert_not_awaited()

    async def test_query_includes_status_and_lease(self):
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])
        mock_conn = self._make_async_conn_mock(mock_cursor)

        with patch("psycopg.AsyncConnection.connect", return_value=mock_conn):
            await resume_interrupted_runs("postgresql://test", AsyncMock())

        sql_call = mock_cursor.execute.call_args_list[0]
        sql_str = sql_call[0][0]
        assert "running" in sql_str
        assert "interrupted" in sql_str
        assert "lease_expires_at" in sql_str
        assert "FOR UPDATE SKIP LOCKED" in sql_str

    async def test_resumes_with_lease_claim(self):
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                ("run-1", "thread-1", "cp-1"),
            ]
        )
        mock_conn = self._make_async_conn_mock(mock_cursor)

        resume_fn = AsyncMock()

        with patch("psycopg.AsyncConnection.connect", return_value=mock_conn):
            results = await resume_interrupted_runs("postgresql://test", resume_fn)

        assert results["run-1"] == "resumed"

        execute_calls = mock_cursor.execute.call_args_list
        claim_call = execute_calls[1]
        claim_sql = claim_call[0][0]
        assert "claimed_by" in claim_sql
        assert "lease_expires_at" in claim_sql
        claim_params = claim_call[0][1]
        assert POD_ID in claim_params

    async def test_resumes_successfully(self):
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                ("run-1", "thread-1", "cp-1"),
                ("run-2", "thread-2", "cp-2"),
            ]
        )
        mock_conn = self._make_async_conn_mock(mock_cursor)

        resume_fn = AsyncMock()

        with patch("psycopg.AsyncConnection.connect", return_value=mock_conn):
            results = await resume_interrupted_runs("postgresql://test", resume_fn)

        assert results["run-1"] == "resumed"
        assert results["run-2"] == "resumed"
        assert resume_fn.await_count == 2

    async def test_handles_resume_failure_sets_error_status(self):
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[("run-1", "thread-1", "cp-1")])
        mock_conn = self._make_async_conn_mock(mock_cursor)

        resume_fn = AsyncMock(side_effect=RuntimeError("resume failed"))

        with patch("psycopg.AsyncConnection.connect", return_value=mock_conn):
            results = await resume_interrupted_runs("postgresql://test", resume_fn)

        assert "error" in results["run-1"]

        execute_calls = mock_cursor.execute.call_args_list
        error_call = execute_calls[-1]
        error_sql = error_call[0][0]
        assert "error" in error_sql


# ── TestBuildExecutionContext ────────────────────────────────────


class TestBuildExecutionContext:
    """Verify build_execution_context helper."""

    def test_basic_context(self):
        ctx = build_execution_context(
            model_name="gemini-pro",
            config_hash="abc123",
            assistant_id="asst-1",
            user_id="user-1",
        )
        assert ctx["model_name"] == "gemini-pro"
        assert ctx["config_hash"] == "abc123"
        assert ctx["assistant_id"] == "asst-1"
        assert ctx["user_id"] == "user-1"
        assert ctx["encrypted_refresh_token"] is None
        assert ctx["mcp_server_names"] == []

    def test_with_encrypted_token(self):
        ctx = build_execution_context(
            model_name="gpt-4",
            config_hash="def456",
            refresh_token="my-refresh-token",
            encryption_secret="test-secret",
            mcp_server_names=["tools", "search"],
            orchestrator_config={"name": "orch"},
        )
        assert ctx["model_name"] == "gpt-4"
        assert ctx["encrypted_refresh_token"] is not None
        assert ctx["encrypted_refresh_token"] != "my-refresh-token"
        assert ctx["mcp_server_names"] == ["tools", "search"]
        assert ctx["orchestrator_config_snapshot"]["name"] == "orch"

        decrypted = decrypt_token(ctx["encrypted_refresh_token"], "test-secret")
        assert decrypted == "my-refresh-token"

    def test_uses_sso_client_secret_by_default(self):
        with patch(
            "deep_agent.aegra.lifecycle._get_encryption_secret",
            return_value="sso-secret-from-settings",
        ):
            ctx = build_execution_context(
                model_name="gpt-4",
                config_hash="abc",
                refresh_token="my-token",
            )

        assert ctx["encrypted_refresh_token"] is not None
        decrypted = decrypt_token(
            ctx["encrypted_refresh_token"], "sso-secret-from-settings"
        )
        assert decrypted == "my-token"
