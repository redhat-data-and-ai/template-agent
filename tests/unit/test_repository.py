"""Unit tests for PersonalizationRepository (mocked DB)."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.src.personalization.models import Rule
from deep_agent.src.personalization.repository import PersonalizationRepository


@pytest.fixture(autouse=True)
def _reset_tables_flag():
    """Reset the module-level _TABLES_ENSURED flag before each test."""
    import deep_agent.src.personalization.repository as repo_mod

    repo_mod._TABLES_ENSURED = False
    yield
    repo_mod._TABLES_ENSURED = False


@pytest.fixture(autouse=True)
def _reset_pool_registry():
    """Clear the pool registry so tests don't share state."""
    import deep_agent.src.personalization.repository as repo_mod

    repo_mod._pool_registry.clear()
    yield
    repo_mod._pool_registry.clear()


@pytest.fixture
def mock_conn():
    """Create a mock async connection context manager."""
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.fetchone = AsyncMock(return_value={"cnt": 0})
    cursor.rowcount = 0
    conn.execute = AsyncMock(return_value=cursor)
    conn.commit = AsyncMock()
    conn.row_factory = None
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    conn._cursor = cursor
    return conn


@pytest.fixture
def mock_pool(mock_conn):
    """Create a mock connection pool that yields mock_conn."""
    pool = AsyncMock()
    pool.connection = MagicMock(return_value=mock_conn)
    pool.open = AsyncMock()
    return pool


@pytest.fixture
def repo(mock_pool):
    r = PersonalizationRepository("postgresql://test:test@localhost/testdb")
    with patch(
        "deep_agent.src.personalization.repository._get_pool",
        return_value=mock_pool,
    ):
        yield r


class TestEnsureTables:
    @pytest.mark.asyncio
    async def test_creates_tables_once(self, repo, mock_conn):
        with patch(
            "deep_agent.src.personalization.repository._get_pool",
            return_value=AsyncMock(connection=MagicMock(return_value=mock_conn)),
        ):
            await repo.ensure_tables()
            assert (
                mock_conn.execute.call_count == 4
            )  # rules + memories + migration + preferences
            mock_conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_if_already_ensured(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True

        with patch(
            "deep_agent.src.personalization.repository._get_pool",
            return_value=AsyncMock(connection=MagicMock(return_value=mock_conn)),
        ):
            await repo.ensure_tables()
            mock_conn.execute.assert_not_called()


class TestListRules:
    @pytest.mark.asyncio
    async def test_returns_rules_active_only(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True

        rule_data = {
            "id": uuid.uuid4(),
            "user_id": "u1",
            "content": "Be concise",
            "is_active": True,
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
        }
        mock_conn._cursor.fetchall = AsyncMock(return_value=[rule_data])

        with patch(
            "deep_agent.src.personalization.repository._get_pool",
            return_value=AsyncMock(connection=MagicMock(return_value=mock_conn)),
        ):
            rules = await repo.list_rules("u1", active_only=True)
            assert len(rules) == 1
            assert rules[0].content == "Be concise"
            sql_arg = mock_conn.execute.call_args[0][0]
            assert "is_active" in sql_arg, "active_only=True must filter by is_active"

    @pytest.mark.asyncio
    async def test_returns_all_rules(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True

        with patch(
            "deep_agent.src.personalization.repository._get_pool",
            return_value=AsyncMock(connection=MagicMock(return_value=mock_conn)),
        ):
            rules = await repo.list_rules("u1", active_only=False)
            assert rules == []
            sql_arg = mock_conn.execute.call_args[0][0]
            assert "is_active" not in sql_arg, (
                "active_only=False must not filter by is_active"
            )


class TestCountRules:
    @pytest.mark.asyncio
    async def test_returns_count(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True
        mock_conn._cursor.fetchone = AsyncMock(return_value={"cnt": 5})

        with patch(
            "deep_agent.src.personalization.repository._get_pool",
            return_value=AsyncMock(connection=MagicMock(return_value=mock_conn)),
        ):
            count = await repo.count_rules("u1")
            assert count == 5


class TestUpsertRule:
    @pytest.mark.asyncio
    async def test_creates_new_rule(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True
        mock_conn._cursor.rowcount = 1

        with patch(
            "deep_agent.src.personalization.repository._get_pool",
            return_value=AsyncMock(connection=MagicMock(return_value=mock_conn)),
        ):
            rule = await repo.upsert_rule("u1", "Be concise")
            assert rule.user_id == "u1"
            assert rule.content == "Be concise"
            assert rule.is_active is True
            mock_conn.commit.assert_awaited_once()


class TestDeleteRule:
    @pytest.mark.asyncio
    async def test_delete_returns_true(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True
        mock_conn._cursor.rowcount = 1
        mock_conn.execute.return_value = mock_conn._cursor

        with patch(
            "deep_agent.src.personalization.repository._get_pool",
            return_value=AsyncMock(connection=MagicMock(return_value=mock_conn)),
        ):
            result = await repo.delete_rule("u1", uuid.uuid4())
            assert result is True


class TestDeleteAllRules:
    @pytest.mark.asyncio
    async def test_delete_all_returns_count(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True
        mock_conn._cursor.rowcount = 3
        mock_conn.execute.return_value = mock_conn._cursor

        with patch(
            "deep_agent.src.personalization.repository._get_pool",
            return_value=AsyncMock(connection=MagicMock(return_value=mock_conn)),
        ):
            result = await repo.delete_all_rules("u1")
            assert result == 3
            sql_arg = mock_conn.execute.call_args[0][0]
            assert "user_id" in sql_arg, "delete_all_rules must filter by user_id"
            params_arg = mock_conn.execute.call_args[0][1]
            assert "u1" in params_arg, "delete_all_rules must pass the user_id param"


class TestUpsertRuleWithGuardian:
    @pytest.mark.asyncio
    async def test_raises_when_guardian_fails(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True

        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = "http://guardian"

        with (
            patch("deep_agent.src.settings.settings", mock_settings),
            patch(
                "deep_agent.src.guardrails.client.check_safety",
                new_callable=AsyncMock,
                return_value=(False, "unsafe"),
            ) as mock_check,
            patch(
                "deep_agent.src.personalization.repository._get_pool",
                return_value=AsyncMock(connection=MagicMock(return_value=mock_conn)),
            ),
        ):
            with pytest.raises(ValueError, match="safety check"):
                await repo.upsert_rule("u1", "bad rule")
            mock_check.assert_awaited_once_with("bad rule", context="rule")
