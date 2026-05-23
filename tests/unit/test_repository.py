"""Unit tests for PersonalizationRepository (mocked DB)."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.src.personalization.models import Memory, Rule
from deep_agent.src.personalization.repository import PersonalizationRepository


@pytest.fixture(autouse=True)
def _reset_tables_flag():
    """Reset the module-level _TABLES_ENSURED flag before each test."""
    import deep_agent.src.personalization.repository as repo_mod

    repo_mod._TABLES_ENSURED = False
    yield
    repo_mod._TABLES_ENSURED = False


@pytest.fixture
def mock_conn():
    """Create a mock async connection context manager."""
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.rowcount = 0
    conn.execute = AsyncMock(return_value=cursor)
    conn.commit = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    conn._cursor = cursor
    return conn


@pytest.fixture
def repo():
    return PersonalizationRepository("postgresql://test:test@localhost/testdb")


class TestEnsureTables:
    @pytest.mark.asyncio
    async def test_creates_tables_once(self, repo, mock_conn):
        with patch(
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.ensure_tables()
            assert mock_conn.execute.call_count == 3
            mock_conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_if_already_ensured(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True

        with patch(
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.ensure_tables()
            mock_conn.execute.assert_not_called()


class TestListMemories:
    @pytest.mark.asyncio
    async def test_returns_memories(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True

        mem_data = {
            "id": uuid.uuid4(),
            "user_id": "u1",
            "content": "Likes Python",
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
        }
        mock_conn._cursor.fetchall = AsyncMock(return_value=[mem_data])

        with patch(
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            memories = await repo.list_memories("u1")
            assert len(memories) == 1
            assert memories[0].content == "Likes Python"

    @pytest.mark.asyncio
    async def test_empty_list(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True

        with patch(
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            memories = await repo.list_memories("nobody")
            assert memories == []


class TestCreateMemory:
    @pytest.mark.asyncio
    async def test_creates_and_returns(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True

        with patch(
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            memory = await repo.create_memory("u1", "Likes Python")
            assert memory.user_id == "u1"
            assert memory.content == "Likes Python"
            mock_conn.execute.assert_awaited_once()
            mock_conn.commit.assert_awaited_once()


class TestDeleteMemory:
    @pytest.mark.asyncio
    async def test_delete_returns_true_when_found(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True
        mock_conn._cursor.rowcount = 1
        mock_conn.execute.return_value = mock_conn._cursor

        with patch(
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            result = await repo.delete_memory("u1", uuid.uuid4())
            assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_not_found(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True
        mock_conn._cursor.rowcount = 0
        mock_conn.execute.return_value = mock_conn._cursor

        with patch(
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            result = await repo.delete_memory("u1", uuid.uuid4())
            assert result is False


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
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            rules = await repo.list_rules("u1", active_only=True)
            assert len(rules) == 1
            assert rules[0].content == "Be concise"

    @pytest.mark.asyncio
    async def test_returns_all_rules(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True

        with patch(
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            rules = await repo.list_rules("u1", active_only=False)
            assert rules == []


class TestUpsertRule:
    @pytest.mark.asyncio
    async def test_creates_new_rule(self, repo, mock_conn):
        import deep_agent.src.personalization.repository as repo_mod

        repo_mod._TABLES_ENSURED = True

        with patch(
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
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
            "deep_agent.src.personalization.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            result = await repo.delete_rule("u1", uuid.uuid4())
            assert result is True
