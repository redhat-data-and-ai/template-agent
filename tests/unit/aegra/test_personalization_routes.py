"""Unit tests for personalization API routes.

Uses FastAPI TestClient with mocked dependencies (repository, store, cache)
so no real Postgres/Redis is needed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_agent.aegra.personalization_routes import (
    MAX_RULES_PER_USER,
    router,
)
from deep_agent.src.personalization.models import Rule, UserPreferences

app = FastAPI()
app.include_router(router)


def _make_rule(
    user_id: str = "test-user",
    content: str = "Be concise",
    **kwargs,
) -> Rule:
    now = datetime.now(timezone.utc)
    return Rule(
        id=kwargs.get("id", uuid.uuid4()),
        user_id=user_id,
        content=content,
        is_active=kwargs.get("is_active", True),
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def _patch_user_id():
    """Make _get_user_id always return 'test-user'."""
    with patch(
        "deep_agent.aegra.personalization_routes._get_user_id",
        return_value="test-user",
    ):
        yield


@pytest.fixture
def mock_repo():
    repo = AsyncMock()
    repo.list_rules = AsyncMock(return_value=[])
    repo.count_rules = AsyncMock(return_value=0)
    repo.upsert_rule = AsyncMock(
        side_effect=lambda uid, content, **kw: _make_rule(uid, content)
    )
    repo.delete_rule = AsyncMock(return_value=True)
    repo.delete_all_rules = AsyncMock(return_value=0)
    repo.get_preferences = AsyncMock(
        return_value=UserPreferences(user_id="test-user", memory_enabled=True)
    )
    repo.update_preferences = AsyncMock(
        return_value=UserPreferences(user_id="test-user", memory_enabled=False)
    )
    with patch(
        "deep_agent.aegra.personalization_routes._get_repo",
        return_value=repo,
    ):
        yield repo


@pytest.fixture
def mock_store():
    store = AsyncMock()
    store.asearch = AsyncMock(return_value=[])
    store.aput = AsyncMock()
    store.adelete = AsyncMock()
    with patch(
        "deep_agent.aegra.personalization_routes._get_store",
        return_value=store,
    ):
        yield store


@pytest.fixture
def mock_namespace():
    with patch(
        "deep_agent.aegra.personalization_routes._get_store_namespace",
        return_value=("default",),
    ):
        yield


@pytest.fixture
def mock_cache():
    with patch(
        "deep_agent.aegra.personalization_routes._invalidate_cache",
        new_callable=AsyncMock,
    ):
        yield


@pytest.fixture
def client(_patch_user_id, mock_repo, mock_store, mock_namespace, mock_cache):
    return TestClient(app)


# ── Rule CRUD ──────────────────────────────────────────────────────


class TestListRules:
    def test_empty(self, client, mock_repo):
        resp = client.get("/personalization/rules")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_rules(self, client, mock_repo):
        rule = _make_rule()
        mock_repo.list_rules.return_value = [rule]
        resp = client.get("/personalization/rules")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "Be concise"


class TestCreateRule:
    def test_success(self, client, mock_repo):
        resp = client.post(
            "/personalization/rules",
            json={"content": "Be concise"},
        )
        assert resp.status_code == 201
        assert resp.json()["content"] == "Be concise"

    def test_empty_content_rejected(self, client):
        resp = client.post(
            "/personalization/rules",
            json={"content": "   "},
        )
        assert resp.status_code == 422

    def test_over_100_words_rejected(self, client):
        long_content = " ".join(["word"] * 101)
        resp = client.post(
            "/personalization/rules",
            json={"content": long_content},
        )
        assert resp.status_code == 422

    def test_exactly_100_words_accepted(self, client, mock_repo):
        content = " ".join(["word"] * 100)
        resp = client.post(
            "/personalization/rules",
            json={"content": content},
        )
        assert resp.status_code == 201

    def test_limit_enforced(self, client, mock_repo):
        mock_repo.upsert_rule.side_effect = ValueError(
            f"You've reached the maximum of {MAX_RULES_PER_USER} rules. "
            "Please delete some existing rules before adding new ones."
        )
        resp = client.post(
            "/personalization/rules",
            json={"content": "one more rule"},
        )
        assert resp.status_code == 400
        assert "maximum" in resp.json()["detail"].lower()


class TestDeleteRule:
    def test_success(self, client, mock_repo):
        rule_id = str(uuid.uuid4())
        resp = client.delete(f"/personalization/rules/{rule_id}")
        assert resp.status_code == 204

    def test_not_found(self, client, mock_repo):
        mock_repo.delete_rule.return_value = False
        rule_id = str(uuid.uuid4())
        resp = client.delete(f"/personalization/rules/{rule_id}")
        assert resp.status_code == 404


class TestDeleteAllRules:
    def test_success(self, client, mock_repo):
        mock_repo.delete_all_rules.return_value = 5
        resp = client.delete("/personalization/rules")
        assert resp.status_code == 204
        mock_repo.delete_all_rules.assert_called_once_with("test-user")


# ── Memory endpoints ──────────────────────────────────────────────


def _make_store_item(key: str, facts: list[str], created_at: str = "") -> MagicMock:
    item = MagicMock()
    item.key = key
    item.value = {"content": facts, "created_at": created_at}
    return item


class TestListMemories:
    def test_empty(self, client, mock_store):
        resp = client.get("/personalization/memories")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_parsed_facts(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item("k1", ["fact one", "fact two"]),
        ]
        resp = client.get("/personalization/memories")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["content"] == "fact one"
        assert data[1]["content"] == "fact two"

    def test_strips_bullet_markers(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item("k1", ["- bulleted fact", "* starred fact"]),
        ]
        resp = client.get("/personalization/memories")
        data = resp.json()
        assert data[0]["content"] == "bulleted fact"
        assert data[1]["content"] == "starred fact"


class TestDeleteAllMemories:
    def test_empty_store(self, client, mock_store):
        resp = client.delete("/personalization/memories")
        assert resp.status_code == 204

    def test_deletes_all_items(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item("k1", ["fact"]),
            _make_store_item("k2", ["fact"]),
        ]
        resp = client.delete("/personalization/memories")
        assert resp.status_code == 204
        assert mock_store.adelete.call_count == 2


class TestDeleteMemory:
    def test_not_found(self, client, mock_store):
        resp = client.delete("/personalization/memories/nonexistent")
        assert resp.status_code == 404


# ── Auth extraction ───────────────────────────────────────────────


class TestGetUserId:
    def test_aegra_user(self):
        from deep_agent.aegra.personalization_routes import _get_user_id

        request = MagicMock()
        request.state.user = SimpleNamespace(identity="aegra-user")
        assert _get_user_id(request) == "aegra-user"

    def test_header_fallback(self):
        from deep_agent.aegra.personalization_routes import _get_user_id

        request = MagicMock()
        request.state.user = None
        request.headers = {"x-user-id": "header-user"}
        assert _get_user_id(request) == "header-user"

    def test_no_identity_falls_back_to_env(self):
        from deep_agent.aegra.personalization_routes import _get_user_id

        request = MagicMock()
        request.state.user = None
        request.headers = {}
        with patch.dict("os.environ", {"USER": "testuser"}):
            assert _get_user_id(request) == "testuser"

    def test_empty_identity_falls_through_to_header(self):
        from deep_agent.aegra.personalization_routes import _get_user_id

        request = MagicMock()
        request.state.user = SimpleNamespace(identity="")
        request.headers = {"x-user-id": "header-user"}
        assert _get_user_id(request) == "header-user"


# ── Feature flags ─────────────────────────────────────────────────────


class TestFeatureFlags:
    def test_memories_disabled(self, client):
        with patch(
            "deep_agent.aegra.personalization_routes.settings.MEMORY_ENABLED",
            False,
        ):
            resp = client.get("/personalization/memories")
        assert resp.status_code == 404
        assert "disabled" in resp.json()["detail"].lower()

    def test_rules_disabled(self, client):
        with patch(
            "deep_agent.aegra.personalization_routes.settings.USER_RULES_ENABLED",
            False,
        ):
            resp = client.get("/personalization/rules")
        assert resp.status_code == 404
        assert "disabled" in resp.json()["detail"].lower()


# ── Helpers ───────────────────────────────────────────────────────────


class TestGetDefaultGraphName:
    def test_reads_first_graph(self, tmp_path):
        from deep_agent.aegra import personalization_routes as pr

        aegra = tmp_path / "aegra.json"
        aegra.write_text('{"graphs": {"health": "./graph.py"}}')
        with patch.object(pr, "_AEGRA_JSON", aegra):
            assert pr._get_default_graph_name() == "health"

    def test_empty_graphs_falls_back(self, tmp_path):
        from deep_agent.aegra import personalization_routes as pr

        aegra = tmp_path / "aegra.json"
        aegra.write_text('{"graphs": {}}')
        with patch.object(pr, "_AEGRA_JSON", aegra):
            assert pr._get_default_graph_name() == "agent"

    def test_unreadable_file_falls_back(self, tmp_path):
        from deep_agent.aegra import personalization_routes as pr

        with patch.object(pr, "_AEGRA_JSON", tmp_path / "missing.json"):
            assert pr._get_default_graph_name() == "agent"


def _asyncpg_pool(conn: AsyncMock) -> MagicMock:
    pool = MagicMock()
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire_cm
    pool_cm = MagicMock()
    pool_cm.__aenter__ = AsyncMock(return_value=pool)
    pool_cm.__aexit__ = AsyncMock(return_value=False)
    return pool_cm


class TestResolveNamespacePrefix:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        from deep_agent.aegra import personalization_routes as pr

        pr._cached_namespace_prefix = None
        pr._cached_namespace_ts = 0.0
        yield
        pr._cached_namespace_prefix = None
        pr._cached_namespace_ts = 0.0

    @pytest.mark.asyncio
    async def test_returns_cached_value(self):
        import time

        from deep_agent.aegra import personalization_routes as pr

        pr._cached_namespace_prefix = "uuid"
        pr._cached_namespace_ts = time.monotonic()
        assert await pr._resolve_namespace_prefix("u") == "uuid"

    @pytest.mark.asyncio
    async def test_detects_uuid_prefix(self):
        from deep_agent.aegra import personalization_routes as pr

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"assistant_id": "asst-1"})
        conn.fetchval = AsyncMock(return_value=1)
        with patch("asyncpg.create_pool", return_value=_asyncpg_pool(conn)):
            assert await pr._resolve_namespace_prefix("u") == "uuid"
        assert pr._cached_namespace_prefix == "uuid"

    @pytest.mark.asyncio
    async def test_detects_default_prefix(self):
        from deep_agent.aegra import personalization_routes as pr

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        conn.fetchval = AsyncMock(return_value=1)
        with patch("asyncpg.create_pool", return_value=_asyncpg_pool(conn)):
            assert await pr._resolve_namespace_prefix("u") == "default"

    @pytest.mark.asyncio
    async def test_db_error_falls_back_to_default(self):
        from deep_agent.aegra import personalization_routes as pr

        with patch("asyncpg.create_pool", side_effect=RuntimeError("db down")):
            assert await pr._resolve_namespace_prefix("u") == "default"


class TestGetStoreNamespace:
    @pytest.mark.asyncio
    async def test_default_mode(self):
        from deep_agent.aegra import personalization_routes as pr

        with patch.object(
            pr, "_resolve_namespace_prefix", AsyncMock(return_value="default")
        ):
            assert await pr._get_store_namespace("alice") == ("default",)

    @pytest.mark.asyncio
    async def test_uuid_mode(self):
        from deep_agent.aegra import personalization_routes as pr

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"assistant_id": "asst-9"})
        with (
            patch.object(
                pr, "_resolve_namespace_prefix", AsyncMock(return_value="uuid")
            ),
            patch("asyncpg.create_pool", return_value=_asyncpg_pool(conn)),
        ):
            assert await pr._get_store_namespace("alice") == ("asst-9", "alice")

    @pytest.mark.asyncio
    async def test_uuid_mode_db_error_falls_back(self):
        from deep_agent.aegra import personalization_routes as pr

        with (
            patch.object(
                pr, "_resolve_namespace_prefix", AsyncMock(return_value="uuid")
            ),
            patch("asyncpg.create_pool", side_effect=RuntimeError("db down")),
        ):
            assert await pr._get_store_namespace("alice") == ("default",)


class TestGetRepo:
    def test_returns_repository(self):
        from deep_agent.aegra.personalization_routes import _get_repo
        from deep_agent.src.personalization.repository import PersonalizationRepository

        with patch.object(PersonalizationRepository, "__init__", return_value=None):
            assert isinstance(_get_repo(), PersonalizationRepository)


class TestGetStore:
    @pytest.fixture(autouse=True)
    def _reset_store(self):
        from deep_agent.aegra import personalization_routes as pr

        pr._store_instance = None
        yield
        pr._store_instance = None

    @pytest.mark.asyncio
    async def test_returns_cached_instance(self):
        from deep_agent.aegra import personalization_routes as pr

        pr._store_instance = "cached-store"
        assert await pr._get_store() == "cached-store"

    @pytest.mark.asyncio
    async def test_creates_store_on_first_call(self):
        from deep_agent.aegra import personalization_routes as pr

        mock_pool = AsyncMock()
        mock_pool.open = AsyncMock()
        mock_store = AsyncMock()
        mock_store.setup = AsyncMock()
        with (
            patch(
                "langgraph.store.postgres.aio.AsyncPostgresStore",
                return_value=mock_store,
            ),
            patch("psycopg_pool.AsyncConnectionPool", return_value=mock_pool),
            patch("psycopg.rows.dict_row", MagicMock()),
        ):
            result = await pr._get_store()
        assert result is mock_store
        mock_pool.open.assert_awaited()
        mock_store.setup.assert_awaited()
        assert pr._store_instance is mock_store


class TestInvalidateCache:
    @pytest.mark.asyncio
    async def test_invalidates_header_and_identity(self):
        from deep_agent.aegra.personalization_routes import _invalidate_cache

        request = MagicMock()
        request.headers = {"x-user-id": "header-user"}
        request.state.user = SimpleNamespace(identity="aegra-user")
        with (
            patch(
                "deep_agent.src.cache.personalization_cache.invalidate",
                new_callable=AsyncMock,
            ) as inv,
            patch("deep_agent.aegra.graph.invalidate_graph_cache") as gcache,
        ):
            await _invalidate_cache("u1", request)
        assert inv.await_count == 3
        gcache.assert_called_once()

    @pytest.mark.asyncio
    async def test_swallows_errors(self):
        from deep_agent.aegra.personalization_routes import _invalidate_cache

        with patch(
            "deep_agent.src.cache.personalization_cache.invalidate",
            side_effect=RuntimeError("redis down"),
        ):
            await _invalidate_cache("u1")


# ── Extra memory endpoints ────────────────────────────────────────────


class TestListMemoriesExtra:
    def test_skips_blank_facts(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item("k1", ["", "  ", "kept fact"]),
        ]
        resp = client.get("/personalization/memories")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "kept fact"

    def test_deduplicate_query_keeps_longest(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item("k1", ["short", "a much longer similar fact"]),
        ]
        with patch(
            "deep_agent.src.memory.clustering.cluster_memories",
            return_value=[[0, 1]],
        ):
            resp = client.get("/personalization/memories?deduplicate=true")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "a much longer similar fact"


class TestListMemoriesClustered:
    def test_empty(self, client, mock_store):
        resp = client.get("/personalization/memories/clustered")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_groups_and_singletons(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item("k1", ["fact a", "fact b", "unique"]),
        ]
        with patch(
            "deep_agent.src.memory.clustering.cluster_memories",
            return_value=[[0, 1]],
        ):
            resp = client.get("/personalization/memories/clustered")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 2
        assert len(data[0]["facts"]) == 2
        assert len(data[1]["facts"]) == 1
        assert data[1]["facts"][0]["content"] == "unique"


class TestDeduplicateMemories:
    def test_too_few_facts(self, client, mock_store):
        mock_store.asearch.return_value = [_make_store_item("k1", ["only one"])]
        resp = client.post("/personalization/memories/deduplicate")
        assert resp.status_code == 200
        assert resp.json() == {"removed": 0, "remaining": 1}

    def test_no_clusters_to_remove(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item("k1", ["alpha", "beta"]),
        ]
        with patch(
            "deep_agent.src.memory.clustering.cluster_memories",
            return_value=[],
        ):
            resp = client.post("/personalization/memories/deduplicate")
        assert resp.json() == {"removed": 0, "remaining": 2}

    def test_rewrites_item_keeping_longest(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item("k1", ["short", "much longer fact"]),
        ]
        with patch(
            "deep_agent.src.memory.clustering.cluster_memories",
            return_value=[[0, 1]],
        ):
            resp = client.post("/personalization/memories/deduplicate")
        assert resp.status_code == 200
        body = resp.json()
        assert body["removed"] == 1
        assert body["remaining"] == 1
        mock_store.aput.assert_awaited()

    def test_skips_blank_lines_when_rewriting(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item("k1", ["short", "   ", "much longer fact"]),
        ]
        with patch(
            "deep_agent.src.memory.clustering.cluster_memories",
            return_value=[[0, 1]],
        ):
            resp = client.post("/personalization/memories/deduplicate")
        assert resp.status_code == 200
        mock_store.aput.assert_awaited()

    def test_deletes_item_when_nothing_remains(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item("k1", ["dup a"]),
            _make_store_item("k2", ["dup b longer"]),
        ]
        with patch(
            "deep_agent.src.memory.clustering.cluster_memories",
            return_value=[[0, 1]],
        ):
            resp = client.post("/personalization/memories/deduplicate")
        assert resp.status_code == 200
        assert resp.json()["removed"] == 1
        mock_store.adelete.assert_awaited()


class TestDeleteMemorySuccess:
    def test_rewrites_remaining_facts(self, client, mock_store):
        import hashlib

        key = "k1"
        facts = ["keep me", "drop me"]
        memory_id = hashlib.sha256(f"{key}:1:drop me".encode()).hexdigest()[:12]
        mock_store.asearch.return_value = [_make_store_item(key, facts)]
        resp = client.delete(f"/personalization/memories/{memory_id}")
        assert resp.status_code == 204
        mock_store.aput.assert_awaited()

    def test_deletes_file_when_last_fact_removed(self, client, mock_store):
        import hashlib

        key = "k1"
        facts = ["only fact"]
        memory_id = hashlib.sha256(f"{key}:0:only fact".encode()).hexdigest()[:12]
        mock_store.asearch.return_value = [_make_store_item(key, facts)]
        resp = client.delete(f"/personalization/memories/{memory_id}")
        assert resp.status_code == 204
        mock_store.adelete.assert_awaited()

    def test_skips_blank_lines_when_deleting(self, client, mock_store):
        import hashlib

        key = "k1"
        mock_store.asearch.return_value = [
            _make_store_item(key, ["keep", "   ", "drop"])
        ]
        memory_id = hashlib.sha256(f"{key}:2:drop".encode()).hexdigest()[:12]
        resp = client.delete(f"/personalization/memories/{memory_id}")
        assert resp.status_code == 204
        mock_store.aput.assert_awaited()


class TestPreferences:
    def test_get(self, client, mock_repo):
        resp = client.get("/personalization/preferences")
        assert resp.status_code == 200
        assert resp.json() == {"memory_enabled": True}
        mock_repo.get_preferences.assert_awaited_once_with("test-user")

    def test_update(self, client, mock_repo):
        resp = client.put(
            "/personalization/preferences",
            json={"memory_enabled": False},
        )
        assert resp.status_code == 200
        assert resp.json() == {"memory_enabled": False}
        mock_repo.update_preferences.assert_awaited()
