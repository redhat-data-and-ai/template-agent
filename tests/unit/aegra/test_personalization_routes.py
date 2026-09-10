"""Unit tests for personalization API routes.

Uses FastAPI TestClient with mocked dependencies (repository, store, cache)
so no real Postgres/Redis is needed.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_agent.aegra.personalization_routes import (
    router,
)
from deep_agent.src.memory.instructions import USER_MEMORY_STORE_KEY
from deep_agent.src.personalization.models import ConsentRecord, Rule, UserPreferences

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
def _patch_memory_user_id():
    """Make memory_user_id always return 'test-user'."""
    with patch(
        "deep_agent.aegra.personalization_routes.memory_user_id",
        new_callable=AsyncMock,
        return_value="test-user",
    ):
        yield


@pytest.fixture
def mock_repo():
    repo = AsyncMock()
    repo.list_rules = AsyncMock(return_value=[])
    repo.list_rules_for_users = AsyncMock(return_value=[])
    repo.upsert_rule = AsyncMock(
        side_effect=lambda uid, content, **kw: _make_rule(uid, content)
    )
    repo.delete_rule = AsyncMock(return_value=True)
    repo.delete_rule_for_users = AsyncMock(return_value=True)
    repo.delete_all_rules_for_users = AsyncMock(return_value=0)
    repo.get_preferences = AsyncMock(
        return_value=UserPreferences(user_id="test-user", memory_enabled=True)
    )
    repo.update_preferences = AsyncMock(
        return_value=UserPreferences(user_id="test-user", memory_enabled=False)
    )
    repo.store_consent = AsyncMock(
        side_effect=lambda uid, action: ConsentRecord(user_id=uid, action=action)
    )
    repo.get_consent_status = AsyncMock(return_value=None)
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
def client(_patch_memory_user_id, mock_repo, mock_store, mock_namespace, mock_cache):
    return TestClient(app)


# ── Rule CRUD ──────────────────────────────────────────────────────


class TestListRules:
    def test_empty(self, client, mock_repo):
        resp = client.get("/personalization/rules")
        assert resp.status_code == 200
        assert resp.json() == {"rules": []}

    def test_returns_rules(self, client, mock_repo):
        rule = _make_rule()
        mock_repo.list_rules.return_value = [rule]
        resp = client.get("/personalization/rules")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["rules"]) == 1
        assert data["rules"][0]["content"] == "Be concise"


class TestCreateRule:
    def test_success(self, client, mock_repo):
        resp = client.post(
            "/personalization/rules",
            json={"content": "Be concise"},
        )
        assert resp.status_code == 201
        assert resp.json()["content"] == "Be concise"

    def test_with_client_id(self, client, mock_repo):
        rule_id = str(uuid.uuid4())
        resp = client.post(
            "/personalization/rules",
            json={"id": rule_id, "content": "Be concise", "is_active": True},
        )
        assert resp.status_code == 201
        mock_repo.upsert_rule.assert_awaited()

    def test_conflict_when_owned_by_other_user(self, client, mock_repo):
        mock_repo.upsert_rule.side_effect = PermissionError("belongs to another user")
        resp = client.post(
            "/personalization/rules",
            json={"content": "Be concise"},
        )
        assert resp.status_code == 409
        assert "another user" in resp.json()["detail"].lower()


class TestDeleteRule:
    def test_success(self, client, mock_repo):
        rule_id = str(uuid.uuid4())
        resp = client.delete(f"/personalization/rules/{rule_id}")
        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted"}

    def test_not_found(self, client, mock_repo):
        mock_repo.delete_rule.return_value = False
        rule_id = str(uuid.uuid4())
        resp = client.delete(f"/personalization/rules/{rule_id}")
        assert resp.status_code == 404

    def test_invalid_id(self, client):
        resp = client.delete("/personalization/rules/not-a-uuid")
        assert resp.status_code == 400


class TestDeleteAllRules:
    def test_success(self, client, mock_repo):
        resp = client.delete("/personalization/rules")
        assert resp.status_code == 204
        mock_repo.delete_all_rules_for_users.assert_awaited()


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
            _make_store_item(USER_MEMORY_STORE_KEY, ["fact one", "fact two"]),
        ]
        resp = client.get("/personalization/memories")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["content"] == "fact one"
        assert data[1]["content"] == "fact two"

    def test_ignores_non_memory_store_keys(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item("quarterly.md", ["a report line"]),
            _make_store_item(USER_MEMORY_STORE_KEY, ["user fact"]),
        ]
        resp = client.get("/personalization/memories")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "user fact"

    def test_strips_bullet_markers(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item(
                USER_MEMORY_STORE_KEY, ["- bulleted fact", "* starred fact"]
            ),
        ]
        resp = client.get("/personalization/memories")
        data = resp.json()
        assert data[0]["content"] == "bulleted fact"
        assert data[1]["content"] == "starred fact"

    def test_string_content_is_one_fact(self, client, mock_store):
        item = MagicMock()
        item.key = "/user_profile.md"
        item.value = {
            "content": "The user's date of birth is June 14, 2003.",
            "created_at": "",
        }
        mock_store.asearch.return_value = [item]
        resp = client.get("/personalization/memories")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "The user's date of birth is June 14, 2003."

    def test_non_string_created_at_becomes_empty(self, client, mock_store):
        item = MagicMock()
        item.key = USER_MEMORY_STORE_KEY
        item.value = {"content": "a fact", "created_at": 123}
        mock_store.asearch.return_value = [item]
        resp = client.get("/personalization/memories")
        assert resp.status_code == 200
        assert resp.json()[0]["created_at"] == ""

    def test_falls_back_to_default_assistant_namespace(self, client, mock_store):
        item = MagicMock()
        item.key = "/user_profile.md"
        item.value = {"content": "The user's name is Darshika.", "created_at": ""}

        async def asearch(namespace, limit=100):
            if namespace == ("asst-uuid", "test-user"):
                return []
            if namespace == ("default", "test-user"):
                return [item]
            return []

        mock_store.asearch.side_effect = asearch
        with patch(
            "deep_agent.aegra.personalization_routes._get_store_namespace",
            new=AsyncMock(return_value=("asst-uuid", "test-user")),
        ):
            resp = client.get("/personalization/memories")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "The user's name is Darshika."


class TestDeleteAllMemories:
    def test_empty_store(self, client, mock_store):
        resp = client.delete("/personalization/memories")
        assert resp.status_code == 204

    def test_deletes_only_memory_file(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item(USER_MEMORY_STORE_KEY, ["fact"]),
            _make_store_item("quarterly.md", ["report"]),
        ]
        resp = client.delete("/personalization/memories")
        assert resp.status_code == 204
        mock_store.adelete.assert_awaited_once()
        assert mock_store.adelete.await_args.args[1] == USER_MEMORY_STORE_KEY


class TestDeleteMemory:
    def test_not_found(self, client, mock_store):
        resp = client.delete("/personalization/memories/nonexistent")
        assert resp.status_code == 404


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


class TestGetAssistantIdForStore:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        from deep_agent.aegra import personalization_routes as pr

        pr._cached_assistant_id = None
        pr._cached_assistant_ts = 0.0
        yield
        pr._cached_assistant_id = None
        pr._cached_assistant_ts = 0.0

    @pytest.mark.asyncio
    async def test_auth_off_is_default(self):
        from deep_agent.aegra import personalization_routes as pr

        with patch("deep_agent.aegra.auth.ENABLE_AUTH", False):
            assert await pr._get_assistant_id_for_store() == "default"

    @pytest.mark.asyncio
    async def test_auth_on_reads_assistant_table(self):
        from deep_agent.aegra import personalization_routes as pr

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"assistant_id": "asst-1"})
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("asyncpg.create_pool", return_value=_asyncpg_pool(conn)),
        ):
            assert await pr._get_assistant_id_for_store() == "asst-1"
        assert pr._cached_assistant_id == "asst-1"

    @pytest.mark.asyncio
    async def test_auth_on_db_error_falls_back_to_default(self):
        from deep_agent.aegra import personalization_routes as pr

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("asyncpg.create_pool", side_effect=RuntimeError("db down")),
        ):
            assert await pr._get_assistant_id_for_store() == "default"

    @pytest.mark.asyncio
    async def test_auth_on_returns_cached_assistant_id(self):
        from deep_agent.aegra import personalization_routes as pr

        pr._cached_assistant_id = "asst-cached"
        pr._cached_assistant_ts = time.monotonic()
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("asyncpg.create_pool") as create_pool,
        ):
            assert await pr._get_assistant_id_for_store() == "asst-cached"
        create_pool.assert_not_called()


class TestGetStoreNamespace:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        from deep_agent.aegra import personalization_routes as pr

        pr._cached_assistant_id = None
        pr._cached_assistant_ts = 0.0
        yield
        pr._cached_assistant_id = None
        pr._cached_assistant_ts = 0.0

    @pytest.mark.asyncio
    async def test_local_namespace(self):
        from deep_agent.aegra import personalization_routes as pr

        with patch("deep_agent.aegra.auth.ENABLE_AUTH", False):
            assert await pr._get_store_namespace("alice") == ("default", "alice")

    @pytest.mark.asyncio
    async def test_prod_namespace_uses_assistant_uuid(self):
        from deep_agent.aegra import personalization_routes as pr

        with patch.object(
            pr, "_get_assistant_id_for_store", AsyncMock(return_value="asst-9")
        ):
            assert await pr._get_store_namespace("alice") == ("asst-9", "alice")


class TestCandidateNamespaces:
    @pytest.mark.asyncio
    async def test_uuid_namespace_also_searches_default(self):
        from deep_agent.aegra import personalization_routes as pr

        with patch.object(
            pr, "_get_store_namespace", AsyncMock(return_value=("asst-9", "alice"))
        ):
            assert await pr._candidate_namespaces("alice") == [
                ("asst-9", "alice"),
                ("default", "alice"),
            ]

    @pytest.mark.asyncio
    async def test_default_namespace_is_not_duplicated(self):
        from deep_agent.aegra import personalization_routes as pr

        with patch.object(
            pr, "_get_store_namespace", AsyncMock(return_value=("default", "alice"))
        ):
            assert await pr._candidate_namespaces("alice") == [("default", "alice")]


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
    async def test_invalidates_extra_ids(self):
        from deep_agent.aegra.personalization_routes import _invalidate_cache

        with (
            patch(
                "deep_agent.src.cache.personalization_cache.invalidate",
                new_callable=AsyncMock,
            ) as inv,
            patch("deep_agent.aegra.graph.invalidate_graph_cache"),
        ):
            await _invalidate_cache("u1", extra_ids=["u2", ""])
        invalidated = {call.args[0] for call in inv.await_args_list}
        assert invalidated == {"u1", "u2"}

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
            _make_store_item(USER_MEMORY_STORE_KEY, ["", "  ", "kept fact"]),
        ]
        resp = client.get("/personalization/memories")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "kept fact"

    def test_deduplicate_query_keeps_longest(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item(
                USER_MEMORY_STORE_KEY, ["short", "a much longer similar fact"]
            ),
        ]
        with patch(
            "deep_agent.src.memory.clustering.near_duplicate_groups",
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
            _make_store_item(USER_MEMORY_STORE_KEY, ["fact a", "fact b", "unique"]),
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
        mock_store.asearch.return_value = [
            _make_store_item(USER_MEMORY_STORE_KEY, ["only one"])
        ]
        resp = client.post("/personalization/memories/deduplicate")
        assert resp.status_code == 200
        assert resp.json() == {"removed": 0, "remaining": 1}

    def test_no_clusters_to_remove(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item(USER_MEMORY_STORE_KEY, ["alpha", "beta"]),
        ]
        with patch(
            "deep_agent.src.memory.clustering.near_duplicate_groups",
            return_value=[],
        ):
            resp = client.post("/personalization/memories/deduplicate")
        assert resp.json() == {"removed": 0, "remaining": 2}

    def test_rewrites_item_keeping_longest(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item(USER_MEMORY_STORE_KEY, ["short", "much longer fact"]),
        ]
        with patch(
            "deep_agent.src.memory.clustering.near_duplicate_groups",
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
            _make_store_item(
                USER_MEMORY_STORE_KEY, ["short", "   ", "much longer fact"]
            ),
        ]
        with patch(
            "deep_agent.src.memory.clustering.near_duplicate_groups",
            return_value=[[0, 1]],
        ):
            resp = client.post("/personalization/memories/deduplicate")
        assert resp.status_code == 200
        mock_store.aput.assert_awaited()

    def test_does_not_touch_report_files(self, client, mock_store):
        mock_store.asearch.return_value = [
            _make_store_item(USER_MEMORY_STORE_KEY, ["short", "much longer fact"]),
            _make_store_item("quarterly.md", ["report"]),
        ]
        with patch(
            "deep_agent.src.memory.clustering.near_duplicate_groups",
            return_value=[[0, 1]],
        ):
            resp = client.post("/personalization/memories/deduplicate")
        assert resp.status_code == 200
        assert resp.json()["removed"] == 1
        mock_store.aput.assert_awaited()
        assert mock_store.aput.await_args.args[1] == USER_MEMORY_STORE_KEY


class TestDeleteMemorySuccess:
    def test_rewrites_remaining_facts(self, client, mock_store):
        import hashlib

        key = USER_MEMORY_STORE_KEY
        facts = ["keep me", "drop me"]
        memory_id = hashlib.sha256(f"{key}:1:drop me".encode()).hexdigest()[:12]
        mock_store.asearch.return_value = [_make_store_item(key, facts)]
        resp = client.delete(f"/personalization/memories/{memory_id}")
        assert resp.status_code == 204
        mock_store.aput.assert_awaited()

    def test_deletes_file_when_last_fact_removed(self, client, mock_store):
        import hashlib

        key = USER_MEMORY_STORE_KEY
        facts = ["only fact"]
        memory_id = hashlib.sha256(f"{key}:0:only fact".encode()).hexdigest()[:12]
        mock_store.asearch.return_value = [_make_store_item(key, facts)]
        resp = client.delete(f"/personalization/memories/{memory_id}")
        assert resp.status_code == 204
        mock_store.adelete.assert_awaited()

    def test_skips_blank_lines_when_deleting(self, client, mock_store):
        import hashlib

        key = USER_MEMORY_STORE_KEY
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


# ── Consent endpoints ─────────────────────────────────────────────────


class TestApproveConsent:
    def test_success(self, client, mock_repo):
        resp = client.post("/personalization/consent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_consent"] is True
        assert "granted_at" in data
        mock_repo.store_consent.assert_awaited_once_with("test-user", "approved")


class TestGetConsentStatus:
    def test_no_consent_record(self, client, mock_repo):
        mock_repo.get_consent_status.return_value = None
        resp = client.get("/personalization/consent")
        assert resp.status_code == 200
        assert resp.json() == {"has_consent": False, "granted_at": None}

    def test_approved(self, client, mock_repo):
        mock_repo.get_consent_status.return_value = ConsentRecord(
            user_id="test-user",
            action="approved",
        )
        resp = client.get("/personalization/consent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_consent"] is True
        assert data["granted_at"] is not None

    def test_revoked(self, client, mock_repo):
        mock_repo.get_consent_status.return_value = ConsentRecord(
            user_id="test-user",
            action="revoked",
        )
        resp = client.get("/personalization/consent")
        assert resp.status_code == 200
        assert resp.json()["has_consent"] is False


class TestRevokeConsent:
    def test_success(self, client, mock_repo):
        resp = client.delete("/personalization/consent")
        assert resp.status_code == 200
        assert resp.json() == {"status": "revoked"}
        mock_repo.store_consent.assert_awaited_once_with("test-user", "revoked")
