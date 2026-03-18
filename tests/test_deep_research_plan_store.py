"""Tests for deep research plan store module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.plan_store import (
    clear_plan,
    delete_plan_for_thread,
    get_plan,
    get_plan_context,
    get_plan_enrichment,
    get_plan_metadata,
    get_plan_with_load,
    get_thread_owner,
    load_plan_from_store,
    load_similar_plans,
    persist_plan_to_store,
    register_thread_owner,
    set_plan,
    set_plan_context,
)


def _unique_thread_id() -> str:
    """Generate unique thread id for test isolation."""
    import uuid

    return f"test-{uuid.uuid4().hex[:12]}"


class TestSetPlanAndGetPlan:
    """Test cases for set_plan and get_plan."""

    @pytest.mark.asyncio
    async def test_set_plan_and_get_plan_roundtrip(self) -> None:
        """Plan set locally is retrieved by get_plan."""
        thread_id = _unique_thread_id()
        plan = ["q1", "q2", "q3"]

        await set_plan(thread_id, plan)
        result = await get_plan(thread_id)

        assert result == plan
        await clear_plan(thread_id)

    @pytest.mark.asyncio
    async def test_get_plan_missing_returns_none(self) -> None:
        """get_plan for unknown thread returns None."""
        thread_id = _unique_thread_id()
        result = await get_plan(thread_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_plan_with_metadata(self) -> None:
        """Metadata is stored and retrievable via get_plan_metadata."""
        thread_id = _unique_thread_id()
        plan = ["q1", "q2"]
        metadata = {"query": "main query", "complexity": "high"}

        await set_plan(thread_id, plan, metadata=metadata)
        result = await get_plan_metadata(thread_id)

        assert result is not None
        assert result.get("query") == "main query"
        assert result.get("complexity") == "high"
        await clear_plan(thread_id)

    @pytest.mark.asyncio
    async def test_set_plan_with_enrichment(self) -> None:
        """Enrichment data is stored and retrievable."""
        thread_id = _unique_thread_id()
        plan = ["q1"]
        enrichment = {"discovered_tools": ["tool1"]}

        await set_plan(thread_id, plan, enrichment=enrichment)
        result = await get_plan_enrichment(thread_id)

        assert result is not None
        assert result.get("discovered_tools") == ["tool1"]
        await clear_plan(thread_id)

    @pytest.mark.asyncio
    async def test_set_plan_with_store_persists_first(self) -> None:
        """When store and user_id provided, plan is persisted before cache."""
        thread_id = _unique_thread_id()
        plan = ["q1", "q2"]
        store = MagicMock()
        store.aput = AsyncMock()

        with patch(
            "template_agent.src.core.deep_research.plan_store.persist_plan_to_store",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await set_plan(
                thread_id, plan, metadata={"query": "q"}, store=store, user_id="u1"
            )

        result = await get_plan(thread_id, user_id="u1")
        assert result == plan
        await clear_plan(thread_id, user_id="u1")

    @pytest.mark.asyncio
    async def test_set_plan_user_scoped_cache_key(self) -> None:
        """Plans are keyed by user_id:thread_id when user_id provided."""
        thread_id = _unique_thread_id()
        plan_a = ["q1"]
        plan_b = ["q2", "q3"]

        await set_plan(thread_id, plan_a, user_id=None)
        await set_plan(thread_id, plan_b, user_id="user1")

        result_no_user = await get_plan(thread_id, user_id=None)
        result_with_user = await get_plan(thread_id, user_id="user1")

        assert result_no_user == plan_a
        assert result_with_user == plan_b
        await clear_plan(thread_id)
        await clear_plan(thread_id, user_id="user1")


class TestClearPlan:
    """Test cases for clear_plan."""

    @pytest.mark.asyncio
    async def test_clear_plan_removes_from_cache(self) -> None:
        """clear_plan removes plan so get_plan returns None."""
        thread_id = _unique_thread_id()
        await set_plan(thread_id, ["q1"])

        await clear_plan(thread_id)
        result = await get_plan(thread_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_clear_plan_removes_metadata_and_enrichment(self) -> None:
        """clear_plan removes metadata and enrichment."""
        thread_id = _unique_thread_id()
        await set_plan(thread_id, ["q1"], metadata={"query": "q"}, enrichment={"x": 1})

        await clear_plan(thread_id)

        assert await get_plan_metadata(thread_id) is None
        assert await get_plan_enrichment(thread_id) is None


class TestGetPlanWithLoad:
    """Test cases for get_plan_with_load."""

    @pytest.mark.asyncio
    async def test_get_plan_with_load_cache_hit(self) -> None:
        """When plan in cache, returns without loading from store."""
        thread_id = _unique_thread_id()
        plan = ["q1", "q2"]
        await set_plan(thread_id, plan)

        result = await get_plan_with_load(thread_id)

        assert result == plan
        await clear_plan(thread_id)

    @pytest.mark.asyncio
    async def test_get_plan_with_load_cache_miss_loads_from_store(self) -> None:
        """When cache miss and store provided, loads from store."""
        thread_id = _unique_thread_id()
        plan = ["q1", "q2"]

        async def mock_aget(*args: object, **kwargs: object) -> dict:
            await asyncio.sleep(0)
            return {"plan": plan, "thread_id": kwargs.get("key", thread_id)}

        store = MagicMock()
        store.aget = mock_aget

        result = await get_plan_with_load(thread_id, store=store, user_id="u1")

        assert result == plan
        await clear_plan(thread_id, user_id="u1")

    @pytest.mark.asyncio
    async def test_get_plan_with_load_no_store_returns_none(self) -> None:
        """When cache miss and no store, returns None."""
        thread_id = _unique_thread_id()
        result = await get_plan_with_load(thread_id)
        assert result is None


class TestLoadPlanFromStore:
    """Test cases for load_plan_from_store."""

    @pytest.mark.asyncio
    async def test_load_plan_from_store_no_user_returns_none(self) -> None:
        """Missing user_id returns None."""
        store = MagicMock()
        result = await load_plan_from_store(store, user_id="", thread_id="t1")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_plan_from_store_no_thread_returns_none(self) -> None:
        """Missing thread_id returns None."""
        store = MagicMock()
        result = await load_plan_from_store(store, user_id="u1", thread_id="")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_plan_from_store_success(self) -> None:
        """Valid store response returns plan."""
        thread_id = _unique_thread_id()
        plan = ["q1", "q2"]

        async def mock_aget(*args: object, **kwargs: object) -> dict:
            await asyncio.sleep(0)
            return {"plan": plan, "thread_id": kwargs.get("key", thread_id)}

        store = MagicMock()
        store.aget = mock_aget

        result = await load_plan_from_store(store, user_id="u1", thread_id=thread_id)

        assert result == plan


class TestPersistPlanToStore:
    """Test cases for persist_plan_to_store."""

    @pytest.mark.asyncio
    async def test_persist_plan_no_user_returns_false(self) -> None:
        """Missing user_id returns False."""
        store = MagicMock()
        result = await persist_plan_to_store(
            store, user_id="", thread_id="t1", plan=["q1"], query="q"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_persist_plan_success(self) -> None:
        """Successful aput returns True."""
        store = MagicMock()
        store.aput = AsyncMock()

        result = await persist_plan_to_store(
            store, user_id="u1", thread_id="t1", plan=["q1", "q2"], query="main"
        )

        assert result is True
        store.aput.assert_awaited_once()
        call_kw = store.aput.await_args[1]
        assert call_kw["value"]["plan"] == ["q1", "q2"]
        assert call_kw["value"]["thread_id"] == "t1"
        assert call_kw["value"]["subquery_count"] == 2


class TestSetPlanContextAndGetPlanContext:
    """Test cases for set_plan_context and get_plan_context."""

    @pytest.mark.asyncio
    async def test_set_and_get_plan_context_roundtrip(self) -> None:
        """Context set is retrieved by get_plan_context."""
        thread_id = _unique_thread_id()
        context = {"query": "q", "complexity": "high"}

        await set_plan_context(thread_id, context)
        result = await get_plan_context(thread_id)

        assert result == context
        await clear_plan(thread_id)

    @pytest.mark.asyncio
    async def test_get_plan_context_missing_returns_none(self) -> None:
        """get_plan_context for unknown thread returns None."""
        thread_id = _unique_thread_id()
        result = await get_plan_context(thread_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_plan_context_with_store_persists(self) -> None:
        """When store and user_id provided, context is persisted."""
        thread_id = _unique_thread_id()
        context = {"query": "q"}
        store = MagicMock()
        store.aput = AsyncMock()

        await set_plan_context(thread_id, context, user_id="u1", store=store)

        store.aput.assert_awaited_once()
        await clear_plan(thread_id, user_id="u1")


class TestRegisterThreadOwnerAndGetThreadOwner:
    """Test cases for register_thread_owner and get_thread_owner."""

    @pytest.mark.asyncio
    async def test_register_and_get_thread_owner(self) -> None:
        """Thread owner is registered and retrievable."""
        thread_id = _unique_thread_id()
        await register_thread_owner(thread_id, "user1")
        owner = await get_thread_owner(thread_id)
        assert owner == "user1"

    @pytest.mark.asyncio
    async def test_get_thread_owner_unregistered_returns_none(self) -> None:
        """Unregistered thread returns None."""
        thread_id = _unique_thread_id()
        owner = await get_thread_owner(thread_id)
        assert owner is None

    @pytest.mark.asyncio
    async def test_register_thread_owner_first_write_wins(self) -> None:
        """First registration wins (no overwrite)."""
        thread_id = _unique_thread_id()
        await register_thread_owner(thread_id, "user1")
        await register_thread_owner(thread_id, "user2")
        owner = await get_thread_owner(thread_id)
        assert owner == "user1"


class TestDeletePlanForThread:
    """Test cases for delete_plan_for_thread."""

    @pytest.mark.asyncio
    async def test_delete_plan_clears_local_and_store(self) -> None:
        """delete_plan_for_thread clears local cache and store."""
        thread_id = _unique_thread_id()
        await set_plan(thread_id, ["q1"], user_id="u1")
        store = MagicMock()
        store.adelete = AsyncMock()

        result = await delete_plan_for_thread(store, user_id="u1", thread_id=thread_id)

        assert result is True
        assert await get_plan(thread_id, user_id="u1") is None
        store.adelete.assert_awaited_once()


class TestLoadSimilarPlans:
    """Test cases for load_similar_plans."""

    @pytest.mark.asyncio
    async def test_load_similar_plans_no_user_returns_empty(self) -> None:
        """Missing user_id returns empty list."""
        store = MagicMock()
        result = await load_similar_plans(store, user_id="", query="q")
        assert result == []

    @pytest.mark.asyncio
    async def test_load_similar_plans_no_asearch_returns_empty(self) -> None:
        """Store without asearch returns empty list."""
        store = MagicMock(spec=[])
        result = await load_similar_plans(store, user_id="u1", query="q")
        assert result == []

    @pytest.mark.asyncio
    async def test_load_similar_plans_returns_results(self) -> None:
        """asearch results are returned as list of dicts."""
        store = MagicMock()
        item = MagicMock()
        item.value = {"plan": ["q1"], "query": "revenue"}
        store.asearch = AsyncMock(return_value=[item])

        result = await load_similar_plans(store, user_id="u1", query="revenue", limit=3)

        assert len(result) == 1
        assert result[0]["plan"] == ["q1"]


class TestConcurrentAccess:
    """Test cases for concurrent access (multiple threads/coroutines)."""

    @pytest.mark.asyncio
    async def test_concurrent_set_and_get_plan(self) -> None:
        """Concurrent set_plan and get_plan are safe."""
        thread_ids = [_unique_thread_id() for _ in range(5)]
        plans = [[f"q{i}"] for i in range(5)]

        async def set_and_get(i: int) -> list[str] | None:
            await set_plan(thread_ids[i], plans[i])
            return await get_plan(thread_ids[i])

        results = await asyncio.gather(*[set_and_get(i) for i in range(5)])

        for i, result in enumerate(results):
            assert result == plans[i]

        for tid in thread_ids:
            await clear_plan(tid)
