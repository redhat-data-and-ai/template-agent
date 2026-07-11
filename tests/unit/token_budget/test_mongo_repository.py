"""Unit tests for Mongo token usage repository retries."""

from __future__ import annotations

from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
from pymongo.errors import ServerSelectionTimeoutError

from deep_agent.src.token_budget.mongo_repository import TokenUsageMongoRepository


@pytest.mark.asyncio
async def test_increment_usage_retries_transient_mongo_error() -> None:
    expected = {
        "thread_id": "thread-1",
        "total_tokens": 150,
        "input_tokens": 100,
        "output_tokens": 50,
    }

    collection = AsyncMock()
    collection.find_one_and_update = AsyncMock(
        side_effect=[
            ServerSelectionTimeoutError("timeout"),
            expected,
        ]
    )

    with (
        patch.object(
            TokenUsageMongoRepository,
            "_thread_collection",
            new_callable=PropertyMock,
            return_value=collection,
        ),
        patch.object(TokenUsageMongoRepository, "ensure_indexes", new=AsyncMock()),
    ):
        repo = TokenUsageMongoRepository(
            "mongodb://mongodb:27017", db_name="tokenusage"
        )
        result = await repo.increment_usage(
            "thread-1", 100, 50, agent_name="health-assistant"
        )

    assert result == expected
    assert collection.find_one_and_update.await_count == 2


@pytest.mark.asyncio
async def test_increment_usage_does_not_retry_runtime_error() -> None:
    collection = AsyncMock()
    collection.find_one_and_update = AsyncMock(return_value=None)

    with (
        patch.object(
            TokenUsageMongoRepository,
            "_thread_collection",
            new_callable=PropertyMock,
            return_value=collection,
        ),
        patch.object(TokenUsageMongoRepository, "ensure_indexes", new=AsyncMock()),
    ):
        repo = TokenUsageMongoRepository(
            "mongodb://mongodb:27017", db_name="tokenusage"
        )
        with pytest.raises(RuntimeError, match="Failed to increment Mongo token usage"):
            await repo.increment_usage("thread-1", 100, 50)

    assert collection.find_one_and_update.await_count == 1


@pytest.mark.asyncio
async def test_ensure_indexes_runs_once_per_process() -> None:
    import deep_agent.src.token_budget.mongo_repository as mongo_module

    mongo_module._INDEXES_ENSURED = False

    thread_collection = AsyncMock()
    daily_collection = AsyncMock()

    with (
        patch.object(
            TokenUsageMongoRepository,
            "_thread_collection",
            new_callable=PropertyMock,
            return_value=thread_collection,
        ),
        patch.object(
            TokenUsageMongoRepository,
            "_daily_collection",
            new_callable=PropertyMock,
            return_value=daily_collection,
        ),
    ):
        repo = TokenUsageMongoRepository(
            "mongodb://mongodb:27017", db_name="tokenusage"
        )
        await repo.ensure_indexes()
        await repo.ensure_indexes()

    assert thread_collection.create_index.await_count == 2
    assert daily_collection.create_index.await_count == 2

    mongo_module._INDEXES_ENSURED = False
