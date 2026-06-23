"""MongoDB store for per-thread and per-user daily token usage."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument

from deep_agent.src.error_handling import mongo_retry
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_INDEXES_ENSURED = False


class TokenUsageMongoRepository:
    """MongoDB token usage: per-thread counts and per-user daily rollup."""

    def __init__(self, mongodb_uri: str, db_name: str) -> None:
        self._uri = mongodb_uri
        self._db_name = db_name
        self._client: AsyncIOMotorClient | None = None

    def _get_client(self) -> AsyncIOMotorClient:
        if self._client is None:
            self._client = AsyncIOMotorClient(self._uri)
        return self._client

    @property
    def _db(self):
        return self._get_client()[self._db_name]

    @property
    def _thread_collection(self):
        return self._db["thread_token_usage"]

    @property
    def _daily_collection(self):
        return self._db["user_daily_token_usage"]

    @mongo_retry
    async def ensure_indexes(self) -> None:
        """Create indexes once per process (also invoked from startup)."""
        global _INDEXES_ENSURED  # noqa: PLW0603
        if _INDEXES_ENSURED:
            return
        await self._thread_collection.create_index("thread_id", unique=True)
        await self._thread_collection.create_index("updated_at")
        await self._daily_collection.create_index(
            [("user_id", 1), ("date", 1)],
            unique=True,
        )
        await self._daily_collection.create_index("date")
        _INDEXES_ENSURED = True
        logger.info("MongoDB token usage indexes ensured")

    @mongo_retry
    async def increment_usage(
        self,
        thread_id: str,
        input_tokens: int,
        output_tokens: int,
        *,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """Atomically add tokens for a thread and return the updated document."""
        input_tokens = max(input_tokens, 0)
        output_tokens = max(output_tokens, 0)
        total_delta = input_tokens + output_tokens
        now = datetime.now(UTC)

        update: dict[str, Any] = {
            "$inc": {
                "total_tokens": total_delta,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "$set": {"updated_at": now},
            "$setOnInsert": {"thread_id": thread_id},
        }
        if agent_name:
            update["$set"]["agent_name"] = agent_name

        result = await self._thread_collection.find_one_and_update(
            {"thread_id": thread_id},
            update,
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        if result is None:
            raise RuntimeError(f"Failed to increment Mongo token usage for {thread_id}")
        return result

    @mongo_retry
    async def increment_daily_usage(
        self,
        user_id: str,
        tokens: int,
        *,
        date: str | None = None,
    ) -> dict[str, Any]:
        """Increment a user's total token usage for a UTC calendar day."""
        tokens = max(tokens, 0)
        day = date or datetime.now(UTC).strftime("%Y-%m-%d")
        now = datetime.now(UTC)

        result = await self._daily_collection.find_one_and_update(
            {"user_id": user_id, "date": day},
            {
                "$inc": {"total_tokens": tokens},
                "$set": {"updated_at": now},
                "$setOnInsert": {"user_id": user_id, "date": day},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        if result is None:
            raise RuntimeError(f"Failed to increment daily usage for {user_id} on {day}")
        return result

    @mongo_retry
    async def get_thread_usage(self, thread_id: str) -> dict[str, Any] | None:
        return await self._thread_collection.find_one({"thread_id": thread_id})

    @mongo_retry
    async def get_daily_usage(
        self,
        user_id: str,
        *,
        date: str | None = None,
    ) -> dict[str, Any] | None:
        day = date or datetime.now(UTC).strftime("%Y-%m-%d")
        return await self._daily_collection.find_one({"user_id": user_id, "date": day})

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
