"""Redis-backed task status store for tracking headless worker tasks."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_TASK_TTL = 86400  # 24 hours
_KEY_PREFIX = "task:"
_USER_INDEX_PREFIX = "user_tasks:"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskRecord:
    """A background task tracked in Redis."""

    task_id: str
    task_name: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    thread_id: str | None = None
    user_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    delivered: bool = False

    def to_json(self) -> str:
        """Serialize the task record to a JSON string."""
        data = asdict(self)
        if isinstance(data.get("result"), (dict, list)):
            pass
        elif data.get("result") is not None:
            data["result"] = str(data["result"])
        return json.dumps(data, default=str)

    @classmethod
    def from_json(cls, raw: str) -> TaskRecord:
        """Deserialize a task record from a JSON string."""
        data = json.loads(raw)
        return cls(**data)


class TaskStore:
    """Redis-backed store for task status tracking with Postgres audit trail."""

    def __init__(self, redis_url: str | None = None) -> None:
        """Initialize the task store with an optional Redis URL."""
        self._redis_url = redis_url or settings.REDIS_URL
        self._client: Any = None
        self._audit: Any = None

    def _get_audit_repo(self) -> Any:
        """Lazily create the Postgres audit repository."""
        if self._audit is None:
            try:
                from deep_agent.src.triggers.task_repository import TaskRepository

                self._audit = TaskRepository(settings.database_uri)
            except Exception:
                logger.debug("audit repository unavailable", exc_info=True)
        return self._audit

    async def _ensure_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    async def create_task(
        self,
        task_name: str,
        payload: dict[str, Any],
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> TaskRecord:
        """Create a new task record and store it in Redis."""
        task_id = uuid.uuid4().hex[:12]
        now = _now_iso()
        record = TaskRecord(
            task_id=task_id,
            task_name=task_name,
            status="queued",
            payload=payload,
            thread_id=thread_id,
            user_id=user_id,
            created_at=now,
            updated_at=now,
        )
        client = await self._ensure_client()
        key = f"{_KEY_PREFIX}{task_id}"
        await client.set(key, record.to_json(), ex=_TASK_TTL)

        if user_id:
            await client.zadd(
                f"{_USER_INDEX_PREFIX}{user_id}",
                {task_id: datetime.now(timezone.utc).timestamp()},
            )
            await client.expire(f"{_USER_INDEX_PREFIX}{user_id}", _TASK_TTL)

        logger.info(
            "task created", task_id=task_id, task_name=task_name, status="queued"
        )

        audit = self._get_audit_repo()
        if audit:
            try:
                await audit.ensure_table()
                await audit.insert_task(task_id, task_name, payload, thread_id, user_id)
            except Exception:
                logger.debug("audit insert failed", task_id=task_id, exc_info=True)

        return record

    async def update_status(
        self,
        task_id: str,
        status: str,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Update the status of an existing task."""
        client = await self._ensure_client()
        key = f"{_KEY_PREFIX}{task_id}"
        raw = await client.get(key)
        if raw is None:
            logger.warning("task not found for status update", task_id=task_id)
            return

        record = TaskRecord.from_json(raw)
        record.status = status
        record.updated_at = _now_iso()
        if result is not None:
            record.result = result
        if error is not None:
            record.error = error

        ttl = await client.ttl(key)
        await client.set(key, record.to_json(), ex=max(ttl, 3600))
        logger.info("task status updated", task_id=task_id, status=status)

        audit = self._get_audit_repo()
        if audit:
            try:
                result_str = str(result)[:2000] if result is not None else None
                await audit.update_status(task_id, status, result_str, error)
            except Exception:
                logger.debug("audit update failed", task_id=task_id, exc_info=True)

    async def get_task(self, task_id: str) -> TaskRecord | None:
        """Retrieve a task record by ID, or None if not found."""
        client = await self._ensure_client()
        raw = await client.get(f"{_KEY_PREFIX}{task_id}")
        if raw is None:
            return None
        return TaskRecord.from_json(raw)

    async def get_pending_results(self, user_id: str) -> list[TaskRecord]:
        """Return completed or failed tasks that have not been delivered."""
        client = await self._ensure_client()
        task_ids = await client.zrange(f"{_USER_INDEX_PREFIX}{user_id}", 0, -1)
        pending = []
        for tid in task_ids:
            record = await self.get_task(tid)
            if (
                record
                and record.status in ("completed", "failed")
                and not record.delivered
            ):
                pending.append(record)
        return pending

    async def mark_delivered(self, task_id: str) -> None:
        """Mark a task as delivered to the user."""
        client = await self._ensure_client()
        key = f"{_KEY_PREFIX}{task_id}"
        raw = await client.get(key)
        if raw is None:
            return
        record = TaskRecord.from_json(raw)
        record.delivered = True
        record.updated_at = _now_iso()
        ttl = await client.ttl(key)
        await client.set(key, record.to_json(), ex=max(ttl, 3600))

        audit = self._get_audit_repo()
        if audit:
            try:
                await audit.mark_delivered(task_id)
            except Exception:
                logger.debug("audit mark_delivered failed", task_id=task_id, exc_info=True)

    async def close(self) -> None:
        """Close the Redis client connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
