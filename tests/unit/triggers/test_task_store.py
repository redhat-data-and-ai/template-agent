"""Unit tests for the Redis-backed task status store."""

from unittest.mock import AsyncMock, patch

from deep_agent.src.triggers.task_store import TaskRecord, TaskStore


class TestTaskRecord:
    def test_to_json_and_from_json_roundtrip(self):
        record = TaskRecord(
            task_id="abc123",
            task_name="test-task",
            status="queued",
            payload={"key": "value"},
            created_at="2026-06-23T00:00:00Z",
            updated_at="2026-06-23T00:00:00Z",
        )
        raw = record.to_json()
        restored = TaskRecord.from_json(raw)
        assert restored.task_id == "abc123"
        assert restored.task_name == "test-task"
        assert restored.status == "queued"
        assert restored.payload == {"key": "value"}
        assert restored.delivered is False

    def test_default_fields(self):
        record = TaskRecord(task_id="x", task_name="y", status="queued")
        assert record.payload == {}
        assert record.result is None
        assert record.error is None
        assert record.thread_id is None
        assert record.user_id is None
        assert record.delivered is False


class TestTaskStoreCreate:
    async def test_create_task_stores_in_redis(self):
        mock_client = AsyncMock()
        mock_client.set = AsyncMock()
        mock_client.zadd = AsyncMock()
        mock_client.expire = AsyncMock()

        store = TaskStore()
        with patch("redis.asyncio.from_url", return_value=mock_client):
            record = await store.create_task(
                task_name="my-task",
                payload={"data": 1},
                user_id="user-1",
                thread_id="thread-1",
            )

        assert record.task_name == "my-task"
        assert record.status == "queued"
        assert record.user_id == "user-1"
        assert record.thread_id == "thread-1"
        assert len(record.task_id) == 12
        mock_client.set.assert_awaited_once()
        mock_client.zadd.assert_awaited_once()

    async def test_create_task_without_user_skips_index(self):
        mock_client = AsyncMock()
        mock_client.set = AsyncMock()
        mock_client.zadd = AsyncMock()

        store = TaskStore()
        with patch("redis.asyncio.from_url", return_value=mock_client):
            await store.create_task(task_name="anon-task", payload={})

        mock_client.set.assert_awaited_once()
        mock_client.zadd.assert_not_awaited()


class TestTaskStoreUpdate:
    async def test_update_status_modifies_record(self):
        original = TaskRecord(
            task_id="t1",
            task_name="job",
            status="queued",
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=original.to_json())
        mock_client.ttl = AsyncMock(return_value=80000)
        mock_client.set = AsyncMock()

        store = TaskStore()
        with patch("redis.asyncio.from_url", return_value=mock_client):
            await store.update_status("t1", "processing")

        saved_json = mock_client.set.call_args[0][1]
        saved = TaskRecord.from_json(saved_json)
        assert saved.status == "processing"

    async def test_update_status_with_result(self):
        original = TaskRecord(
            task_id="t2",
            task_name="job",
            status="processing",
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=original.to_json())
        mock_client.ttl = AsyncMock(return_value=80000)
        mock_client.set = AsyncMock()

        store = TaskStore()
        with patch("redis.asyncio.from_url", return_value=mock_client):
            await store.update_status("t2", "completed", result={"answer": 42})

        saved = TaskRecord.from_json(mock_client.set.call_args[0][1])
        assert saved.status == "completed"
        assert saved.result == {"answer": 42}

    async def test_update_nonexistent_task_is_noop(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)

        store = TaskStore()
        with patch("redis.asyncio.from_url", return_value=mock_client):
            await store.update_status("missing", "processing")

        mock_client.set.assert_not_awaited()


class TestTaskStoreQuery:
    async def test_get_task_returns_record(self):
        record = TaskRecord(
            task_id="t1",
            task_name="job",
            status="completed",
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=record.to_json())

        store = TaskStore()
        with patch("redis.asyncio.from_url", return_value=mock_client):
            result = await store.get_task("t1")

        assert result is not None
        assert result.task_id == "t1"
        assert result.status == "completed"

    async def test_get_task_returns_none_when_missing(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)

        store = TaskStore()
        with patch("redis.asyncio.from_url", return_value=mock_client):
            result = await store.get_task("nonexistent")

        assert result is None

    async def test_get_pending_results_returns_undelivered(self):
        completed = TaskRecord(
            task_id="t1",
            task_name="done-job",
            status="completed",
            result="output",
            delivered=False,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        delivered = TaskRecord(
            task_id="t2",
            task_name="old-job",
            status="completed",
            result="old",
            delivered=True,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        queued = TaskRecord(
            task_id="t3",
            task_name="pending",
            status="queued",
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )

        mock_client = AsyncMock()
        mock_client.zrange = AsyncMock(return_value=["t1", "t2", "t3"])

        def _get_side_effect(key):
            mapping = {
                "task:t1": completed.to_json(),
                "task:t2": delivered.to_json(),
                "task:t3": queued.to_json(),
            }
            return mapping.get(key)

        mock_client.get = AsyncMock(side_effect=_get_side_effect)

        store = TaskStore()
        with patch("redis.asyncio.from_url", return_value=mock_client):
            pending = await store.get_pending_results("user-1")

        assert len(pending) == 1
        assert pending[0].task_id == "t1"

    async def test_mark_delivered_sets_flag(self):
        record = TaskRecord(
            task_id="t1",
            task_name="job",
            status="completed",
            delivered=False,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=record.to_json())
        mock_client.ttl = AsyncMock(return_value=80000)
        mock_client.set = AsyncMock()

        store = TaskStore()
        with patch("redis.asyncio.from_url", return_value=mock_client):
            await store.mark_delivered("t1")

        saved = TaskRecord.from_json(mock_client.set.call_args[0][1])
        assert saved.delivered is True
