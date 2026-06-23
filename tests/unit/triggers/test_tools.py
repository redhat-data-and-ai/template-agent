"""Unit tests for headless worker tools."""

from unittest.mock import AsyncMock, patch

from deep_agent.src.triggers.task_store import TaskRecord
from deep_agent.src.triggers.tools import (
    check_task_status,
    get_builtin_tools,
    get_pending_results,
    queue_task,
)


class TestQueueTask:
    async def test_creates_task_record_and_pushes_to_stream(self):
        mock_record = TaskRecord(
            task_id="abc123",
            task_name="test",
            status="queued",
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(return_value="1234-0")
        mock_redis.aclose = AsyncMock()

        with (
            patch("deep_agent.src.triggers.tools._store") as mock_store,
            patch("redis.asyncio.from_url", return_value=mock_redis),
        ):
            mock_store.create_task = AsyncMock(return_value=mock_record)
            result = await queue_task(
                task_name="test",
                payload={"key": "val"},
                thread_id="thread-1",
                user_id="user-1",
            )

        assert "abc123" in result
        assert "queued" in result.lower()
        mock_store.create_task.assert_awaited_once()
        mock_redis.xadd.assert_awaited_once()

    async def test_returns_task_id_in_response(self):
        mock_record = TaskRecord(
            task_id="xyz789",
            task_name="report",
            status="queued",
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(return_value="5678-0")
        mock_redis.aclose = AsyncMock()

        with (
            patch("deep_agent.src.triggers.tools._store") as mock_store,
            patch("redis.asyncio.from_url", return_value=mock_redis),
        ):
            mock_store.create_task = AsyncMock(return_value=mock_record)
            result = await queue_task(task_name="report", payload={})

        assert "xyz789" in result


class TestCheckTaskStatus:
    async def test_completed_task_returns_result(self):
        record = TaskRecord(
            task_id="t1",
            task_name="report",
            status="completed",
            result={"answer": 42},
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        with patch("deep_agent.src.triggers.tools._store") as mock_store:
            mock_store.get_task = AsyncMock(return_value=record)
            result = await check_task_status("t1")

        assert "COMPLETED" in result
        assert "42" in result

    async def test_failed_task_returns_error(self):
        record = TaskRecord(
            task_id="t2",
            task_name="export",
            status="failed",
            error="Connection timeout",
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        with patch("deep_agent.src.triggers.tools._store") as mock_store:
            mock_store.get_task = AsyncMock(return_value=record)
            result = await check_task_status("t2")

        assert "FAILED" in result
        assert "Connection timeout" in result

    async def test_queued_task_returns_status(self):
        record = TaskRecord(
            task_id="t3",
            task_name="batch",
            status="queued",
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        with patch("deep_agent.src.triggers.tools._store") as mock_store:
            mock_store.get_task = AsyncMock(return_value=record)
            result = await check_task_status("t3")

        assert "QUEUED" in result

    async def test_missing_task_returns_not_found(self):
        with patch("deep_agent.src.triggers.tools._store") as mock_store:
            mock_store.get_task = AsyncMock(return_value=None)
            result = await check_task_status("nonexistent")

        assert "not found" in result.lower()


class TestGetPendingResults:
    async def test_returns_completed_undelivered_tasks(self):
        records = [
            TaskRecord(
                task_id="t1",
                task_name="report",
                status="completed",
                result="Report data",
                delivered=False,
                created_at="2026-01-01",
                updated_at="2026-01-01",
            ),
        ]
        with patch("deep_agent.src.triggers.tools._store") as mock_store:
            mock_store.get_pending_results = AsyncMock(return_value=records)
            mock_store.mark_delivered = AsyncMock()
            result = await get_pending_results("user-1")

        assert "1 background task(s)" in result
        assert "report" in result
        assert "COMPLETED" in result
        mock_store.mark_delivered.assert_awaited_once_with("t1")

    async def test_returns_no_pending_message(self):
        with patch("deep_agent.src.triggers.tools._store") as mock_store:
            mock_store.get_pending_results = AsyncMock(return_value=[])
            result = await get_pending_results("user-1")

        assert "No pending" in result


class TestGetBuiltinTools:
    def test_returns_three_tools(self):
        tools = get_builtin_tools()
        assert len(tools) == 3

    def test_tool_names(self):
        tools = get_builtin_tools()
        names = {t.name for t in tools}
        assert names == {"queue_task", "check_task_status", "get_pending_results"}

    def test_tools_are_callable(self):
        tools = get_builtin_tools()
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
