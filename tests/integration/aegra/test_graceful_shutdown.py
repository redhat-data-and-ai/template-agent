"""Integration tests for graceful shutdown under concurrent load.

Proves the shutdown sequence completes cleanly while simulated
graph runs are active, and that all subsystems are torn down
within the configured timeout budget.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import deep_agent.aegra.shutdown as shutdown_mod

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _reset_shutdown_state():
    shutdown_mod._shutting_down = False
    shutdown_mod._shutdown_complete = False
    yield
    shutdown_mod._shutting_down = False
    shutdown_mod._shutdown_complete = False


def _mock_all_subsystems(drain_seconds=0):
    """Context manager that mocks all external subsystems for shutdown."""
    mock_langfuse = MagicMock()
    mock_langfuse.shutdown = MagicMock()

    return (
        patch.object(shutdown_mod, "SHUTDOWN_DRAIN_SECONDS", drain_seconds),
        patch(
            "deep_agent.aegra.telemetry.get_langfuse_client",
            return_value=mock_langfuse,
        ),
        patch(
            "deep_agent.src.memory.scheduler.stop_scheduler",
            new_callable=AsyncMock,
        ),
        patch("deep_agent.aegra.redis.close_redis_client"),
    )


class TestShutdownUnderConcurrentActivity:
    async def test_shutdown_while_tasks_are_running(self):
        """Simulate concurrent graph runs during shutdown.

        Active tasks should be able to continue during the drain
        period. After drain, cleanup runs and completes.
        """
        completed_tasks = []

        async def simulate_graph_run(task_id: int, duration: float):
            await asyncio.sleep(duration)
            completed_tasks.append(task_id)

        tasks = [
            asyncio.create_task(simulate_graph_run(i, i * 0.05))
            for i in range(5)
        ]

        patches = _mock_all_subsystems(drain_seconds=0.3)
        with patches[0], patches[1], patches[2], patches[3]:
            t0 = time.monotonic()
            result = await shutdown_mod.run_shutdown()
            elapsed = time.monotonic() - t0

        assert result["drain"] == "ok"
        assert result["langfuse"] == "ok"
        assert result["scheduler"] == "ok"
        assert result["redis"] == "ok"
        assert shutdown_mod._shutdown_complete is True
        assert elapsed < 2.0

        await asyncio.gather(*tasks, return_exceptions=True)
        assert len(completed_tasks) == 5

    async def test_resources_cleaned_up_after_drain(self):
        """After drain period, all subsystems are torn down."""
        mock_langfuse = MagicMock()
        mock_langfuse.shutdown = MagicMock()
        mock_stop = AsyncMock()
        mock_close = MagicMock()

        with (
            patch.object(shutdown_mod, "SHUTDOWN_DRAIN_SECONDS", 0),
            patch(
                "deep_agent.aegra.telemetry.get_langfuse_client",
                return_value=mock_langfuse,
            ),
            patch(
                "deep_agent.src.memory.scheduler.stop_scheduler",
                mock_stop,
            ),
            patch("deep_agent.aegra.redis.close_redis_client", mock_close),
        ):
            await shutdown_mod.run_shutdown()

        mock_langfuse.shutdown.assert_called_once()
        mock_stop.assert_awaited_once()
        mock_close.assert_called_once()


class TestHealthDuringShutdown:
    async def test_health_returns_503_during_shutdown(self):
        from deep_agent.aegra.health import health_response

        with patch(
            "deep_agent.aegra.health.get_health_status",
            new_callable=AsyncMock,
            return_value={"status": "healthy"},
        ):
            code_before, _ = await health_response()
        assert code_before == 200

        shutdown_mod._shutting_down = True

        code_after, body = await health_response()
        assert code_after == 503
        assert body["status"] == "shutting_down"


class TestLangfuseTimeoutResilience:
    async def test_slow_langfuse_does_not_block_shutdown(self):
        """A hanging Langfuse server must not prevent Redis/scheduler cleanup."""

        def slow_shutdown():
            time.sleep(30)

        mock_langfuse = MagicMock()
        mock_langfuse.shutdown = slow_shutdown
        mock_close = MagicMock()
        mock_stop = AsyncMock()

        with (
            patch.object(shutdown_mod, "SHUTDOWN_DRAIN_SECONDS", 0),
            patch.object(shutdown_mod, "SHUTDOWN_LANGFUSE_TIMEOUT_SECONDS", 0.2),
            patch(
                "deep_agent.aegra.telemetry.get_langfuse_client",
                return_value=mock_langfuse,
            ),
            patch(
                "deep_agent.src.memory.scheduler.stop_scheduler",
                mock_stop,
            ),
            patch("deep_agent.aegra.redis.close_redis_client", mock_close),
        ):
            t0 = time.monotonic()
            result = await shutdown_mod.run_shutdown()
            elapsed = time.monotonic() - t0

        assert result["langfuse"] == "timeout"
        assert result["scheduler"] == "ok"
        assert result["redis"] == "ok"
        mock_stop.assert_awaited_once()
        mock_close.assert_called_once()
        assert elapsed < 3.0


class TestIdempotentConcurrentShutdown:
    async def test_concurrent_calls_execute_once(self):
        """Two concurrent run_shutdown() calls should only execute steps once."""
        call_count = 0

        async def counting_drain():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return "ok"

        patches = _mock_all_subsystems(drain_seconds=0)
        with patches[0], patches[1], patches[2], patches[3]:
            with patch.object(shutdown_mod, "_drain", side_effect=counting_drain):
                results = await asyncio.gather(
                    shutdown_mod.run_shutdown(),
                    shutdown_mod.run_shutdown(),
                )

        real_runs = [r for r in results if "drain" in r]
        skipped = [r for r in results if r.get("status") == "already_complete"]
        assert len(real_runs) == 1
        assert len(skipped) == 1
