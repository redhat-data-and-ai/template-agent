"""Unit tests for memory scheduler."""

from unittest.mock import AsyncMock, MagicMock, patch

from deep_agent.src.memory import scheduler
from deep_agent.src.memory.config import MemorySettings


class TestScheduler:
    def setup_method(self):
        scheduler._scheduler = None

    async def test_scheduler_uses_in_memory_store(self):
        """Guard against adding a persistent data store (CVE-2026-31072)."""
        enabled = MemorySettings(MEMORY_CONSOLIDATION_ENABLED=True)
        mock_cls = MagicMock()
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        with (
            patch.object(scheduler, "memory_settings", enabled),
            patch("apscheduler.AsyncScheduler", mock_cls),
            patch(
                "apscheduler.triggers.interval.IntervalTrigger",
                return_value=MagicMock(),
            ),
        ):
            result = await scheduler.start_scheduler("postgresql://test")
            assert result is True
            mock_cls.assert_called_once_with()

    async def test_start_skips_when_disabled(self):
        disabled = MemorySettings(MEMORY_CONSOLIDATION_ENABLED=False)
        with patch.object(scheduler, "memory_settings", disabled):
            result = await scheduler.start_scheduler("postgresql://test")
            assert result is False

    async def test_stop_is_safe_when_not_started(self):
        await scheduler.stop_scheduler()

    async def test_run_once_calls_all_jobs(self):
        enabled = MemorySettings(
            MEMORY_CONSOLIDATION_ENABLED=True,
            MEMORY_DECAY_ENABLED=True,
            MEMORY_CLUSTERING_ENABLED=True,
            MEMORY_RELATIONSHIPS_ENABLED=True,
        )
        with (
            patch.object(scheduler, "memory_settings", enabled),
            patch(
                "deep_agent.src.memory.scoring.decay_all_memories",
                new_callable=AsyncMock,
                return_value=5,
            ),
            patch(
                "deep_agent.src.memory.consolidation.consolidate_all_users",
                new_callable=AsyncMock,
                return_value=3,
            ),
            patch(
                "deep_agent.src.memory.clustering.cluster_all_users",
                new_callable=AsyncMock,
                return_value=2,
            ),
            patch(
                "deep_agent.src.memory.relationships.infer_all_relationships",
                new_callable=AsyncMock,
                return_value=4,
            ),
        ):
            results = await scheduler.run_once("postgresql://test")
            assert results["decay"] == 5
            assert results["consolidation"] == 3
            assert results["clustering"] == 2
            assert results["relationships"] == 4

    async def test_run_once_handles_job_failure(self):
        with (
            patch(
                "deep_agent.src.memory.scoring.decay_all_memories",
                new_callable=AsyncMock,
                side_effect=Exception("boom"),
            ),
            patch(
                "deep_agent.src.memory.consolidation.consolidate_all_users",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "deep_agent.src.memory.clustering.cluster_all_users",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "deep_agent.src.memory.relationships.infer_all_relationships",
                new_callable=AsyncMock,
                return_value=0,
            ),
        ):
            results = await scheduler.run_once("postgresql://test")
            assert results["decay"] == -1
            assert results["consolidation"] == 0
