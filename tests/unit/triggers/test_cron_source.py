"""Unit tests for CronTriggerSource."""

import asyncio
from unittest.mock import AsyncMock, patch

from deep_agent.src.triggers.config import CronJobConfig, CronTriggerConfig
from deep_agent.src.triggers.sources.cron import CronTriggerSource, _parse_cron_fields


class TestParseCronFields:
    """Test the cron expression parser."""

    def test_valid_5_field_expression(self):
        result = _parse_cron_fields("0 9 * * 1-5")
        assert result == {
            "minute": "0",
            "hour": "9",
            "day": "*",
            "month": "*",
            "day_of_week": "1-5",
        }

    def test_every_minute(self):
        result = _parse_cron_fields("* * * * *")
        assert result["minute"] == "*"

    def test_invalid_too_few_fields(self):
        assert _parse_cron_fields("0 9 * *") is None

    def test_invalid_too_many_fields(self):
        assert _parse_cron_fields("0 9 * * * 2026") is None

    def test_invalid_empty(self):
        assert _parse_cron_fields("") is None


class TestCronTriggerSource:
    """Test the cron trigger source lifecycle."""

    async def test_start_creates_tasks_for_each_job(self):
        jobs = [
            CronJobConfig(name="job-a", schedule="0 * * * *"),
            CronJobConfig(name="job-b", schedule="0 0 * * *"),
        ]
        config = CronTriggerConfig(enabled=True, jobs=jobs)
        source = CronTriggerSource(config)

        with patch.object(source, "_run_job", new_callable=AsyncMock):
            await source.start()

        assert len(source._tasks) == 2
        await source.stop()

    async def test_stop_cancels_all_tasks(self):
        config = CronTriggerConfig(
            enabled=True,
            jobs=[
                CronJobConfig(name="job", schedule="0 * * * *"),
            ],
        )
        source = CronTriggerSource(config)

        async def _hang() -> None:
            await asyncio.sleep(3600)

        source._tasks = [asyncio.create_task(_hang())]
        await source.stop()
        assert source._tasks == []

    async def test_stop_when_not_started_is_safe(self):
        config = CronTriggerConfig()
        source = CronTriggerSource(config)
        await source.stop()
        assert source._tasks == []

    async def test_invalid_cron_schedule_logs_warning(self):
        jobs = [
            CronJobConfig(name="bad-job", schedule="not-a-cron"),
            CronJobConfig(name="good-job", schedule="0 * * * *"),
        ]
        config = CronTriggerConfig(enabled=True, jobs=jobs)

        with patch("deep_agent.src.triggers.sources.cron.logger") as mock_logger:
            source = CronTriggerSource(config)

            with patch.object(source, "_run_job", new_callable=AsyncMock):
                await source.start()

        mock_logger.warning.assert_called_once()
        assert "invalid cron schedule" in mock_logger.warning.call_args.args[0]
        assert len(source._tasks) == 1
        await source.stop()

    async def test_empty_jobs_list_is_valid(self):
        config = CronTriggerConfig(enabled=True, jobs=[])
        source = CronTriggerSource(config)
        await source.start()
        assert source._tasks == []
        await source.stop()

    async def test_aiter_returns_self(self):
        config = CronTriggerConfig()
        source = CronTriggerSource(config)
        assert source.__aiter__() is source
