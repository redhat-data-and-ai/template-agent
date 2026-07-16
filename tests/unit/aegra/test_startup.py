"""Unit tests for startup orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

from deep_agent.aegra import startup


class TestRunStartup:
    def setup_method(self):
        startup._startup_complete = False

    async def test_runs_all_steps(self):
        with (
            patch.object(
                startup, "_validate_config", new_callable=AsyncMock, return_value="ok"
            ),
            patch.object(
                startup, "_ensure_database", new_callable=AsyncMock, return_value="ok"
            ),
            patch.object(
                startup, "_warm_caches", new_callable=AsyncMock, return_value="ok"
            ),
            patch.object(
                startup,
                "_start_scheduler",
                new_callable=AsyncMock,
                return_value="ok",
            ),
            patch.object(startup, "_setup_telemetry", return_value="ok"),
        ):
            result = await startup.run_startup()
        assert result["config"] == "ok"
        assert result["database"] == "ok"
        assert result["cache"] == "ok"
        assert result["scheduler"] == "ok"
        assert result["telemetry"] == "ok"
        assert startup.is_ready() is True

    async def test_idempotent(self):
        startup._startup_complete = True
        result = await startup.run_startup()
        assert result["status"] == "already_complete"


class TestValidateConfig:
    async def test_valid(self):
        with patch(
            "deep_agent.src.settings.validate_config",
        ):
            result = await startup._validate_config()
        assert result == "ok"

    async def test_warning(self):
        with patch(
            "deep_agent.src.settings.validate_config",
            side_effect=ValueError("bad port"),
        ):
            result = await startup._validate_config()
        assert "warning" in result


class TestEnsureDatabase:
    async def test_no_db(self):
        mock_settings = MagicMock()
        mock_settings.database_uri = ""
        mock_settings.MONGODB_URI = ""
        with patch("deep_agent.src.settings.settings", mock_settings):
            result = await startup._ensure_database()
        assert "skipped" in result

    async def test_db_ok(self):
        mock_settings = MagicMock()
        mock_settings.database_uri = "postgresql://test"
        mock_settings.MONGODB_URI = ""
        mock_personalization = AsyncMock()
        mock_feedback = AsyncMock()
        mock_mcp_store = AsyncMock()
        with (
            patch("deep_agent.src.settings.settings", mock_settings),
            patch(
                "deep_agent.src.personalization.repository.PersonalizationRepository",
                return_value=mock_personalization,
            ),
            patch(
                "deep_agent.src.feedback.repository.FeedbackRepository",
                return_value=mock_feedback,
            ),
            patch(
                "deep_agent.aegra.mcp_token_store.McpTokenStore",
                return_value=mock_mcp_store,
            ),
        ):
            result = await startup._ensure_database()
        assert result == "ok"
        mock_personalization.ensure_tables.assert_awaited_once()
        mock_feedback.ensure_table.assert_awaited_once()
        mock_mcp_store.ensure_tables.assert_awaited_once()

    async def test_mongo_indexes_when_configured(self):
        import sys

        mock_settings = MagicMock()
        mock_settings.database_uri = ""
        mock_settings.MONGODB_URI = "mongodb://test"
        mock_settings.MONGODB_DB = "tokenusage"
        mock_mongo = AsyncMock()
        mock_module = MagicMock()
        mock_module.TokenUsageMongoRepository.return_value = mock_mongo
        with (
            patch("deep_agent.src.settings.settings", mock_settings),
            patch.dict(
                sys.modules,
                {"deep_agent.src.token_budget.mongo_repository": mock_module},
            ),
        ):
            result = await startup._ensure_database()
        assert result == "ok"
        mock_mongo.ensure_indexes.assert_awaited_once()


class TestWarmCaches:
    async def test_disabled(self):
        mock_cache_settings = MagicMock()
        mock_cache_settings.CACHE_ENABLED = False
        with patch("deep_agent.src.cache.config.cache_settings", mock_cache_settings):
            result = await startup._warm_caches()
        assert "skipped" in result

    async def test_enabled(self):
        mock_cache_settings = MagicMock()
        mock_cache_settings.CACHE_ENABLED = True
        with (
            patch("deep_agent.src.cache.config.cache_settings", mock_cache_settings),
            patch(
                "deep_agent.src.cache.warming.warm_caches",
                new_callable=AsyncMock,
            ),
        ):
            result = await startup._warm_caches()
        assert result == "ok"


class TestStartScheduler:
    async def test_disabled(self):
        mock_mem_settings = MagicMock()
        mock_mem_settings.MEMORY_CONSOLIDATION_ENABLED = False
        with patch("deep_agent.src.memory.config.memory_settings", mock_mem_settings):
            result = await startup._start_scheduler()
        assert "skipped" in result


class TestSetupTelemetry:
    def test_ok(self):
        with patch("deep_agent.aegra.telemetry.setup_langfuse_tracing"):
            result = startup._setup_telemetry()
        assert result == "ok"

    def test_failure(self):
        with patch(
            "deep_agent.aegra.telemetry.setup_langfuse_tracing",
            side_effect=Exception("boom"),
        ):
            result = startup._setup_telemetry()
        assert "warning" in result


class TestIsReady:
    def test_not_ready_initially(self):
        startup._startup_complete = False
        assert startup.is_ready() is False
