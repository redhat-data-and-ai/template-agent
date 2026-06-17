"""Unit tests for startup orchestrator."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.aegra import startup
from deep_agent.src.exceptions import ConfigurationError
from deep_agent.src.settings import Environment


class TestCheckPrerequisites:
    """Tests for environment-aware startup prerequisite checks."""

    def _make_settings(self, env=Environment.LOCAL, **overrides):
        """Build a mock settings object with sane defaults."""
        defaults = {
            "ENVIRONMENT": env,
            "database_uri": "postgresql://u:p@localhost:5432/db",
            "REDIS_URL": "redis://localhost:6379/0",
            "GOOGLE_APPLICATION_CREDENTIALS_CONTENT": "fake-creds",
            "VLLM_BASE_URL": None,
            "SSO_ISSUER_URL": "https://sso.example.com",
            "SSO_CLIENT_ID": "client-id",
            "SSO_CLIENT_SECRET": "client-secret",
            "LANGFUSE_PUBLIC_KEY": "pk",
            "LANGFUSE_SECRET_KEY": "sk",
        }
        defaults.update(overrides)
        mock = MagicMock()
        for k, v in defaults.items():
            setattr(mock, k, v)
        return mock

    def test_local_warns_on_missing_db(self, caplog):
        mock_settings = self._make_settings(env=Environment.LOCAL)
        with (
            patch("deep_agent.aegra.startup.settings", mock_settings),
            patch(
                "deep_agent.aegra.startup._check_db", side_effect=Exception("refused")
            ),
            patch("deep_agent.aegra.startup._check_redis"),
            patch("deep_agent.aegra.startup._prompt_md_exists", return_value=True),
        ):
            result = startup.check_prerequisites()
        assert "warn" in result
        assert "db" in caplog.text.lower() or "database" in caplog.text.lower()

    def test_demo_fails_on_missing_db(self):
        mock_settings = self._make_settings(env=Environment.DEMO)
        with (
            patch("deep_agent.aegra.startup.settings", mock_settings),
            patch(
                "deep_agent.aegra.startup._check_db", side_effect=Exception("refused")
            ),
            patch("deep_agent.aegra.startup._check_redis"),
            patch("deep_agent.aegra.startup._prompt_md_exists", return_value=True),
        ):
            with pytest.raises(ConfigurationError, match="database"):
                startup.check_prerequisites()

    def test_production_fails_on_missing_db(self):
        mock_settings = self._make_settings(env=Environment.PRODUCTION)
        with (
            patch("deep_agent.aegra.startup.settings", mock_settings),
            patch(
                "deep_agent.aegra.startup._check_db", side_effect=Exception("refused")
            ),
            patch("deep_agent.aegra.startup._check_redis"),
            patch("deep_agent.aegra.startup._prompt_md_exists", return_value=True),
        ):
            with pytest.raises(ConfigurationError, match="database"):
                startup.check_prerequisites()

    def test_local_warns_on_missing_redis(self, caplog):
        mock_settings = self._make_settings(env=Environment.LOCAL)
        with (
            patch("deep_agent.aegra.startup.settings", mock_settings),
            patch("deep_agent.aegra.startup._check_db"),
            patch(
                "deep_agent.aegra.startup._check_redis",
                side_effect=Exception("refused"),
            ),
            patch("deep_agent.aegra.startup._prompt_md_exists", return_value=True),
        ):
            result = startup.check_prerequisites()
        assert "warn" in result

    def test_demo_fails_on_missing_redis(self):
        mock_settings = self._make_settings(env=Environment.DEMO)
        with (
            patch("deep_agent.aegra.startup.settings", mock_settings),
            patch("deep_agent.aegra.startup._check_db"),
            patch(
                "deep_agent.aegra.startup._check_redis",
                side_effect=Exception("refused"),
            ),
            patch("deep_agent.aegra.startup._prompt_md_exists", return_value=True),
        ):
            with pytest.raises(ConfigurationError, match="[Rr]edis"):
                startup.check_prerequisites()

    def test_all_envs_fail_on_missing_model_config(self):
        for env in Environment:
            mock_settings = self._make_settings(
                env=env,
                GOOGLE_APPLICATION_CREDENTIALS_CONTENT=None,
                VLLM_BASE_URL=None,
            )
            with (
                patch("deep_agent.aegra.startup.settings", mock_settings),
                patch("deep_agent.aegra.startup._check_db"),
                patch("deep_agent.aegra.startup._check_redis"),
                patch("deep_agent.aegra.startup._prompt_md_exists", return_value=True),
                patch.dict(os.environ, {}, clear=False),
            ):
                os.environ.pop("OPENAI_API_KEY", None)
                with pytest.raises(ConfigurationError, match="[Mm]odel"):
                    startup.check_prerequisites()

    def test_all_envs_fail_on_missing_prompt_md(self):
        for env in Environment:
            mock_settings = self._make_settings(env=env)
            with (
                patch("deep_agent.aegra.startup.settings", mock_settings),
                patch("deep_agent.aegra.startup._check_db"),
                patch("deep_agent.aegra.startup._check_redis"),
                patch("deep_agent.aegra.startup._prompt_md_exists", return_value=False),
            ):
                with pytest.raises(ConfigurationError, match="PROMPT.md"):
                    startup.check_prerequisites()

    def test_staging_fails_on_missing_sso(self):
        mock_settings = self._make_settings(
            env=Environment.STAGING,
            SSO_ISSUER_URL=None,
            SSO_CLIENT_ID=None,
            SSO_CLIENT_SECRET=None,
        )
        with (
            patch("deep_agent.aegra.startup.settings", mock_settings),
            patch("deep_agent.aegra.startup._check_db"),
            patch("deep_agent.aegra.startup._check_redis"),
            patch("deep_agent.aegra.startup._prompt_md_exists", return_value=True),
        ):
            with pytest.raises(ConfigurationError, match="SSO"):
                startup.check_prerequisites()

    def test_local_skips_sso_check(self):
        mock_settings = self._make_settings(
            env=Environment.LOCAL,
            SSO_ISSUER_URL=None,
            SSO_CLIENT_ID=None,
            SSO_CLIENT_SECRET=None,
        )
        with (
            patch("deep_agent.aegra.startup.settings", mock_settings),
            patch("deep_agent.aegra.startup._check_db"),
            patch("deep_agent.aegra.startup._check_redis"),
            patch("deep_agent.aegra.startup._prompt_md_exists", return_value=True),
        ):
            result = startup.check_prerequisites()
        assert result == "ok"

    def test_happy_path_all_checks_pass(self):
        mock_settings = self._make_settings(env=Environment.PRODUCTION)
        with (
            patch("deep_agent.aegra.startup.settings", mock_settings),
            patch("deep_agent.aegra.startup._check_db"),
            patch("deep_agent.aegra.startup._check_redis"),
            patch("deep_agent.aegra.startup._prompt_md_exists", return_value=True),
        ):
            result = startup.check_prerequisites()
        assert result == "ok"


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
