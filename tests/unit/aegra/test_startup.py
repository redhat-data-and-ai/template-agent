"""Unit tests for startup orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.aegra import startup


def _patched_startup(**overrides):
    """Patch every startup step.

    Including the ones a given test does not assert on: an unpatched
    step reaches for Postgres or LDAP, which succeeds on a laptop
    running the dev stack and hangs for the length of the job on a
    runner with neither.
    """
    from contextlib import ExitStack

    sync_steps = (
        "_init_ldap",
        "_check_mcp_encryption_key",
        "_setup_mcp_apps_capability",
        "_setup_otel",
        "_setup_telemetry",
        "_warmup_graph_imports",
        "_start_inflight_metrics",
        "_upgrade_signal_handlers",
    )
    async_steps = (
        "_validate_config",
        "_init_aegra_db",
        "_ensure_database",
        "_resume_interrupted_runs",
        "_warm_caches",
        "_warm_graph",
    )

    stack = ExitStack()
    for name in sync_steps:
        if name in overrides:
            stack.enter_context(patch.object(startup, name, overrides[name]))
        else:
            stack.enter_context(patch.object(startup, name, return_value="ok"))
    for name in async_steps:
        if name in overrides:
            stack.enter_context(patch.object(startup, name, overrides[name]))
        else:
            stack.enter_context(
                patch.object(startup, name, new_callable=AsyncMock, return_value="ok")
            )
    return stack


class TestRunStartup:
    def setup_method(self):
        startup._startup_complete = False

    async def test_runs_all_steps(self):
        with _patched_startup():
            result = await startup.run_startup()
        assert result["config"] == "ok"
        assert result["database"] == "ok"
        assert result["cache"] == "ok"
        assert result["telemetry"] == "ok"
        assert startup.is_ready() is True

    async def test_idempotent(self):
        startup._startup_complete = True
        with _patched_startup():
            result = await startup.run_startup()
        assert result["status"] == "already_complete"

    async def test_not_ready_when_database_fails(self):
        with _patched_startup(
            _ensure_database=AsyncMock(side_effect=ConnectionError("db down"))
        ):
            with pytest.raises(ConnectionError, match="db down"):
                await startup.run_startup()
        assert startup.is_ready() is False


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
            with pytest.raises(ValueError, match="bad port"):
                await startup._validate_config()


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
            patch(
                "deep_agent.src.projects.ProjectsRepository.ensure_tables",
                new_callable=AsyncMock,
            ) as mock_projects_ensure,
        ):
            result = await startup._ensure_database()
        assert result == "ok"
        mock_personalization.ensure_tables.assert_awaited_once()
        mock_feedback.ensure_table.assert_awaited_once()
        mock_mcp_store.ensure_tables.assert_awaited_once()
        mock_projects_ensure.assert_awaited_once()

    async def test_db_failure_propagates(self):
        mock_settings = MagicMock()
        mock_settings.database_uri = "postgresql://test"
        mock_settings.MONGODB_URI = ""
        mock_personalization = AsyncMock()
        mock_personalization.ensure_tables.side_effect = ConnectionError("refused")
        with (
            patch("deep_agent.src.settings.settings", mock_settings),
            patch(
                "deep_agent.src.personalization.repository.PersonalizationRepository",
                return_value=mock_personalization,
            ),
            patch(
                "deep_agent.src.feedback.repository.FeedbackRepository",
                return_value=AsyncMock(),
            ),
            patch(
                "deep_agent.aegra.mcp_token_store.McpTokenStore",
                return_value=AsyncMock(),
            ),
            patch(
                "deep_agent.src.projects.ProjectsRepository.ensure_tables",
                new_callable=AsyncMock,
            ),
        ):
            with pytest.raises(ConnectionError, match="refused"):
                await startup._ensure_database()

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


class TestSetupMcpAppsCapability:
    def test_ok_when_newly_installed(self):
        with patch(
            "deep_agent.aegra.mcp_apps.ensure_mcp_apps_capability_advertised",
            return_value=True,
        ):
            assert startup._setup_mcp_apps_capability() == "ok"

    def test_already_installed(self):
        with patch(
            "deep_agent.aegra.mcp_apps.ensure_mcp_apps_capability_advertised",
            return_value=False,
        ):
            assert startup._setup_mcp_apps_capability() == "already_installed"

    def test_error(self):
        with patch(
            "deep_agent.aegra.mcp_apps.ensure_mcp_apps_capability_advertised",
            side_effect=RuntimeError("boom"),
        ):
            result = startup._setup_mcp_apps_capability()
        assert result.startswith("error:")
        assert "boom" in result


class TestWarmupGraphImports:
    def test_ok(self):
        """All graph dependency imports succeed and return 'ok'."""
        with (
            patch.dict("sys.modules", {"deepagents": MagicMock()}),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                create=True,
            ),
            patch(
                "deep_agent.aegra.mcp_resource_tools.get_mcp_resource_tools",
                create=True,
            ),
            patch(
                "deep_agent.aegra.mcp_tool_auth.wrap_mcp_tools_for_auth",
                create=True,
            ),
            patch(
                "deep_agent.src.agent.config.agent_config",
                create=True,
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                create=True,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                create=True,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                create=True,
            ),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                create=True,
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                create=True,
            ),
            patch(
                "deep_agent.src.agent.config.model.parse_model_config",
                create=True,
            ),
            patch(
                "deep_agent.src.agent.provider_factory.create_model_from_spec",
                create=True,
            ),
            patch(
                "deep_agent.src.cache.model_cache.get_or_create_model_from_spec",
                create=True,
            ),
        ):
            result = startup._warmup_graph_imports()
        assert result == "ok"

    def test_import_failure_returns_warning(self):
        """A broken import returns a warning instead of crashing startup."""
        with patch.dict(
            "sys.modules",
            {"deepagents": None},
        ):
            result = startup._warmup_graph_imports()
        assert result.startswith("warning:")

    async def test_called_during_run_startup(self):
        """Verify graph_warmup is included in run_startup results."""
        startup._startup_complete = False
        mock_warmup = MagicMock(return_value="warmup-sentinel")
        with _patched_startup(_warmup_graph_imports=mock_warmup):
            result = await startup.run_startup()
        assert result["graph_warmup"] == "warmup-sentinel"
        mock_warmup.assert_called_once_with()
        startup._startup_complete = False


class TestIsReady:
    def test_not_ready_initially(self):
        startup._startup_complete = False
        assert startup.is_ready() is False


class TestWarmGraph:
    """Pre-building the graph is never fatal.

    A cold build is ~2s against ~45ms warm, paid on every deploy and —
    with scale-to-zero — on the first request after an idle period,
    which is exactly when someone is watching. A pod that cannot
    pre-build must still serve.
    """

    async def test_warms_the_graph(self):
        agent = AsyncMock()
        with patch.dict(
            "sys.modules",
            {"deep_agent.aegra.graph": MagicMock(agent=agent)},
        ):
            assert await startup._warm_graph() == "graph warmed"
        agent.assert_awaited_once()

    async def test_builds_with_no_user_so_the_cache_key_is_the_shared_one(self):
        # The factory skips MCP when user is None, and the entry it
        # populates is the one the first real request looks for.
        agent = AsyncMock()
        with patch.dict(
            "sys.modules", {"deep_agent.aegra.graph": MagicMock(agent=agent)}
        ):
            await startup._warm_graph()
        runtime = agent.await_args.args[0]
        assert runtime.user is None

    async def test_a_failed_warmup_is_reported_not_raised(self):
        agent = AsyncMock(side_effect=RuntimeError("no credentials"))
        with patch.dict(
            "sys.modules", {"deep_agent.aegra.graph": MagicMock(agent=agent)}
        ):
            result = await startup._warm_graph()
        assert result.startswith("graph not warmed")
        assert "no credentials" in result
