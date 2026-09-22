"""Unit tests for shutdown orchestrator."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import deep_agent.aegra.shutdown as shutdown_mod
from deep_agent.aegra.shutdown import (
    _clear_graph_cache,
    _close_postgres,
    _close_postgres_sync,
    _close_redis,
    _drain,
    _shutdown_langfuse,
    _shutdown_langfuse_sync,
    is_shutting_down,
    register_atexit,
    register_signal_handlers,
    run_shutdown,
    run_shutdown_sync,
)


@pytest.fixture(autouse=True)
def _reset_shutdown_state():
    """Reset module-level flags before each test."""
    shutdown_mod._shutting_down = False
    shutdown_mod._shutdown_complete = False
    shutdown_mod._async_shutdown_started = False
    shutdown_mod._atexit_registered = False
    yield
    shutdown_mod._shutting_down = False
    shutdown_mod._shutdown_complete = False
    shutdown_mod._async_shutdown_started = False
    shutdown_mod._atexit_registered = False


class TestIsShuttingDown:
    def test_false_initially(self):
        assert is_shutting_down() is False

    def test_true_after_flag_set(self):
        shutdown_mod._shutting_down = True
        assert is_shutting_down() is True


class TestRunShutdown:
    async def test_runs_all_steps(self):
        with (
            patch.object(
                shutdown_mod, "_drain", new_callable=AsyncMock, return_value="ok"
            ),
            patch.object(
                shutdown_mod,
                "_shutdown_langfuse",
                new_callable=AsyncMock,
                return_value="ok",
            ),
            patch.object(shutdown_mod, "_clear_graph_cache", return_value="ok"),
            patch.object(shutdown_mod, "_close_redis", return_value="ok"),
        ):
            result = await run_shutdown()

        assert result["drain"] == "ok"
        assert result["langfuse"] == "ok"
        assert result["graph_cache"] == "ok"
        assert result["redis"] == "ok"
        assert is_shutting_down() is True
        assert shutdown_mod._shutdown_complete is True

    async def test_idempotent(self):
        shutdown_mod._shutting_down = True
        shutdown_mod._shutdown_complete = True
        result = await run_shutdown()
        assert result["status"] == "already_complete"

    async def test_sets_flag_immediately(self):
        flag_during_drain = None

        async def capture_flag():
            nonlocal flag_during_drain
            flag_during_drain = is_shutting_down()
            return "ok"

        with (
            patch.object(shutdown_mod, "_drain", side_effect=capture_flag),
            patch.object(
                shutdown_mod,
                "_shutdown_langfuse",
                new_callable=AsyncMock,
                return_value="ok",
            ),
            patch.object(shutdown_mod, "_clear_graph_cache", return_value="ok"),
            patch.object(shutdown_mod, "_close_redis", return_value="ok"),
        ):
            await run_shutdown()

        assert flag_during_drain is True

    async def test_continues_after_step_failure(self):
        with (
            patch.object(
                shutdown_mod, "_drain", new_callable=AsyncMock, return_value="ok"
            ),
            patch.object(
                shutdown_mod,
                "_shutdown_langfuse",
                new_callable=AsyncMock,
                side_effect=Exception("langfuse boom"),
            ),
            patch.object(shutdown_mod, "_clear_graph_cache", return_value="ok"),
            patch.object(shutdown_mod, "_close_redis", return_value="ok") as mock_redis,
        ):
            result = await run_shutdown()

        mock_redis.assert_called_once()
        assert shutdown_mod._shutdown_complete is True


class TestDrain:
    async def test_skips_when_zero(self):
        with patch.object(shutdown_mod, "SHUTDOWN_DRAIN_SECONDS", 0):
            result = await _drain()
        assert "skipped" in result

    async def test_sleeps_configured_duration(self):
        with patch.object(shutdown_mod, "SHUTDOWN_DRAIN_SECONDS", 0.05):
            t0 = time.monotonic()
            result = await _drain()
            elapsed = time.monotonic() - t0
        assert result == "ok"
        assert elapsed >= 0.04


class TestShutdownLangfuse:
    async def test_calls_shutdown(self):
        mock_client = MagicMock()
        mock_client.shutdown = MagicMock()
        with patch(
            "deep_agent.aegra.telemetry.get_langfuse_client", return_value=mock_client
        ):
            result = await _shutdown_langfuse()
        assert result == "ok"
        mock_client.shutdown.assert_called_once()

    async def test_falls_back_to_flush(self):
        mock_client = MagicMock(spec=[])
        mock_client.flush = MagicMock()
        with patch(
            "deep_agent.aegra.telemetry.get_langfuse_client", return_value=mock_client
        ):
            result = await _shutdown_langfuse()
        assert result == "ok"
        mock_client.flush.assert_called_once()

    async def test_skips_when_not_configured(self):
        with patch("deep_agent.aegra.telemetry.get_langfuse_client", return_value=None):
            result = await _shutdown_langfuse()
        assert "skipped" in result

    async def test_handles_timeout(self):
        def slow_shutdown():
            time.sleep(5)

        mock_client = MagicMock()
        mock_client.shutdown = slow_shutdown
        with (
            patch(
                "deep_agent.aegra.telemetry.get_langfuse_client",
                return_value=mock_client,
            ),
            patch.object(shutdown_mod, "SHUTDOWN_LANGFUSE_TIMEOUT_SECONDS", 0.1),
        ):
            result = await _shutdown_langfuse()
        assert result == "timeout"

    async def test_handles_exception(self):
        mock_client = MagicMock()
        mock_client.shutdown.side_effect = RuntimeError("boom")
        with patch(
            "deep_agent.aegra.telemetry.get_langfuse_client", return_value=mock_client
        ):
            result = await _shutdown_langfuse()
        assert "error" in result


class TestClearGraphCache:
    @pytest.fixture(autouse=True)
    def _mock_graph_module(self):
        """Pre-load a fake graph module to avoid langgraph_sdk import."""
        import sys
        import types

        fake_graph = types.ModuleType("deep_agent.aegra.graph")
        fake_graph._graph_cache = {}
        fake_graph._graph_cache_ts = {}
        self._fake_graph = fake_graph
        sys.modules["deep_agent.aegra.graph"] = fake_graph
        yield
        sys.modules.pop("deep_agent.aegra.graph", None)

    def test_clears_both_dicts(self):
        self._fake_graph._graph_cache["key1"] = "value1"
        self._fake_graph._graph_cache_ts["key1"] = 1234.0

        result = _clear_graph_cache()

        assert result == "ok"
        assert len(self._fake_graph._graph_cache) == 0
        assert len(self._fake_graph._graph_cache_ts) == 0

    def test_ok_when_empty(self):
        result = _clear_graph_cache()
        assert result == "ok"


class TestCloseRedis:
    def test_calls_close(self):
        with patch("deep_agent.aegra.redis.close_redis_client") as mock_close:
            result = _close_redis()
        assert result == "ok"
        mock_close.assert_called_once()

    def test_handles_exception(self):
        with patch(
            "deep_agent.aegra.redis.close_redis_client",
            side_effect=RuntimeError("boom"),
        ):
            result = _close_redis()
        assert "error" in result


class TestRunShutdownSync:
    def test_noop_when_complete(self):
        shutdown_mod._shutdown_complete = True
        run_shutdown_sync()

    def test_skips_when_async_already_ran(self):
        shutdown_mod._shutting_down = True
        run_shutdown_sync()
        assert shutdown_mod._shutdown_complete is True

    def test_runs_sync_cleanup(self):
        with (
            patch.object(shutdown_mod, "_shutdown_langfuse_sync", return_value="ok"),
            patch.object(shutdown_mod, "_clear_graph_cache", return_value="ok"),
            patch.object(shutdown_mod, "_close_redis", return_value="ok"),
        ):
            run_shutdown_sync()
        assert shutdown_mod._shutting_down is True
        assert shutdown_mod._shutdown_complete is True


class TestRegisterAtexit:
    def test_registers_callback(self):
        import atexit

        with patch.object(atexit, "register") as mock_register:
            register_atexit()
            mock_register.assert_called_once_with(run_shutdown_sync)

    def test_idempotent(self):
        import atexit

        with patch.object(atexit, "register") as mock_register:
            register_atexit()
            register_atexit()
            mock_register.assert_called_once()


class TestRegisterSignalHandlers:
    async def test_registers_on_running_loop(self):
        import signal

        register_signal_handlers()

        loop = asyncio.get_running_loop()
        assert loop.remove_signal_handler(signal.SIGTERM) is True
        assert loop.remove_signal_handler(signal.SIGINT) is True


class TestClosePostgres:
    @pytest.mark.asyncio
    async def test_closes_db_manager_and_pools(self):
        mock_db_manager = MagicMock()
        mock_db_manager.engine = MagicMock()
        mock_db_manager.close = AsyncMock()

        mock_pool = AsyncMock()
        mock_pool.close = AsyncMock()
        mock_registry = {"postgresql://localhost/test": mock_pool}
        mock_lock = asyncio.Lock()

        with (
            patch("aegra_api.core.database.db_manager", mock_db_manager),
            patch(
                "deep_agent.src.personalization.repository._pool_registry",
                mock_registry,
            ),
            patch(
                "deep_agent.src.personalization.repository._pool_lock",
                mock_lock,
            ),
        ):
            result = await _close_postgres()

        assert "db_manager" in result
        assert "personalization" in result
        mock_db_manager.close.assert_awaited_once()
        mock_pool.close.assert_awaited_once()
        assert len(mock_registry) == 0

    @pytest.mark.asyncio
    async def test_skips_uninitialized_db_manager(self):
        mock_db_manager = MagicMock()
        mock_db_manager.engine = None

        mock_lock = asyncio.Lock()

        with (
            patch("aegra_api.core.database.db_manager", mock_db_manager),
            patch(
                "deep_agent.src.personalization.repository._pool_registry",
                {},
            ),
            patch(
                "deep_agent.src.personalization.repository._pool_lock",
                mock_lock,
            ),
        ):
            result = await _close_postgres()

        assert result == "skipped: no pools"
        mock_db_manager.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_reports_no_pools_when_both_empty(self):
        mock_db_manager = MagicMock()
        mock_db_manager.engine = None

        mock_lock = asyncio.Lock()

        with (
            patch("aegra_api.core.database.db_manager", mock_db_manager),
            patch(
                "deep_agent.src.personalization.repository._pool_registry",
                {},
            ),
            patch(
                "deep_agent.src.personalization.repository._pool_lock",
                mock_lock,
            ),
        ):
            result = await _close_postgres()

        assert result == "skipped: no pools"
        assert "personalization" not in result
        assert "db_manager" not in result

    @pytest.mark.asyncio
    async def test_continues_after_pool_close_failure(self):
        mock_db_manager = MagicMock()
        mock_db_manager.engine = None

        pool1 = AsyncMock()
        pool1.close = AsyncMock(side_effect=RuntimeError("broken"))
        pool2 = AsyncMock()
        pool2.close = AsyncMock()
        mock_registry = {"uri1": pool1, "uri2": pool2}
        mock_lock = asyncio.Lock()

        with (
            patch("aegra_api.core.database.db_manager", mock_db_manager),
            patch(
                "deep_agent.src.personalization.repository._pool_registry",
                mock_registry,
            ),
            patch(
                "deep_agent.src.personalization.repository._pool_lock",
                mock_lock,
            ),
        ):
            result = await _close_postgres()

        assert "personalization" in result
        pool1.close.assert_awaited_once()
        pool2.close.assert_awaited_once()
        assert len(mock_registry) == 0

    @pytest.mark.asyncio
    async def test_handles_db_manager_close_failure(self):
        mock_db_manager = MagicMock()
        mock_db_manager.engine = MagicMock()
        mock_db_manager.close = AsyncMock(side_effect=RuntimeError("db error"))

        mock_lock = asyncio.Lock()

        with (
            patch("aegra_api.core.database.db_manager", mock_db_manager),
            patch(
                "deep_agent.src.personalization.repository._pool_registry",
                {},
            ),
            patch(
                "deep_agent.src.personalization.repository._pool_lock",
                mock_lock,
            ),
        ):
            result = await _close_postgres()

        assert result == "skipped: no pools"
        assert "db_manager" not in result
        mock_db_manager.close.assert_awaited_once()


class TestClosePostgresSync:
    def test_skips_when_not_initialized(self):
        mock_db_manager = MagicMock()
        mock_db_manager.engine = None
        mock_db_manager.lg_pool = None

        with (
            patch("aegra_api.core.database.db_manager", mock_db_manager),
            patch(
                "deep_agent.src.personalization.repository._pool_registry",
                {},
            ),
        ):
            result = _close_postgres_sync()

        assert result == "skipped: not initialized"

    def test_runs_cleanup_when_initialized(self):
        mock_db_manager = MagicMock()
        mock_db_manager.engine = MagicMock()
        mock_db_manager.lg_pool = MagicMock()
        mock_db_manager.close = AsyncMock()

        mock_lock = asyncio.Lock()

        with (
            patch("aegra_api.core.database.db_manager", mock_db_manager),
            patch(
                "deep_agent.src.personalization.repository._pool_registry",
                {},
            ),
            patch(
                "deep_agent.src.personalization.repository._pool_lock",
                mock_lock,
            ),
        ):
            result = _close_postgres_sync()

        assert "db_manager" in result
        mock_db_manager.close.assert_awaited_once()

    def test_runs_when_only_pool_registry_initialized(self):
        mock_db_manager = MagicMock()
        mock_db_manager.engine = None
        mock_db_manager.lg_pool = None

        mock_pool = AsyncMock()
        mock_pool.close = AsyncMock()
        mock_registry = {"uri": mock_pool}
        mock_lock = asyncio.Lock()

        with (
            patch("aegra_api.core.database.db_manager", mock_db_manager),
            patch(
                "deep_agent.src.personalization.repository._pool_registry",
                mock_registry,
            ),
            patch(
                "deep_agent.src.personalization.repository._pool_lock",
                mock_lock,
            ),
        ):
            result = _close_postgres_sync()

        assert "personalization" in result
        mock_pool.close.assert_awaited_once()
