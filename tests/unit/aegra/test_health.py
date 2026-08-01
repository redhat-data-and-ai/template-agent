"""Unit tests for health check endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.aegra.health import (
    check_cache,
    check_config,
    check_database,
    check_redis,
    get_health_status,
    health_response,
)


def _patch_all_checks(**overrides):
    """Return a context-manager stack that mocks every health sub-check.

    Defaults to ``{"status": "ok"}`` for each check.  Pass keyword
    overrides keyed by check name to customise individual results.
    """
    defaults = {
        "database": {"status": "ok"},
        "redis": {"status": "ok"},
        "config": {"status": "ok"},
        "cache": {"status": "ok"},
        "mcp_servers": {"status": "ok", "servers": {}, "healthy": 0, "total": 0},
        "llm_provider": {"status": "ok", "provider": "vllm"},
    }
    defaults.update(overrides)

    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        patch(
            "deep_agent.aegra.health.check_database",
            new_callable=AsyncMock,
            return_value=defaults["database"],
        )
    )
    stack.enter_context(
        patch(
            "deep_agent.aegra.health.check_redis",
            new_callable=AsyncMock,
            return_value=defaults["redis"],
        )
    )
    stack.enter_context(
        patch(
            "deep_agent.aegra.health.check_config",
            return_value=defaults["config"],
        )
    )
    stack.enter_context(
        patch(
            "deep_agent.aegra.health.check_cache",
            return_value=defaults["cache"],
        )
    )
    stack.enter_context(
        patch(
            "deep_agent.aegra.mcp_health.check_mcp_servers",
            new_callable=AsyncMock,
            return_value=defaults["mcp_servers"],
        )
    )
    stack.enter_context(
        patch(
            "deep_agent.aegra.mcp_health.check_llm_provider",
            new_callable=AsyncMock,
            return_value=defaults["llm_provider"],
        )
    )
    return stack


class TestCheckConfig:
    def test_valid_config(self):
        mock_settings = MagicMock()
        mock_settings.database_uri = "postgresql://test"
        mock_settings.AGENT_PORT = 5002
        with patch("deep_agent.src.settings.settings", mock_settings):
            result = check_config()
        assert result["status"] == "ok"

    def test_missing_database(self):
        mock_settings = MagicMock()
        mock_settings.database_uri = ""
        mock_settings.AGENT_PORT = 5002
        with patch("deep_agent.src.settings.settings", mock_settings):
            result = check_config()
        assert result["status"] == "warning"


class TestCheckDatabase:
    async def test_no_database_uri(self):
        mock_settings = MagicMock()
        mock_settings.database_uri = ""
        with patch("deep_agent.src.settings.settings", mock_settings):
            result = await check_database()
        assert result["status"] == "skipped"

    async def test_database_error(self):
        mock_settings = MagicMock()
        mock_settings.database_uri = "postgresql://bad"
        with (
            patch("deep_agent.src.settings.settings", mock_settings),
            patch(
                "psycopg.AsyncConnection.connect",
                side_effect=Exception("connection refused"),
            ),
        ):
            result = await check_database()
        assert result["status"] == "error"


class TestCheckRedis:
    async def test_no_redis(self):
        with patch(
            "deep_agent.aegra.redis.get_redis_client",
            return_value=None,
        ):
            result = await check_redis()
        assert result["status"] == "skipped"

    async def test_redis_ok(self):
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        with patch(
            "deep_agent.aegra.redis.get_redis_client",
            return_value=mock_client,
        ):
            result = await check_redis()
        assert result["status"] == "ok"
        assert "latency_ms" in result


class TestCheckCache:
    def test_returns_stats(self):
        with patch(
            "deep_agent.src.cache.metrics.get_stats",
            return_value={"hits": 10, "misses": 2},
        ):
            result = check_cache()
        assert result["status"] == "ok"


class TestGetHealthStatus:
    async def test_healthy(self):
        with _patch_all_checks():
            result = await get_health_status()
        assert result["status"] == "healthy"
        assert "uptime_seconds" in result
        assert "checks" in result
        assert "mcp_servers" in result["checks"]
        assert "llm_provider" in result["checks"]

    async def test_unhealthy_on_db_error(self):
        with _patch_all_checks(database={"status": "error", "error": "down"}):
            result = await get_health_status()
        assert result["status"] == "unhealthy"

    async def test_degraded_when_mcp_subset_down(self):
        """MCP servers partially down → degraded, NOT unhealthy."""
        mcp = {
            "status": "warning",
            "servers": {
                "a": {"status": "healthy"},
                "b": {"status": "unreachable"},
            },
            "healthy": 1,
            "total": 2,
        }
        with _patch_all_checks(mcp_servers=mcp):
            result = await get_health_status()
        assert result["status"] == "degraded"

    async def test_degraded_when_all_mcp_down(self):
        """All MCP servers down → degraded (pod stays in rotation)."""
        mcp = {
            "status": "warning",
            "servers": {
                "a": {"status": "unreachable"},
                "b": {"status": "timeout"},
            },
            "healthy": 0,
            "total": 2,
        }
        with _patch_all_checks(mcp_servers=mcp):
            result = await get_health_status()
        assert result["status"] == "degraded"

    async def test_degraded_when_llm_down(self):
        """LLM provider down → degraded, NOT unhealthy."""
        llm = {"status": "warning", "provider": "vllm", "error": "timeout"}
        with _patch_all_checks(llm_provider=llm):
            result = await get_health_status()
        assert result["status"] == "degraded"

    async def test_db_error_overrides_mcp_warning(self):
        """DB error + MCP warning → unhealthy (critical wins)."""
        with _patch_all_checks(
            database={"status": "error", "error": "down"},
            mcp_servers={"status": "warning", "servers": {}, "healthy": 0, "total": 1},
        ):
            result = await get_health_status()
        assert result["status"] == "unhealthy"

    async def test_redis_error_is_degraded_not_unhealthy(self):
        """Redis is non-critical so an error produces degraded."""
        with _patch_all_checks(redis={"status": "error", "error": "refused"}):
            result = await get_health_status()
        assert result["status"] == "degraded"


class TestHealthResponse:
    async def test_200_when_healthy(self):
        with patch(
            "deep_agent.aegra.health.get_health_status",
            new_callable=AsyncMock,
            return_value={"status": "healthy"},
        ):
            code, body = await health_response()
        assert code == 200

    async def test_200_when_degraded(self):
        with patch(
            "deep_agent.aegra.health.get_health_status",
            new_callable=AsyncMock,
            return_value={"status": "degraded"},
        ):
            code, body = await health_response()
        assert code == 200

    async def test_503_when_unhealthy(self):
        with patch(
            "deep_agent.aegra.health.get_health_status",
            new_callable=AsyncMock,
            return_value={"status": "unhealthy"},
        ):
            code, body = await health_response()
        assert code == 503
