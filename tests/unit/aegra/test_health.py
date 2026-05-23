"""Unit tests for health check endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

from deep_agent.aegra.health import (
    check_cache,
    check_config,
    check_database,
    check_redis,
    get_health_status,
    health_response,
)


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
        with (
            patch(
                "deep_agent.aegra.health.check_database",
                new_callable=AsyncMock,
                return_value={"status": "ok"},
            ),
            patch(
                "deep_agent.aegra.health.check_redis",
                new_callable=AsyncMock,
                return_value={"status": "ok"},
            ),
            patch(
                "deep_agent.aegra.health.check_config",
                return_value={"status": "ok"},
            ),
            patch(
                "deep_agent.aegra.health.check_cache",
                return_value={"status": "ok"},
            ),
        ):
            result = await get_health_status()
        assert result["status"] == "healthy"
        assert "uptime_seconds" in result
        assert "checks" in result

    async def test_unhealthy_on_db_error(self):
        with (
            patch(
                "deep_agent.aegra.health.check_database",
                new_callable=AsyncMock,
                return_value={"status": "error", "error": "down"},
            ),
            patch(
                "deep_agent.aegra.health.check_redis",
                new_callable=AsyncMock,
                return_value={"status": "ok"},
            ),
            patch(
                "deep_agent.aegra.health.check_config",
                return_value={"status": "ok"},
            ),
            patch(
                "deep_agent.aegra.health.check_cache",
                return_value={"status": "ok"},
            ),
        ):
            result = await get_health_status()
        assert result["status"] == "unhealthy"


class TestHealthResponse:
    async def test_200_when_healthy(self):
        with (
            patch(
                "deep_agent.aegra.health.get_health_status",
                new_callable=AsyncMock,
                return_value={"status": "healthy"},
            ),
        ):
            code, body = await health_response()
        assert code == 200

    async def test_503_when_unhealthy(self):
        with (
            patch(
                "deep_agent.aegra.health.get_health_status",
                new_callable=AsyncMock,
                return_value={"status": "unhealthy"},
            ),
        ):
            code, body = await health_response()
        assert code == 503
