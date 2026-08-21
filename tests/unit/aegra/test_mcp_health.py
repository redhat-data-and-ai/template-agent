"""Unit tests for MCP and LLM provider health checks."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from deep_agent.aegra import mcp_health
from deep_agent.aegra.mcp_health import (
    _HEALTH_CACHE_TTL,
    _ping_mcp_server,
    check_llm_provider,
    check_mcp_servers,
    invalidate_health_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset module-level caches and OTEL state between tests."""
    invalidate_health_cache()
    mcp_health._gauge_initialized = False
    mcp_health._mcp_health_gauge = None
    mcp_health._llm_health_gauge = None
    yield
    invalidate_health_cache()


def _mock_servers(servers: dict):
    """Patch agent_config.get_mcp_servers to return *servers*."""
    mock_config = MagicMock()
    mock_config.get_mcp_servers.return_value = servers
    return patch("deep_agent.src.agent.config.agent_config", mock_config)


TWO_SERVERS = {
    "server-a": {
        "url": "http://a:5001/mcp",
        "transport": "streamable_http",
        "enabled": True,
        "auth": False,
        "ssl_verify": False,
        "timeout": 10,
    },
    "server-b": {
        "url": "http://b:5002/mcp",
        "transport": "streamable_http",
        "enabled": True,
        "auth": False,
        "ssl_verify": False,
        "timeout": 10,
    },
}


# ── _ping_mcp_server ─────────────────────────────────────────────


class TestPingMcpServer:
    async def test_healthy(self):
        mock_resp = MagicMock(status_code=200)
        with patch("deep_agent.aegra.mcp_health.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await _ping_mcp_server("test", "http://x/mcp", 5.0, False)

        assert result["status"] == "healthy"
        assert "latency_ms" in result
        assert result["http_status"] == 200

    async def test_server_error(self):
        mock_resp = MagicMock(status_code=502)
        with patch("deep_agent.aegra.mcp_health.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await _ping_mcp_server("test", "http://x/mcp", 5.0, False)

        assert result["status"] == "unreachable"

    async def test_timeout(self):
        with patch("deep_agent.aegra.mcp_health.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.TimeoutException("timeout")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await _ping_mcp_server("test", "http://x/mcp", 5.0, False)

        assert result["status"] == "timeout"

    async def test_connection_refused(self):
        with patch("deep_agent.aegra.mcp_health.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await _ping_mcp_server("test", "http://x/mcp", 5.0, False)

        assert result["status"] == "unreachable"
        assert "error" in result

    async def test_4xx_counts_as_healthy(self):
        mock_resp = MagicMock(status_code=405)
        with patch("deep_agent.aegra.mcp_health.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await _ping_mcp_server("test", "http://x/mcp", 5.0, False)

        assert result["status"] == "healthy"


# ── check_mcp_servers ────────────────────────────────────────────


class TestCheckMcpServers:
    async def test_all_healthy(self):
        healthy = {"status": "healthy", "latency_ms": 1.0, "http_status": 200}
        with (
            _mock_servers(TWO_SERVERS),
            patch(
                "deep_agent.aegra.mcp_health._ping_mcp_server",
                new_callable=AsyncMock,
                return_value=healthy,
            ),
            patch("deep_agent.aegra.mcp._get_mcp_breaker") as mock_breaker,
        ):
            mock_breaker.return_value.is_open = False
            result = await check_mcp_servers()

        assert result["status"] == "ok"
        assert result["healthy"] == 2
        assert result["total"] == 2
        assert "server-a" in result["servers"]
        assert "server-b" in result["servers"]

    async def test_one_of_two_down(self):
        """Partial failure → status is 'warning', not 'error'."""

        async def _ping_side_effect(name, url, timeout, ssl_verify):
            if name == "server-a":
                return {"status": "healthy", "latency_ms": 1.0, "http_status": 200}
            return {"status": "unreachable", "error": "connection refused"}

        with (
            _mock_servers(TWO_SERVERS),
            patch(
                "deep_agent.aegra.mcp_health._ping_mcp_server",
                side_effect=_ping_side_effect,
            ),
            patch("deep_agent.aegra.mcp._get_mcp_breaker") as mock_breaker,
        ):
            mock_breaker.return_value.is_open = False
            result = await check_mcp_servers()

        assert result["status"] == "warning"
        assert result["healthy"] == 1
        assert result["total"] == 2
        assert result["servers"]["server-a"]["status"] == "healthy"
        assert result["servers"]["server-b"]["status"] == "unreachable"

    async def test_all_down_still_warning_not_error(self):
        """All MCP servers down → 'warning' so agent reports degraded, not unhealthy."""
        down = {"status": "unreachable", "error": "connection refused"}
        with (
            _mock_servers(TWO_SERVERS),
            patch(
                "deep_agent.aegra.mcp_health._ping_mcp_server",
                new_callable=AsyncMock,
                return_value=down,
            ),
            patch("deep_agent.aegra.mcp._get_mcp_breaker") as mock_breaker,
        ):
            mock_breaker.return_value.is_open = False
            result = await check_mcp_servers()

        assert result["status"] == "warning"
        assert result["healthy"] == 0

    async def test_breaker_open(self):
        with (
            _mock_servers(TWO_SERVERS),
            patch("deep_agent.aegra.mcp._get_mcp_breaker") as mock_breaker,
        ):
            mock_breaker.return_value.is_open = True
            result = await check_mcp_servers()

        assert result["status"] == "warning"
        for srv in result["servers"].values():
            assert srv["status"] == "breaker-open"

    async def test_no_servers_enabled(self):
        disabled = {
            "x": {"url": "http://x/mcp", "enabled": False},
        }
        with _mock_servers(disabled):
            result = await check_mcp_servers()

        assert result["status"] == "skipped"

    async def test_cache_hit(self):
        healthy = {"status": "healthy", "latency_ms": 1.0, "http_status": 200}
        with (
            _mock_servers(TWO_SERVERS),
            patch(
                "deep_agent.aegra.mcp_health._ping_mcp_server",
                new_callable=AsyncMock,
                return_value=healthy,
            ) as mock_ping,
            patch("deep_agent.aegra.mcp._get_mcp_breaker") as mock_breaker,
        ):
            mock_breaker.return_value.is_open = False
            first = await check_mcp_servers()
            second = await check_mcp_servers()

        assert first is second
        assert mock_ping.await_count == 2  # only the first round (2 servers)


# ── check_llm_provider ──────────────────────────────────────────


class TestCheckLlmProvider:
    async def test_vllm_healthy(self):
        mock_settings = MagicMock()
        mock_settings.VLLM_BASE_URL = "http://vllm:8000/v1"
        mock_settings.VLLM_API_KEY = "EMPTY"

        mock_resp = MagicMock(status_code=200)
        with (
            patch("deep_agent.aegra.mcp_health.httpx.AsyncClient") as mock_cls,
            patch("deep_agent.aegra.mcp_health.settings", mock_settings),
        ):
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await check_llm_provider()

        assert result["status"] == "ok"
        assert result["provider"] == "vllm"

    async def test_vllm_unreachable(self):
        mock_settings = MagicMock()
        mock_settings.VLLM_BASE_URL = "http://vllm:8000/v1"
        mock_settings.VLLM_API_KEY = "EMPTY"

        with (
            patch("deep_agent.aegra.mcp_health.httpx.AsyncClient") as mock_cls,
            patch("deep_agent.aegra.mcp_health.settings", mock_settings),
        ):
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await check_llm_provider()

        assert result["status"] == "warning"
        assert result["provider"] == "vllm"

    async def test_vertex_ai_ok(self):
        mock_settings = MagicMock()
        mock_settings.VLLM_BASE_URL = ""

        with (
            patch("deep_agent.aegra.mcp_health.settings", mock_settings),
            patch(
                "deep_agent.aegra.mcp_health.get_service_account_credentials",
                return_value=(MagicMock(), "my-project"),
            ),
        ):
            result = await check_llm_provider()

        assert result["status"] == "ok"
        assert result["provider"] == "vertex_ai"
        assert result["project"] == "my-project"

    async def test_vertex_ai_no_creds(self):
        mock_settings = MagicMock()
        mock_settings.VLLM_BASE_URL = ""

        with (
            patch("deep_agent.aegra.mcp_health.settings", mock_settings),
            patch(
                "deep_agent.aegra.mcp_health.get_service_account_credentials",
                side_effect=Exception("no credentials"),
            ),
        ):
            result = await check_llm_provider()

        assert result["status"] == "warning"
        assert result["provider"] == "vertex_ai"

    async def test_cache_hit(self):
        mock_settings = MagicMock()
        mock_settings.VLLM_BASE_URL = ""

        with (
            patch("deep_agent.aegra.mcp_health.settings", mock_settings),
            patch(
                "deep_agent.aegra.mcp_health.get_service_account_credentials",
                return_value=(MagicMock(), "proj"),
            ) as mock_creds,
        ):
            first = await check_llm_provider()
            second = await check_llm_provider()

        assert first is second
        assert mock_creds.call_count == 1


# ── OTEL gauge emission ─────────────────────────────────────────


class TestOtelGauges:
    def test_gauge_noop_when_otel_disabled(self):
        mock_settings = MagicMock()
        mock_settings.ENABLE_OTEL_METRICS = False
        mock_settings.OTEL_EXPORTER_OTLP_ENDPOINT = ""

        with patch("deep_agent.src.settings.settings", mock_settings):
            mcp_health._ensure_gauges()

        assert mcp_health._mcp_health_gauge is None

    def test_gauge_created_when_otel_enabled(self):
        mock_settings = MagicMock()
        mock_settings.ENABLE_OTEL_METRICS = True
        mock_settings.OTEL_EXPORTER_OTLP_ENDPOINT = "http://otel:4317"

        mock_gauge = MagicMock()
        mock_meter = MagicMock()
        mock_meter.create_gauge.return_value = mock_gauge

        with (
            patch("deep_agent.aegra.mcp_health.settings", mock_settings),
            patch("opentelemetry.metrics.get_meter", return_value=mock_meter),
        ):
            mcp_health._ensure_gauges()

        assert mock_meter.create_gauge.call_count == 2

    def test_gauge_initialized_set_even_when_otel_disabled(self):
        """_gauge_initialized is True after _ensure_gauges even if OTEL is off."""
        mock_settings = MagicMock()
        mock_settings.ENABLE_OTEL_METRICS = False
        mock_settings.OTEL_EXPORTER_OTLP_ENDPOINT = ""

        with patch("deep_agent.aegra.mcp_health.settings", mock_settings):
            mcp_health._ensure_gauges()

        assert mcp_health._gauge_initialized is True
        assert mcp_health._mcp_health_gauge is None

        with patch("deep_agent.aegra.mcp_health.settings", mock_settings):
            mcp_health._ensure_gauges()

        assert mcp_health._mcp_health_gauge is None
