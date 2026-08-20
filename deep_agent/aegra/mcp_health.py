"""Per-server MCP and LLM provider health checks with caching and OTEL gauges.

Pings each MCP server defined in mcp.json and the configured LLM provider.
Results are cached to avoid hammering dependencies on every probe.

Emits ``mcp_server_health`` and ``llm_provider_health`` OTEL gauges when
metrics export is enabled.
"""

import asyncio
import threading
import time
from typing import Any

import httpx

from deep_agent.src.settings import settings
from deep_agent.utils.google_creds import get_service_account_credentials
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_HEALTH_CACHE_TTL: float = 30.0
_HEALTH_CHECK_TIMEOUT: float = 5.0

_cached_mcp_health: dict[str, Any] | None = None
_cached_mcp_health_ts: float = 0.0
_cached_llm_health: dict[str, Any] | None = None
_cached_llm_health_ts: float = 0.0

_gauge_initialized = False
_mcp_health_gauge: Any = None
_llm_health_gauge: Any = None
_gauge_lock = threading.Lock()


def _ensure_gauges() -> None:
    global _gauge_initialized, _mcp_health_gauge, _llm_health_gauge  # noqa: PLW0603

    if _gauge_initialized:
        return

    with _gauge_lock:
        if _gauge_initialized:
            return  # type: ignore[unreachable]
        try:
            if (
                not settings.ENABLE_OTEL_METRICS
                or not settings.OTEL_EXPORTER_OTLP_ENDPOINT
            ):
                return

            from opentelemetry import metrics

            meter = metrics.get_meter("template-agent.health")
            _mcp_health_gauge = meter.create_gauge(
                "mcp_server_health",
                description="MCP server health (1=healthy, 0=unhealthy)",
            )
            _llm_health_gauge = meter.create_gauge(
                "llm_provider_health",
                description="LLM provider health (1=healthy, 0=unhealthy)",
            )
        except Exception:
            logger.warning("mcp_health_otel_gauge_init_failed", exc_info=True)
        finally:
            _gauge_initialized = True


def _emit_mcp_gauge(server_name: str, healthy: bool) -> None:
    _ensure_gauges()
    if _mcp_health_gauge is not None:
        _mcp_health_gauge.set(1 if healthy else 0, {"mcp.server": server_name})


def _emit_llm_gauge(provider: str, healthy: bool) -> None:
    _ensure_gauges()
    if _llm_health_gauge is not None:
        _llm_health_gauge.set(1 if healthy else 0, {"llm.provider": provider})


async def _ping_mcp_server(
    name: str, url: str, timeout: float, ssl_verify: bool
) -> dict[str, Any]:
    """Ping a single MCP server. Any non-5xx response means reachable."""
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(verify=ssl_verify, timeout=timeout) as client:
            resp = await client.get(url)
        latency_ms = (time.monotonic() - t0) * 1000
        healthy = resp.status_code < 500
        status = "healthy" if healthy else "unreachable"
        _emit_mcp_gauge(name, healthy)
        return {
            "status": status,
            "latency_ms": round(latency_ms, 1),
            "http_status": resp.status_code,
        }
    except httpx.TimeoutException:
        _emit_mcp_gauge(name, False)
        return {
            "status": "timeout",
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }
    except Exception as exc:
        _emit_mcp_gauge(name, False)
        return {"status": "unreachable", "error": str(exc)[:200]}


async def check_mcp_servers() -> dict[str, Any]:
    """Check health of all configured MCP servers (cached).

    The aggregate status is always ``"ok"`` or ``"warning"`` — never
    ``"error"`` — so the overall agent health becomes *degraded*, not
    *unhealthy*, when MCP servers are down.
    """
    global _cached_mcp_health, _cached_mcp_health_ts  # noqa: PLW0603

    now = time.time()
    if (
        _cached_mcp_health is not None
        and (now - _cached_mcp_health_ts) < _HEALTH_CACHE_TTL
    ):
        return _cached_mcp_health

    try:
        from deep_agent.src.agent.config import agent_config

        servers: dict[str, dict[str, Any]] = agent_config.get_mcp_servers()
    except Exception as exc:
        result: dict[str, Any] = {
            "status": "warning",
            "error": f"Failed to load MCP config: {exc}",
        }
        _cached_mcp_health = result
        _cached_mcp_health_ts = now
        return result

    enabled = {k: v for k, v in servers.items() if v.get("enabled", False)}

    if not enabled:
        result = {"status": "skipped", "reason": "no MCP servers enabled"}
        _cached_mcp_health = result
        _cached_mcp_health_ts = now
        return result

    breaker_open = False
    try:
        from deep_agent.aegra.mcp import _get_mcp_breaker

        breaker_open = _get_mcp_breaker().is_open
    except Exception:
        pass

    per_server: dict[str, Any] = {}

    if breaker_open:
        for name in enabled:
            per_server[name] = {"status": "breaker-open"}
            _emit_mcp_gauge(name, False)
    else:
        names = list(enabled.keys())
        pings = [
            _ping_mcp_server(
                name=name,
                url=cfg["url"],
                timeout=min(cfg.get("timeout", 30), _HEALTH_CHECK_TIMEOUT),
                ssl_verify=cfg.get("ssl_verify", True),
            )
            for name, cfg in enabled.items()
        ]
        results = await asyncio.gather(*pings)
        for name, res in zip(names, results):
            per_server[name] = res

    statuses = [v.get("status") for v in per_server.values()]
    healthy_count = sum(1 for s in statuses if s == "healthy")
    total = len(statuses)

    aggregate = "ok" if healthy_count == total else "warning"

    result = {
        "status": aggregate,
        "servers": per_server,
        "healthy": healthy_count,
        "total": total,
    }
    _cached_mcp_health = result
    _cached_mcp_health_ts = now
    return result


async def check_llm_provider() -> dict[str, Any]:
    """Check LLM provider reachability (cached).

    vLLM / OpenAI-compatible: ``GET /models`` on the base URL.
    Vertex AI: verify that service-account credentials are loadable.

    Returns ``"warning"`` on failure so the agent reports *degraded*.
    """
    global _cached_llm_health, _cached_llm_health_ts  # noqa: PLW0603

    now = time.time()
    if (
        _cached_llm_health is not None
        and (now - _cached_llm_health_ts) < _HEALTH_CACHE_TTL
    ):
        return _cached_llm_health

    if settings.VLLM_BASE_URL:
        result = await _check_vllm(settings)
    else:
        result = _check_vertex_ai()

    _cached_llm_health = result
    _cached_llm_health_ts = now
    return result


async def _check_vllm(settings: Any) -> dict[str, Any]:
    base = settings.VLLM_BASE_URL.rstrip("/")
    headers: dict[str, str] = {}
    if settings.VLLM_API_KEY and settings.VLLM_API_KEY != "EMPTY":
        headers["Authorization"] = f"Bearer {settings.VLLM_API_KEY}"

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_CHECK_TIMEOUT) as client:
            resp = await client.get(f"{base}/models", headers=headers)
        latency_ms = (time.monotonic() - t0) * 1000
        healthy = resp.status_code < 500
        _emit_llm_gauge("vllm", healthy)
        return {
            "status": "ok" if healthy else "warning",
            "provider": "vllm",
            "endpoint": base,
            "latency_ms": round(latency_ms, 1),
        }
    except httpx.TimeoutException:
        _emit_llm_gauge("vllm", False)
        return {"status": "warning", "provider": "vllm", "error": "timeout"}
    except Exception as exc:
        _emit_llm_gauge("vllm", False)
        return {"status": "warning", "provider": "vllm", "error": str(exc)[:200]}


def _check_vertex_ai() -> dict[str, Any]:
    try:
        _credentials, project = get_service_account_credentials()
        _emit_llm_gauge("vertex_ai", True)
        return {"status": "ok", "provider": "vertex_ai", "project": project}
    except Exception as exc:
        _emit_llm_gauge("vertex_ai", False)
        return {"status": "warning", "provider": "vertex_ai", "error": str(exc)[:200]}


def invalidate_health_cache() -> None:
    """Clear cached health check results."""
    global _cached_mcp_health, _cached_mcp_health_ts  # noqa: PLW0603
    global _cached_llm_health, _cached_llm_health_ts  # noqa: PLW0603
    _cached_mcp_health = None
    _cached_mcp_health_ts = 0.0
    _cached_llm_health = None
    _cached_llm_health_ts = 0.0
