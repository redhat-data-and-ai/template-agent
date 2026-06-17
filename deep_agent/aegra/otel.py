"""OpenTelemetry instrumentation for the template agent.

Provides centralized telemetry with:
- OTLP exporter when enabled (via YAML or ENABLE_OTEL env var)
- InMemoryMetricReader (no-op) when disabled
- FastAPI auto-instrumentation for distributed tracing
- Conversation, streaming, and thread management metrics

Config resolution order (highest wins):
    1. Environment variables (ENABLE_OTEL, OTEL_EXPORTER_OTLP_ENDPOINT, ...)
    2. observability.yaml otel: section
    3. Pydantic model defaults
"""

import os
import threading
import time
from typing import Any, Optional

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.aggregation import (
    ExplicitBucketHistogramAggregation,
)
from opentelemetry.sdk.metrics.view import View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

SERVICE_NAME = "template-agent"
SERVICE_VERSION = "1.0.0"

DURATION_BUCKETS = [
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    15.0,
    30.0,
    60.0,
    120.0,
    300.0,
]

TTFT_BUCKETS = [
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
]

MESSAGES_COUNT_BUCKETS = [
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
]

# ---------------------------------------------------------------------------
# Dual tracing architecture
# ---------------------------------------------------------------------------
# This agent has TWO independent observability systems:
#
# 1. OpenTelemetry (this module) — metrics + distributed tracing.
#    Uses an SDK TracerProvider stored in ``_tracer_provider`` AND set as
#    the global provider (required by FastAPI auto-instrumentation).
#    Custom spans are created via ``get_tracer()`` which reads from
#    ``_tracer_provider`` directly, not the global.
#
# 2. Langfuse (telemetry.py) — LLM-specific tracing via LangChain's
#    ``register_configure_hook`` + ``CallbackHandler``.  Langfuse does
#    NOT use the OTEL TracerProvider; it has its own SDK.
#
# The two systems coexist without conflict.  Langfuse traces LLM calls
# with prompt/completion detail; OTEL traces infrastructure spans
# (graph builds, MCP connections, memory ops) and exports metrics.
# ---------------------------------------------------------------------------

_tracer_provider: Optional[TracerProvider] = None
_meter: Optional[metrics.Meter] = None
_metrics_container: Optional["MetricsContainer"] = None
_snapshot_reader: Optional[Any] = None
_initialized: bool = False
_otel_enabled: bool = False

_threads_active_tracked: set[str] = set()
_threads_active_lock = threading.Lock()


class MetricsContainer:
    """Container for all template agent OpenTelemetry metric instruments."""

    def __init__(self, meter: metrics.Meter) -> None:
        """Create all metric instruments on the given meter."""
        prefix = "template_agent"

        self.conversations_total = meter.create_counter(
            name=f"{prefix}_conversations_total",
            description="Total conversations by status",
            unit="1",
        )
        self.messages_total = meter.create_counter(
            name=f"{prefix}_messages_total",
            description="Messages sent/received",
            unit="1",
        )
        self.conversation_duration_seconds = meter.create_histogram(
            name=f"{prefix}_conversation_duration_seconds",
            description="Time from conversation start to completion",
            unit="s",
        )
        self.active_conversations = meter.create_up_down_counter(
            name=f"{prefix}_active_conversations",
            description="Currently active conversations",
            unit="1",
        )

        self.stream_tokens_total = meter.create_counter(
            name=f"{prefix}_stream_tokens_total",
            description="Tokens streamed to clients",
            unit="1",
        )
        self.stream_duration_seconds = meter.create_histogram(
            name=f"{prefix}_stream_duration_seconds",
            description="Time to complete stream",
            unit="s",
        )
        self.stream_errors_total = meter.create_counter(
            name=f"{prefix}_stream_errors_total",
            description="Stream failures by type",
            unit="1",
        )
        self.time_to_first_token_seconds = meter.create_histogram(
            name=f"{prefix}_time_to_first_token_seconds",
            description="Latency until first token",
            unit="s",
        )

        self.threads_created_total = meter.create_counter(
            name=f"{prefix}_threads_created_total",
            description="New threads created",
            unit="1",
        )
        self.threads_active = meter.create_up_down_counter(
            name=f"{prefix}_threads_active",
            description="Currently active threads",
            unit="1",
        )
        self.threads_deleted_total = meter.create_counter(
            name=f"{prefix}_threads_deleted_total",
            description="Threads deleted",
            unit="1",
        )
        self.thread_messages_count = meter.create_histogram(
            name=f"{prefix}_thread_messages_count",
            description="Messages per thread",
            unit="1",
        )

        # ponytail: seed all instruments so /api/metrics shows them from startup.
        # OTEL SDK only reports instruments after first measurement.
        self.conversations_total.add(0)
        self.messages_total.add(0)
        self.conversation_duration_seconds.record(0)
        self.active_conversations.add(0)
        self.stream_tokens_total.add(0)
        self.stream_duration_seconds.record(0)
        self.stream_errors_total.add(0)
        self.time_to_first_token_seconds.record(0)
        self.threads_created_total.add(0)
        self.threads_active.add(0)
        self.threads_deleted_total.add(0)
        self.thread_messages_count.record(0)


def _resolve_service_name() -> str:
    """Resolve service name from agent config, fall back to default."""
    try:
        from deep_agent.src.agent.config import agent_config

        return agent_config.get_name()
    except Exception:
        return SERVICE_NAME


def _resolve_config() -> tuple[bool, str, bool, int, bool]:
    """Resolve OTEL config: env vars override YAML defaults.

    Returns:
        (enabled, endpoint, insecure, export_interval_ms, auto_instrument)
    """
    try:
        from deep_agent.src.agent.config import agent_config

        cfg = agent_config.get_otel_config()
    except Exception:
        from deep_agent.src.agent.config.otel import OtelFileConfig

        cfg = OtelFileConfig()

    enabled = os.environ.get("ENABLE_OTEL", str(cfg.enabled)).lower() == "true"
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", cfg.exporter.endpoint)
    insecure = (
        os.environ.get(
            "OTEL_EXPORTER_OTLP_INSECURE", str(cfg.exporter.insecure)
        ).lower()
        == "true"
    )
    export_interval = int(
        os.environ.get(
            "OTEL_METRIC_EXPORT_INTERVAL", str(cfg.metrics.export_interval_ms)
        )
    )
    auto_instrument = cfg.tracing.fastapi_auto_instrument

    return enabled, endpoint, insecure, export_interval, auto_instrument


def _build_resource() -> Resource:
    """Build the OTel resource with service metadata."""
    environment = os.environ.get("ENVIRONMENT", "dev")
    version = os.environ.get("APPLICATION_VERSION", SERVICE_VERSION)
    instance_id = os.environ.get("HOSTNAME", "local")

    return Resource.create(
        {
            "service.name": _resolve_service_name(),
            "service.version": version,
            "service.instance.id": instance_id,
            "deployment.environment": environment,
        }
    )


def _create_histogram_views() -> list[View]:
    """Create histogram bucket views for metrics."""
    prefix = "template_agent"
    return [
        View(
            instrument_name=f"{prefix}_conversation_duration_seconds",
            aggregation=ExplicitBucketHistogramAggregation(boundaries=DURATION_BUCKETS),
        ),
        View(
            instrument_name=f"{prefix}_stream_duration_seconds",
            aggregation=ExplicitBucketHistogramAggregation(boundaries=DURATION_BUCKETS),
        ),
        View(
            instrument_name=f"{prefix}_time_to_first_token_seconds",
            aggregation=ExplicitBucketHistogramAggregation(boundaries=TTFT_BUCKETS),
        ),
        View(
            instrument_name=f"{prefix}_thread_messages_count",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=MESSAGES_COUNT_BUCKETS,
            ),
        ),
    ]


def initialize_telemetry() -> None:
    """Initialize OpenTelemetry metrics and tracing.

    Reads config from observability.yaml with env var overrides.
    When disabled (default), uses InMemoryMetricReader and NoOpTracerProvider.
    When enabled, configures OTLP gRPC exporters for both metrics and traces.

    The TracerProvider is both stored in ``_tracer_provider`` (for
    ``get_tracer()``) and set as the global (for FastAPI auto-instrumentation).
    """
    global _meter, _metrics_container, _initialized, _otel_enabled, _tracer_provider
    global _snapshot_reader

    if _initialized:
        return

    enabled, endpoint, insecure, export_interval, _ = _resolve_config()
    _otel_enabled = enabled

    resource = _build_resource()
    views = _create_histogram_views()

    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    # Always keep an InMemoryMetricReader so /api/metrics can read values back.
    _snapshot_reader = InMemoryMetricReader()

    if not enabled:
        logger.info(
            "OTEL disabled (ENABLE_OTEL not true) — metrics and traces are in-memory"
        )

        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[_snapshot_reader],
            views=views,
        )
        tracer_provider = TracerProvider(resource=resource)
    else:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        logger.info(
            "OTEL enabled — exporting to %s (interval=%sms)",
            endpoint,
            export_interval,
        )

        otlp_metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=insecure)
        otlp_reader = PeriodicExportingMetricReader(
            otlp_metric_exporter,
            export_interval_millis=export_interval,
        )
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[otlp_reader, _snapshot_reader],
            views=views,
        )

        otlp_span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_span_exporter))

    _tracer_provider = tracer_provider

    metrics.set_meter_provider(meter_provider)
    # Set global TracerProvider — required by FastAPI auto-instrumentation
    # which reads from trace.get_tracer_provider(). Custom spans use
    # get_tracer() which reads _tracer_provider directly.
    trace.set_tracer_provider(tracer_provider)

    service_name = _resolve_service_name()
    _meter = meter_provider.get_meter(service_name, SERVICE_VERSION)
    _metrics_container = MetricsContainer(_meter)
    _initialized = True


def instrument_fastapi(app: Any) -> None:
    """Auto-instrument a FastAPI app for distributed tracing.

    Only instruments if OTEL is initialized and auto-instrumentation
    is enabled in config. Safe to call before initialize_telemetry() —
    instrumentation picks up the global TracerProvider lazily.
    """
    _, _, _, _, auto_instrument = _resolve_config()
    if not auto_instrument:
        logger.debug("FastAPI auto-instrumentation disabled in config")
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI auto-instrumentation enabled")
    except ImportError:
        logger.warning("opentelemetry-instrumentation-fastapi not installed — skipping")
    except Exception:
        logger.warning("FastAPI instrumentation failed", exc_info=True)


def shutdown_telemetry() -> None:
    """Flush and shut down both meter and tracer providers."""
    global _initialized, _tracer_provider

    if _tracer_provider is not None and hasattr(_tracer_provider, "shutdown"):
        _tracer_provider.shutdown()
    _tracer_provider = None

    meter_provider = metrics.get_meter_provider()
    if hasattr(meter_provider, "shutdown"):
        meter_provider.shutdown()

    reset_thread_active_tracking()
    _initialized = False


def get_metrics() -> Optional[MetricsContainer]:
    """Return the global metrics container, or None if not initialized."""
    return _metrics_container


def get_tracer(name: str = "template-agent") -> trace.Tracer:
    """Return a tracer from the module-owned TracerProvider.

    Uses ``_tracer_provider`` (set during ``initialize_telemetry``) rather
    than the global provider, keeping ownership explicit.  If telemetry has
    not been initialised yet, falls back to the global (which may be a
    no-op ``ProxyTracerProvider``).

    Args:
        name: Instrumentation scope name for the tracer.

    Returns:
        An OTEL ``Tracer`` instance.
    """
    if _tracer_provider is not None:
        return _tracer_provider.get_tracer(name)
    return trace.get_tracer(name)


def is_tracing_enabled() -> bool:
    """Return True if OTEL has been initialised and is enabled."""
    return _initialized and _otel_enabled


def get_metrics_snapshot() -> dict[str, Any]:
    """Read current metric values from the InMemoryMetricReader.

    Returns a flat dict keyed by metric name. Counters/UpDownCounters
    are summed across all attribute sets. Histograms aggregate count
    and sum across all attribute sets.
    """
    if _snapshot_reader is None:
        return {}

    data = _snapshot_reader.get_metrics_data()
    result: dict[str, Any] = {}

    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                name = metric.name
                points = list(metric.data.data_points)
                if not points:
                    continue

                if hasattr(points[0], "bucket_counts"):
                    total_count = sum(pt.count for pt in points)
                    total_sum = sum(pt.sum for pt in points)
                    result[name] = {
                        "count": total_count,
                        "sum": round(total_sum, 3),
                    }
                else:
                    result[name] = sum(pt.value for pt in points)

    return result


# ---------------------------------------------------------------------------
# Helper instrumentation functions
# ---------------------------------------------------------------------------


def _attrs(extra: Optional[dict[str, Any]] = None) -> dict[str, str]:
    """Merge optional extra attributes, stringifying values for OTel."""
    if not extra:
        return {}
    return {k: str(v) for k, v in extra.items() if v is not None}


def _release_thread_active_if_tracked(thread_id: str) -> bool:
    with _threads_active_lock:
        if thread_id in _threads_active_tracked:
            _threads_active_tracked.discard(thread_id)
            return True
        return False


def reset_thread_active_tracking() -> None:
    """Clear in-process thread tracking (for tests and shutdown)."""
    with _threads_active_lock:
        _threads_active_tracked.clear()


def record_conversation_started(
    *,
    status: str = "started",
    attributes: Optional[dict[str, Any]] = None,
) -> float:
    """Record conversation start. Returns monotonic timestamp for duration."""
    m = get_metrics()
    if m:
        base_attrs = _attrs(attributes)
        m.conversations_total.add(1, {"status": status, **base_attrs})
        m.active_conversations.add(1, base_attrs)
    return time.monotonic()


def record_conversation_completed(
    start_mono: float,
    *,
    status: str = "completed",
    attributes: Optional[dict[str, Any]] = None,
) -> None:
    """Record conversation completion with duration."""
    m = get_metrics()
    if m:
        base_attrs = _attrs(attributes)
        merged = {"status": status, **base_attrs}
        duration = time.monotonic() - start_mono
        m.conversations_total.add(1, merged)
        m.active_conversations.add(-1, base_attrs)
        m.conversation_duration_seconds.record(duration, merged)


def record_message_sent(
    *,
    direction: str = "sent",
    message_type: str = "human",
    attributes: Optional[dict[str, Any]] = None,
) -> None:
    """Record a message sent or received."""
    m = get_metrics()
    if m:
        merged = {
            "direction": direction,
            "message_type": message_type,
            **_attrs(attributes),
        }
        m.messages_total.add(1, merged)


def record_stream_started() -> float:
    """Record stream start. Returns monotonic timestamp."""
    return time.monotonic()


def record_first_token(
    stream_start_mono: float,
    *,
    attributes: Optional[dict[str, Any]] = None,
) -> None:
    """Record time-to-first-token from stream start."""
    m = get_metrics()
    if m:
        ttft = time.monotonic() - stream_start_mono
        m.time_to_first_token_seconds.record(ttft, _attrs(attributes))


def record_stream_completed(
    stream_start_mono: float,
    token_count: int,
    *,
    attributes: Optional[dict[str, Any]] = None,
) -> None:
    """Record stream completion with duration and token count."""
    m = get_metrics()
    if m:
        merged = _attrs(attributes)
        duration = time.monotonic() - stream_start_mono
        m.stream_duration_seconds.record(duration, merged)
        m.stream_tokens_total.add(token_count, merged)


def record_stream_error(
    *,
    error_type: str = "unknown",
    attributes: Optional[dict[str, Any]] = None,
) -> None:
    """Record a stream error."""
    m = get_metrics()
    if m:
        merged = {"error_type": error_type, **_attrs(attributes)}
        m.stream_errors_total.add(1, merged)


def record_thread_created(
    *,
    attributes: Optional[dict[str, Any]] = None,
) -> None:
    """Record thread creation with active tracking."""
    m = get_metrics()
    if not m:
        return
    merged = _attrs(attributes)
    thread_id = merged.get("thread_id")
    with _threads_active_lock:
        if thread_id and thread_id in _threads_active_tracked:
            return
        if thread_id:
            _threads_active_tracked.add(thread_id)
    m.threads_created_total.add(1, merged)
    m.threads_active.add(1, merged)


def record_thread_deleted(
    *,
    count: int = 1,
    attributes: Optional[dict[str, Any]] = None,
) -> None:
    """Record thread deletion. Decrements active only if previously tracked."""
    m = get_metrics()
    if not m:
        return
    merged = _attrs(attributes)
    m.threads_deleted_total.add(count, merged)
    if count != 1:
        return
    tid = merged.get("thread_id")
    if not tid:
        return
    if _release_thread_active_if_tracked(str(tid)):
        m.threads_active.add(-1, merged)


def record_threads_deleted_bulk(
    deleted_thread_ids: list[str],
    *,
    attributes: Optional[dict[str, Any]] = None,
) -> None:
    """Record bulk thread deletion with per-ID active tracking."""
    m = get_metrics()
    if not m or not deleted_thread_ids:
        return
    base = _attrs(attributes)
    m.threads_deleted_total.add(len(deleted_thread_ids), base)
    for tid in deleted_thread_ids:
        row = {**base, "thread_id": tid}
        if _release_thread_active_if_tracked(tid):
            m.threads_active.add(-1, row)


def record_thread_messages(
    message_count: int,
    *,
    attributes: Optional[dict[str, Any]] = None,
) -> None:
    """Record the final message count for a thread."""
    m = get_metrics()
    if m:
        m.thread_messages_count.record(message_count, _attrs(attributes))
