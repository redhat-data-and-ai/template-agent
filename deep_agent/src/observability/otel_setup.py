"""OTLP metrics (otel-gateway) and traces bootstrap."""

from __future__ import annotations

from typing import Any

_fastapi_instrumented = False
_metrics_initialized = False
_traces_initialized = False
_service_resource = None


def _service_resource_for(settings: Any) -> Any:
    """Return a shared OTEL resource for metrics and traces providers."""
    global _service_resource  # noqa: PLW0603

    if _service_resource is None:
        from opentelemetry.sdk.resources import Resource

        _service_resource = Resource.create(
            {"service.name": settings.OTEL_SERVICE_NAME}
        )
    return _service_resource


def _otlp_grpc_exporter_kwargs(endpoint: str, settings: Any) -> dict[str, Any]:
    """Build OTLP/gRPC exporter kwargs from a config endpoint string."""
    raw = endpoint.strip()
    kwargs: dict[str, Any] = {}
    lower = raw.lower()
    if lower.startswith("https://"):
        kwargs["endpoint"] = raw[len("https://") :]
    elif lower.startswith("http://"):
        kwargs["endpoint"] = raw[len("http://") :]
        kwargs["insecure"] = True
    else:
        kwargs["endpoint"] = raw
        kwargs["insecure"] = True
    token = (getattr(settings, "OTEL_AUTH_TOKEN", None) or "").strip()
    if token and not token.startswith("<"):
        kwargs["headers"] = (("authorization", f"Bearer {token}"),)
    return kwargs


def _instrument_fastapi(app: Any, log: Any) -> None:
    """HTTP server metrics + traces (needs MeterProvider and/or TracerProvider set first)."""
    global _fastapi_instrumented  # noqa: PLW0603
    if _fastapi_instrumented:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        _fastapi_instrumented = True
    except Exception:
        log.warning("template_agent_fastapi_instrument_failed", exc_info=True)


def setup_otel_metrics(settings: Any, log: Any) -> None:
    """Export OTLP metrics to OTEL_EXPORTER_OTLP_ENDPOINT when enabled."""
    global _metrics_initialized  # noqa: PLW0603

    if _metrics_initialized:
        return
    if not settings.ENABLE_OTEL_METRICS or not settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        return

    try:
        from opentelemetry import metrics
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        resource = _service_resource_for(settings)
        exporter = OTLPMetricExporter(
            **_otlp_grpc_exporter_kwargs(settings.OTEL_EXPORTER_OTLP_ENDPOINT, settings)
        )
        reader = PeriodicExportingMetricReader(
            exporter,
            export_interval_millis=settings.OTEL_METRIC_EXPORT_INTERVAL_MILLIS,
        )
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
        _metrics_initialized = True
        log.info("template_agent_otel_metrics_export_enabled")
    except Exception:
        log.warning("template_agent_otel_metrics_export_failed", exc_info=True)


def setup_otel_traces(app: Any, settings: Any, log: Any) -> None:
    """Export OTLP traces when enabled; instrument FastAPI when metrics or traces on."""
    global _traces_initialized  # noqa: PLW0603

    traces_endpoint = settings.resolved_otel_traces_endpoint()
    metrics_on = bool(
        settings.ENABLE_OTEL_METRICS and settings.OTEL_EXPORTER_OTLP_ENDPOINT
    )
    traces_on = bool(settings.otel_traces_active() and traces_endpoint)

    if not metrics_on and not traces_on:
        return

    if traces_on and not _traces_initialized:
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = _service_resource_for(settings)
            provider = TracerProvider(resource=resource)
            processor = BatchSpanProcessor(
                OTLPSpanExporter(
                    **_otlp_grpc_exporter_kwargs(traces_endpoint, settings)
                )
            )
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            _traces_initialized = True
            log.info("template_agent_otel_tracing_enabled")
        except Exception:
            log.warning("template_agent_otel_tracing_failed", exc_info=True)

    if metrics_on or traces_on:
        _instrument_fastapi(app, log)
