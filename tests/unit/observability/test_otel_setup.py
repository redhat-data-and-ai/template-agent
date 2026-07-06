"""Unit tests for platform-style OTEL bootstrap."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from deep_agent.src.observability import otel_setup


def test_setup_otel_metrics_skips_when_disabled() -> None:
    settings = MagicMock()
    settings.ENABLE_OTEL_METRICS = False
    settings.OTEL_EXPORTER_OTLP_ENDPOINT = "otel-gateway:4327"
    log = MagicMock()

    with patch("opentelemetry.metrics.set_meter_provider") as set_provider:
        otel_setup.setup_otel_metrics(settings, log)

    set_provider.assert_not_called()


def test_setup_otel_traces_skips_when_both_disabled() -> None:
    settings = MagicMock()
    settings.ENABLE_OTEL_METRICS = False
    settings.OTEL_EXPORTER_OTLP_ENDPOINT = ""
    settings.otel_traces_active.return_value = False
    settings.resolved_otel_traces_endpoint.return_value = ""
    log = MagicMock()
    app = MagicMock()

    with patch.object(otel_setup, "_instrument_fastapi") as instrument:
        otel_setup.setup_otel_traces(app, settings, log)

    instrument.assert_not_called()


def test_setup_otel_metrics_is_idempotent() -> None:
    otel_setup._metrics_initialized = False
    settings = MagicMock()
    settings.ENABLE_OTEL_METRICS = True
    settings.OTEL_EXPORTER_OTLP_ENDPOINT = "otel-gateway:4327"
    settings.OTEL_SERVICE_NAME = "template-agent"
    settings.OTEL_METRIC_EXPORT_INTERVAL_MILLIS = 10000
    settings.OTEL_AUTH_TOKEN = ""
    log = MagicMock()

    with patch("opentelemetry.metrics.set_meter_provider") as set_provider:
        otel_setup.setup_otel_metrics(settings, log)
        otel_setup.setup_otel_metrics(settings, log)

    set_provider.assert_called_once()
    otel_setup._metrics_initialized = False
