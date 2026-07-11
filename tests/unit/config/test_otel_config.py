"""Unit tests for OtelFileConfig Pydantic models."""

import pytest
from pydantic import ValidationError

from deep_agent.src.agent.config.otel import (
    OtelExporterConfig,
    OtelFileConfig,
    OtelMetricsConfig,
    OtelTracingConfig,
)


class TestOtelExporterConfig:
    """Test OtelExporterConfig defaults and validation."""

    def test_defaults(self):
        config = OtelExporterConfig()
        assert config.endpoint == "http://localhost:4317"
        assert config.insecure is True

    def test_custom_values(self):
        config = OtelExporterConfig(
            endpoint="https://collector.prod:4317",
            insecure=False,
        )
        assert config.endpoint == "https://collector.prod:4317"
        assert config.insecure is False

    def test_from_dict(self):
        config = OtelExporterConfig.model_validate(
            {"endpoint": "http://otel:4317", "insecure": False}
        )
        assert config.endpoint == "http://otel:4317"
        assert config.insecure is False


class TestOtelMetricsConfig:
    """Test OtelMetricsConfig defaults and validation."""

    def test_default_interval(self):
        config = OtelMetricsConfig()
        assert config.export_interval_ms == 5000

    def test_custom_interval(self):
        config = OtelMetricsConfig(export_interval_ms=10000)
        assert config.export_interval_ms == 10000

    def test_minimum_interval_boundary(self):
        config = OtelMetricsConfig(export_interval_ms=1000)
        assert config.export_interval_ms == 1000

    def test_maximum_interval_boundary(self):
        config = OtelMetricsConfig(export_interval_ms=60000)
        assert config.export_interval_ms == 60000

    def test_rejects_interval_below_minimum(self):
        with pytest.raises(ValidationError, match="greater than or equal to 1000"):
            OtelMetricsConfig(export_interval_ms=999)

    def test_rejects_interval_above_maximum(self):
        with pytest.raises(ValidationError, match="less than or equal to 60000"):
            OtelMetricsConfig(export_interval_ms=60001)


class TestOtelTracingConfig:
    """Test OtelTracingConfig defaults."""

    def test_auto_instrument_default_true(self):
        config = OtelTracingConfig()
        assert config.fastapi_auto_instrument is True

    def test_disable_auto_instrument(self):
        config = OtelTracingConfig(fastapi_auto_instrument=False)
        assert config.fastapi_auto_instrument is False


class TestOtelFileConfig:
    """Test top-level OtelFileConfig model."""

    def test_defaults(self):
        config = OtelFileConfig()
        assert config.enabled is False
        assert config.exporter.endpoint == "http://localhost:4317"
        assert config.exporter.insecure is True
        assert config.metrics.export_interval_ms == 5000
        assert config.tracing.fastapi_auto_instrument is True

    def test_enabled_flag(self):
        config = OtelFileConfig(enabled=True)
        assert config.enabled is True

    def test_from_dict(self):
        """Parse from a dict matching the YAML structure."""
        config = OtelFileConfig.model_validate(
            {
                "enabled": True,
                "exporter": {
                    "endpoint": "http://collector:4317",
                    "insecure": False,
                },
                "metrics": {
                    "export_interval_ms": 15000,
                },
                "tracing": {
                    "fastapi_auto_instrument": False,
                },
            }
        )
        assert config.enabled is True
        assert config.exporter.endpoint == "http://collector:4317"
        assert config.exporter.insecure is False
        assert config.metrics.export_interval_ms == 15000
        assert config.tracing.fastapi_auto_instrument is False

    def test_from_empty_dict(self):
        """Empty dict should produce all defaults (matches observability.yaml loading)."""
        config = OtelFileConfig.model_validate({})
        assert config.enabled is False
        assert config.exporter.endpoint == "http://localhost:4317"
        assert config.metrics.export_interval_ms == 5000
        assert config.tracing.fastapi_auto_instrument is True

    def test_partial_dict(self):
        """Partial dict should fill in defaults for missing fields."""
        config = OtelFileConfig.model_validate({"enabled": True})
        assert config.enabled is True
        assert config.exporter.endpoint == "http://localhost:4317"
        assert config.metrics.export_interval_ms == 5000

    def test_nested_validation_propagates(self):
        """Invalid nested config should raise ValidationError."""
        with pytest.raises(ValidationError):
            OtelFileConfig.model_validate({"metrics": {"export_interval_ms": 500}})
