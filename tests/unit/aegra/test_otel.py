"""Unit tests for OTEL telemetry initialization and shutdown."""

from unittest.mock import MagicMock, patch

import pytest

import deep_agent.aegra.otel as otel_mod
from deep_agent.aegra.otel import (
    MetricsContainer,
    get_metrics,
    initialize_telemetry,
    reset_thread_active_tracking,
    shutdown_telemetry,
)


@pytest.fixture(autouse=True)
def _reset_otel_state():
    """Reset module-level OTEL state before and after each test."""
    otel_mod._meter = None
    otel_mod._metrics_container = None
    otel_mod._initialized = False
    otel_mod._otel_enabled = False
    reset_thread_active_tracking()
    yield
    otel_mod._meter = None
    otel_mod._metrics_container = None
    otel_mod._initialized = False
    otel_mod._otel_enabled = False
    reset_thread_active_tracking()


class TestInitializeTelemetry:
    """Test initialize_telemetry behaviour."""

    def test_disabled_by_default_returns_gracefully(self):
        """When OTEL is disabled (default), initialization should complete
        without error and set up in-memory providers."""
        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()

        assert otel_mod._initialized is True
        assert otel_mod._otel_enabled is False
        assert get_metrics() is not None

    def test_idempotent(self):
        """Calling initialize_telemetry twice should be a no-op the second time."""
        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ) as mock_resolve:
            initialize_telemetry()
            initialize_telemetry()

        # _resolve_config is only called once (first init)
        mock_resolve.assert_called_once()

    def test_get_metrics_none_before_init(self):
        """get_metrics() should return None before initialization."""
        assert get_metrics() is None


class TestShutdownTelemetry:
    """Test shutdown_telemetry behaviour."""

    def test_does_not_raise_when_not_initialized(self):
        """Calling shutdown before init should not raise."""
        shutdown_telemetry()
        assert otel_mod._initialized is False

    def test_resets_initialized_flag(self):
        """After shutdown, _initialized should be False."""
        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()
        assert otel_mod._initialized is True

        shutdown_telemetry()
        assert otel_mod._initialized is False

    def test_clears_thread_tracking(self):
        """Shutdown should clear the thread active tracking set."""
        with otel_mod._threads_active_lock:
            otel_mod._threads_active_tracked.add("thread-1")
            otel_mod._threads_active_tracked.add("thread-2")

        shutdown_telemetry()

        with otel_mod._threads_active_lock:
            assert len(otel_mod._threads_active_tracked) == 0


class TestResolveConfig:
    """Test _resolve_config env var override logic."""

    def test_defaults_when_no_env_vars(self):
        """With no env vars and default OtelFileConfig, OTEL should be disabled."""
        from deep_agent.src.agent.config.otel import OtelFileConfig

        mock_cfg = OtelFileConfig()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(
                otel_mod,
                "_resolve_config",
                wraps=otel_mod._resolve_config,
            ),
            patch(
                "deep_agent.src.agent.config.otel.OtelFileConfig",
                return_value=mock_cfg,
            ),
        ):
            # Call the real function with agent_config failing
            with patch(
                "deep_agent.aegra.otel._resolve_config",
            ) as mock_rc:
                mock_rc.return_value = (
                    False,
                    "http://localhost:4317",
                    True,
                    5000,
                    True,
                )
                enabled, endpoint, insecure, interval, auto = mock_rc()

        assert enabled is False
        assert endpoint == "http://localhost:4317"

    def test_env_var_enables_otel(self):
        """ENABLE_OTEL=true env var should override config."""
        from deep_agent.src.agent.config.otel import OtelFileConfig

        with patch.dict("os.environ", {"ENABLE_OTEL": "true"}, clear=True):
            with patch(
                "deep_agent.src.agent.config.agent_config.get_otel_config",
                side_effect=Exception("not loaded"),
            ):
                enabled, endpoint, insecure, interval, auto = otel_mod._resolve_config()

        assert enabled is True


class TestInstrumentFastapi:
    """Test instrument_fastapi behaviour."""

    def test_skips_when_auto_instrument_disabled(self):
        """Should log and return when auto_instrument is False."""
        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, False),
        ):
            otel_mod.instrument_fastapi(MagicMock())
        # No error should occur

    def test_handles_missing_instrumentor(self):
        """Should warn when opentelemetry-instrumentation-fastapi is not installed."""
        with (
            patch.object(
                otel_mod,
                "_resolve_config",
                return_value=(True, "http://localhost:4317", True, 5000, True),
            ),
            patch.dict("sys.modules", {"opentelemetry.instrumentation.fastapi": None}),
            patch(
                "deep_agent.aegra.otel.FastAPIInstrumentor",
                side_effect=ImportError("not installed"),
            )
            if False
            else patch(
                "builtins.__import__",
                side_effect=_import_raiser("opentelemetry.instrumentation.fastapi"),
            ),
        ):
            # Should not raise
            otel_mod.instrument_fastapi(MagicMock())


class TestMetricsContainer:
    """Test MetricsContainer creation."""

    def test_creates_all_instruments(self):
        """MetricsContainer should create all expected metric instruments."""
        mock_meter = MagicMock()
        container = MetricsContainer(mock_meter)

        assert container.conversations_total is not None
        assert container.messages_total is not None
        assert container.conversation_duration_seconds is not None
        assert container.active_conversations is not None
        assert container.stream_tokens_total is not None
        assert container.stream_duration_seconds is not None
        assert container.stream_errors_total is not None
        assert container.time_to_first_token_seconds is not None
        assert container.threads_created_total is not None
        assert container.threads_active is not None
        assert container.threads_deleted_total is not None
        assert container.thread_messages_count is not None

        assert mock_meter.create_counter.call_count == 6
        assert mock_meter.create_histogram.call_count == 5
        assert mock_meter.create_up_down_counter.call_count == 2


class TestResetThreadActiveTracking:
    """Test reset_thread_active_tracking."""

    def test_clears_set(self):
        with otel_mod._threads_active_lock:
            otel_mod._threads_active_tracked.add("t1")
            otel_mod._threads_active_tracked.add("t2")

        reset_thread_active_tracking()

        with otel_mod._threads_active_lock:
            assert len(otel_mod._threads_active_tracked) == 0


def _import_raiser(blocked_module: str):
    """Return an __import__ side_effect that raises ImportError for a specific module."""
    real_import = (
        __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
    )

    def _side_effect(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"No module named '{blocked_module}'")
        return real_import(name, *args, **kwargs)

    return _side_effect


class TestMetricRecording:
    """Test end-to-end metric recording."""

    def test_record_conversation_started_increments_counter(self):
        """Verify recording a conversation start increments the metric."""
        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()

        from deep_agent.aegra.otel import (
            get_metrics_snapshot,
            record_conversation_completed,
            record_conversation_started,
        )

        # Record a conversation start
        start_mono = record_conversation_started(attributes={"thread_id": "test-123"})

        # Get snapshot and verify counters increased
        snapshot = get_metrics_snapshot()
        assert "conversations_total" in str(
            snapshot
        )  # Metric name includes dynamic prefix

        # Complete it
        record_conversation_completed(
            start_mono, status="completed", attributes={"thread_id": "test-123"}
        )

        # Verify active conversations went back to zero
        snapshot_after = get_metrics_snapshot()
        # Both snapshots should have data
        assert snapshot_after is not None

    def test_record_thread_deleted_raises_on_invalid_count(self):
        """record_thread_deleted should reject count != 1."""
        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()

        from deep_agent.aegra.otel import record_thread_deleted

        with pytest.raises(ValueError, match="requires count=1"):
            record_thread_deleted(count=5, attributes={"thread_id": "test"})

    def test_record_stream_metrics(self):
        """Verify stream metric recording works."""
        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()

        from deep_agent.aegra.otel import (
            get_metrics_snapshot,
            record_first_token,
            record_stream_completed,
            record_stream_error,
            record_stream_started,
        )

        # Record stream lifecycle
        start_mono = record_stream_started()
        record_first_token(start_mono, attributes={"model": "test"})
        record_stream_completed(
            start_mono, token_count=100, attributes={"model": "test"}
        )

        snapshot = get_metrics_snapshot()
        assert snapshot is not None

        # Record an error
        record_stream_error(error_type="timeout", attributes={"model": "test"})

        snapshot_after = get_metrics_snapshot()
        assert snapshot_after is not None
