"""Unit tests for token budget OTEL emission."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.token_budget import otel_emit


@pytest.fixture(autouse=True)
def reset_otel_emit_state() -> None:
    otel_emit._counters_initialized = False
    otel_emit._token_counter = None
    otel_emit._thread_total_counter = None
    otel_emit._daily_total_counter = None


def test_token_budget_otel_enabled_requires_metrics_flag_and_endpoint() -> None:
    mock_settings = MagicMock()
    mock_settings.ENABLE_OTEL_METRICS = True
    mock_settings.OTEL_EXPORTER_OTLP_ENDPOINT = "otel-gateway:4327"
    with patch("deep_agent.src.token_budget.otel_emit.settings", mock_settings):
        assert otel_emit.token_budget_otel_enabled() is True

    mock_settings.ENABLE_OTEL_METRICS = False
    with patch("deep_agent.src.token_budget.otel_emit.settings", mock_settings):
        assert otel_emit.token_budget_otel_enabled() is False


def test_emit_token_usage_skipped_when_metrics_disabled() -> None:
    mock_settings = MagicMock()
    mock_settings.ENABLE_OTEL_METRICS = False
    mock_settings.OTEL_EXPORTER_OTLP_ENDPOINT = ""
    with (
        patch("deep_agent.src.token_budget.otel_emit.settings", mock_settings),
        patch.object(otel_emit.logger, "info") as log_info,
    ):
        otel_emit.emit_token_usage(
            thread_id="thread-1",
            user_id="user-1",
            input_tokens=10,
            output_tokens=5,
            cumulative_total=15,
            cumulative_input=10,
            cumulative_output=5,
        )

    log_info.assert_not_called()


def test_emit_token_usage_logs_expected_payload() -> None:
    mock_settings = MagicMock()
    mock_settings.ENABLE_OTEL_METRICS = True
    mock_settings.OTEL_EXPORTER_OTLP_ENDPOINT = "otel-gateway:4327"
    mock_settings.ENABLE_OTEL_TRACES = False
    mock_settings.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT = ""

    with (
        patch("deep_agent.src.token_budget.otel_emit.settings", mock_settings),
        patch(
            "deep_agent.src.token_budget.otel_emit._agent_name",
            return_value="health-assistant",
        ),
        patch.object(otel_emit, "_ensure_counters"),
        patch.object(otel_emit.logger, "info") as log_info,
    ):
        otel_emit.emit_token_usage(
            thread_id="thread-abc",
            user_id="user-xyz",
            input_tokens=100,
            output_tokens=25,
            cumulative_total=125,
            cumulative_input=100,
            cumulative_output=25,
            timestamp="2026-06-23T12:00:00+00:00",
        )

    log_info.assert_called_once_with(
        "token_budget_usage",
        thread_id="thread-abc",
        user_id="user-xyz",
        input_tokens=100,
        output_tokens=25,
        total_tokens=125,
        cumulative_total_tokens=125,
        cumulative_input_tokens=100,
        cumulative_output_tokens=25,
        timestamp="2026-06-23T12:00:00+00:00",
        **{"agent.name": "health-assistant"},
    )


def test_emit_token_usage_records_metrics() -> None:
    mock_token_counter = MagicMock()
    mock_thread_counter = MagicMock()
    mock_settings = MagicMock()
    mock_settings.ENABLE_OTEL_METRICS = True
    mock_settings.OTEL_EXPORTER_OTLP_ENDPOINT = "otel-gateway:4327"
    mock_settings.ENABLE_OTEL_TRACES = False
    mock_settings.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT = ""

    with (
        patch("deep_agent.src.token_budget.otel_emit.settings", mock_settings),
        patch(
            "deep_agent.src.token_budget.otel_emit._agent_name",
            return_value="health-assistant",
        ),
        patch.object(otel_emit, "_token_counter", mock_token_counter),
        patch.object(otel_emit, "_thread_total_counter", mock_thread_counter),
        patch.object(otel_emit, "_counters_initialized", True),
        patch.object(otel_emit.logger, "info"),
    ):
        otel_emit.emit_token_usage(
            thread_id="thread-1",
            user_id="user-1",
            input_tokens=80,
            output_tokens=20,
            cumulative_total=100,
            cumulative_input=80,
            cumulative_output=20,
        )

    base_attrs = {
        "agent.name": "health-assistant",
        "thread_id": "thread-1",
        "user_id": "user-1",
    }
    mock_token_counter.add.assert_any_call(80, {**base_attrs, "token.type": "input"})
    mock_token_counter.add.assert_any_call(20, {**base_attrs, "token.type": "output"})
    mock_thread_counter.add.assert_called_once_with(
        100,
        {**base_attrs, "aggregation": "cumulative"},
    )


def test_emit_token_usage_adds_span_event_when_traces_enabled() -> None:
    mock_settings = MagicMock()
    mock_settings.ENABLE_OTEL_METRICS = True
    mock_settings.OTEL_EXPORTER_OTLP_ENDPOINT = "otel-gateway:4327"
    mock_settings.ENABLE_OTEL_TRACES = True
    mock_settings.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT = "jaeger:4317"
    mock_settings.otel_traces_active.return_value = True
    mock_settings.resolved_otel_traces_endpoint.return_value = "jaeger:4317"
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True

    with (
        patch("deep_agent.src.token_budget.otel_emit.settings", mock_settings),
        patch(
            "deep_agent.src.token_budget.otel_emit._agent_name",
            return_value="health-assistant",
        ),
        patch.object(otel_emit, "_ensure_counters"),
        patch.object(otel_emit.logger, "info"),
        patch("opentelemetry.trace.get_current_span", return_value=mock_span),
    ):
        otel_emit.emit_token_usage(
            thread_id="thread-1",
            user_id=None,
            input_tokens=10,
            output_tokens=5,
            cumulative_total=15,
            cumulative_input=10,
            cumulative_output=5,
        )

    mock_span.add_event.assert_called_once()
    event_name = mock_span.add_event.call_args[0][0]
    kwargs = mock_span.add_event.call_args[1]
    assert event_name == "token_budget.usage"
    assert kwargs["attributes"]["thread_id"] == "thread-1"
    assert kwargs["attributes"]["timestamp"]


def test_emit_daily_token_usage_logs_expected_payload() -> None:
    mock_settings = MagicMock()
    mock_settings.ENABLE_OTEL_METRICS = True
    mock_settings.OTEL_EXPORTER_OTLP_ENDPOINT = "otel-gateway:4327"
    mock_settings.ENABLE_OTEL_TRACES = False
    mock_settings.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT = ""
    mock_daily_counter = MagicMock()

    with (
        patch("deep_agent.src.token_budget.otel_emit.settings", mock_settings),
        patch(
            "deep_agent.src.token_budget.otel_emit._agent_name",
            return_value="health-assistant",
        ),
        patch.object(otel_emit, "_daily_total_counter", mock_daily_counter),
        patch.object(otel_emit, "_counters_initialized", True),
        patch.object(otel_emit.logger, "info") as log_info,
    ):
        otel_emit.emit_daily_token_usage(
            user_id="user-1",
            total_tokens=5000,
            date="2026-06-23",
            timestamp="2026-06-23T18:30:00+00:00",
        )

    log_info.assert_called_once_with(
        "token_budget_daily_usage",
        user_id="user-1",
        total_tokens=5000,
        date="2026-06-23",
        timestamp="2026-06-23T18:30:00+00:00",
        **{"agent.name": "health-assistant"},
    )
    mock_daily_counter.add.assert_called_once_with(
        5000,
        {
            "agent.name": "health-assistant",
            "user_id": "user-1",
            "date": "2026-06-23",
        },
    )
