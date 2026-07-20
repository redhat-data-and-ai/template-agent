"""Tests for CodeExecutionMetrics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestCodeExecutionMetrics:
    @patch("deep_agent.src.code_execution.metrics.emit_audit_event")
    @patch("deep_agent.src.code_execution.metrics._get_tracer")
    def test_emit_audit(self, mock_tracer, mock_emit):
        from deep_agent.src.code_execution.metrics import CodeExecutionMetrics

        metrics = CodeExecutionMetrics()
        metrics.emit_audit(
            language="python",
            status="success",
            exit_code=0,
            latency_ms=1234.5,
            code_hash="sha256:abc",
            namespace="ap-test-agent",
            image="python:3.12-slim",
            job_name="code-exec-abc",
            timeout=60,
            stdout_bytes=100,
            stderr_bytes=0,
        )
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == "code_execution"
        assert call_args[1]["language"] == "python"
        assert call_args[1]["status"] == "success"
        assert call_args[1]["code_hash"] == "sha256:abc"

    @patch("deep_agent.src.code_execution.metrics._get_tracer")
    def test_start_span(self, mock_get_tracer):
        from deep_agent.src.code_execution.metrics import CodeExecutionMetrics

        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer

        metrics = CodeExecutionMetrics()
        metrics.start_span("code_execution", language="python")
        mock_tracer.start_span.assert_called_once()

    @patch("deep_agent.src.code_execution.metrics._get_tracer")
    def test_trace_span_context_manager(self, mock_get_tracer):
        from deep_agent.src.code_execution.metrics import CodeExecutionMetrics

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer

        metrics = CodeExecutionMetrics()
        with metrics.trace_span("code_execution", language="python") as span:
            assert span is mock_span
        mock_span.end.assert_called_once()

    @patch("deep_agent.src.code_execution.metrics._get_tracer")
    def test_trace_span_none_when_no_tracer(self, mock_get_tracer):
        from deep_agent.src.code_execution.metrics import CodeExecutionMetrics

        mock_get_tracer.return_value = None
        metrics = CodeExecutionMetrics()
        with metrics.trace_span("test") as span:
            assert span is None

    def test_log_started(self):
        from deep_agent.src.code_execution.metrics import CodeExecutionMetrics

        metrics = CodeExecutionMetrics()
        metrics.log_started(language="python", job_name="j", namespace="ns")

    def test_log_completed(self):
        from deep_agent.src.code_execution.metrics import CodeExecutionMetrics

        metrics = CodeExecutionMetrics()
        metrics.log_completed(exit_code=0, duration_ms=1234, status="success")

    def test_compute_code_hash(self):
        from deep_agent.src.code_execution.metrics import compute_code_hash

        h = compute_code_hash("print('hello')")
        assert h.startswith("sha256:")
        assert len(h) > 10

    def test_compute_code_hash_deterministic(self):
        from deep_agent.src.code_execution.metrics import compute_code_hash

        h1 = compute_code_hash("x = 1")
        h2 = compute_code_hash("x = 1")
        assert h1 == h2

    def test_compute_code_hash_different_for_different_code(self):
        from deep_agent.src.code_execution.metrics import compute_code_hash

        h1 = compute_code_hash("x = 1")
        h2 = compute_code_hash("x = 2")
        assert h1 != h2

    @patch("deep_agent.src.code_execution.metrics._get_otel_metrics")
    def test_record_execution_with_otel(self, mock_get_metrics):
        from deep_agent.src.code_execution.metrics import CodeExecutionMetrics

        mock_mc = MagicMock()
        mock_get_metrics.return_value = mock_mc

        metrics = CodeExecutionMetrics()
        metrics.record_execution(
            language="python", org="test", exit_code=0, status="success", duration=2.5
        )
        mock_mc.code_execution_duration_seconds.record.assert_called_once()
        mock_mc.code_executions_total.add.assert_called_once()

    @patch("deep_agent.src.code_execution.metrics._get_otel_metrics")
    def test_record_execution_without_otel(self, mock_get_metrics):
        from deep_agent.src.code_execution.metrics import CodeExecutionMetrics

        mock_get_metrics.return_value = None
        metrics = CodeExecutionMetrics()
        metrics.record_execution(
            language="python", org="test", exit_code=0, status="success", duration=2.5
        )

    def test_record_scheduling_latency(self):
        from deep_agent.src.code_execution.metrics import CodeExecutionMetrics

        metrics = CodeExecutionMetrics()
        metrics.record_scheduling_latency(org="test-org", duration=1.5)


class TestAuditEmitter:
    def test_emit_audit_event_writes_to_stdout(self, capsys):
        from deep_agent.src.audit.emitter import emit_audit_event

        emit_audit_event("code_execution", language="python", status="success")
        captured = capsys.readouterr()
        assert "platform.audit" in captured.out
        assert "code_execution" in captured.out
        assert "python" in captured.out

    def test_scrub_sensitive_keys(self, capsys):
        from deep_agent.src.audit.emitter import emit_audit_event

        emit_audit_event("test", password="secret123", api_key="abc")
        captured = capsys.readouterr()
        assert "secret123" not in captured.out
        assert "[REDACTED]" in captured.out

    def test_audit_context_binding(self):
        from deep_agent.src.audit.context import (
            bind_audit_context,
            clear_audit_context,
            get_audit_context,
        )

        bind_audit_context(user="alice@test.com", org="test-org", trace_id="tr-123")
        ctx = get_audit_context()
        assert ctx["user"] == "alice@test.com"
        assert ctx["org"] == "test-org"
        assert ctx["trace_id"] == "tr-123"
        clear_audit_context()
        ctx = get_audit_context()
        assert ctx["user"] is None
