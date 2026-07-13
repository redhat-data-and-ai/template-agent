"""Tests for CodeExecutionMetrics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestCodeExecutionMetrics:
    @patch("deep_agent.src.code_execution.metrics.emit_audit_event")
    @patch("deep_agent.src.code_execution.metrics.get_tracer")
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

    @patch("deep_agent.src.code_execution.metrics.get_tracer")
    def test_start_span(self, mock_get_tracer):
        from deep_agent.src.code_execution.metrics import CodeExecutionMetrics

        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer

        metrics = CodeExecutionMetrics()
        metrics.start_span("code_execution", language="python")
        mock_tracer.start_span.assert_called_once()

    @patch("deep_agent.src.code_execution.metrics.get_tracer")
    def test_log_started(self, mock_tracer):
        from deep_agent.src.code_execution.metrics import CodeExecutionMetrics

        metrics = CodeExecutionMetrics()
        metrics.log_started(language="python", job_name="j", namespace="ns")

    @patch("deep_agent.src.code_execution.metrics.get_tracer")
    def test_log_completed(self, mock_tracer):
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
