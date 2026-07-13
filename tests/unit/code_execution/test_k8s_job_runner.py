"""Tests for K8sJobRunner."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from deep_agent.src.code_execution.config import CodeExecutionConfig


class TestExecutionResult:
    def test_format_success(self):
        from deep_agent.src.code_execution.k8s_job_runner import ExecutionResult

        result = ExecutionResult(
            stdout="hello world",
            stderr="",
            exit_code=0,
            duration_seconds=1.5,
            status="success",
            job_name="code-exec-abc123",
            namespace="ap-test-org-test-agent",
        )
        formatted = result.format()
        assert "stdout:" in formatted
        assert "hello world" in formatted
        assert "exit_code: 0" in formatted

    def test_format_timeout(self):
        from deep_agent.src.code_execution.k8s_job_runner import ExecutionResult

        result = ExecutionResult(
            stdout="",
            stderr="",
            exit_code=-1,
            duration_seconds=60.0,
            status="timeout",
            job_name="j",
            namespace="ns",
        )
        formatted = result.format()
        assert "timed out" in formatted

    def test_format_oom(self):
        from deep_agent.src.code_execution.k8s_job_runner import ExecutionResult

        result = ExecutionResult(
            stdout="",
            stderr="",
            exit_code=137,
            duration_seconds=5.0,
            status="oom_killed",
            job_name="j",
            namespace="ns",
        )
        formatted = result.format()
        assert "out of memory" in formatted

    def test_format_no_stdout(self):
        from deep_agent.src.code_execution.k8s_job_runner import ExecutionResult

        result = ExecutionResult(
            stdout="",
            stderr="some error",
            exit_code=1,
            duration_seconds=0.5,
            status="failed",
            job_name="j",
            namespace="ns",
        )
        formatted = result.format()
        assert "stdout:" not in formatted
        assert "stderr:" in formatted
        assert "some error" in formatted


class TestJobManifest:
    def test_manifest_security_fields(self):
        from deep_agent.src.code_execution.k8s_job_runner import K8sJobRunner

        config = CodeExecutionConfig()
        runner = K8sJobRunner(config)
        manifest = runner.build_job_manifest(
            language="python",
            code="print('hello')",
            timeout=60,
            namespace="ap-test-org-test-agent",
            execution_id="test-uuid",
            trace_id="trace-abc",
        )
        spec = manifest.spec
        pod_spec = spec.template.spec

        assert spec.backoff_limit == 0
        assert spec.active_deadline_seconds == 60
        assert spec.ttl_seconds_after_finished == 30
        assert pod_spec.restart_policy == "Never"
        assert pod_spec.automount_service_account_token is False
        assert pod_spec.security_context.run_as_non_root is True
        assert pod_spec.security_context.run_as_user == 1000

        container = pod_spec.containers[0]
        assert container.image == "python:3.12-slim"
        assert container.command == ["python", "-c"]
        assert container.args == ["print('hello')"]
        assert container.security_context.allow_privilege_escalation is False
        assert container.security_context.read_only_root_filesystem is True

    def test_manifest_labels(self):
        from deep_agent.src.code_execution.k8s_job_runner import K8sJobRunner

        config = CodeExecutionConfig()
        runner = K8sJobRunner(config)
        manifest = runner.build_job_manifest(
            language="python",
            code="x=1",
            timeout=30,
            namespace="ap-myorg-myagent",
            execution_id="exec-123",
            trace_id="tr-456",
        )
        labels = manifest.metadata.labels
        assert labels["app.kubernetes.io/managed-by"] == "template-agent"
        assert labels["ai-platform.io/execution-id"] == "exec-123"

    @pytest.mark.parametrize(
        "language,expected_image,expected_cmd",
        [
            ("python", "python:3.12-slim", ["python", "-c"]),
            ("shell", "bash:5", ["bash", "-c"]),
            ("node", "node:22-slim", ["node", "-e"]),
        ],
    )
    def test_manifest_language_mapping(self, language, expected_image, expected_cmd):
        from deep_agent.src.code_execution.k8s_job_runner import K8sJobRunner

        config = CodeExecutionConfig()
        runner = K8sJobRunner(config)
        manifest = runner.build_job_manifest(
            language=language,
            code="code",
            timeout=60,
            namespace="ns",
            execution_id="id",
            trace_id="tr",
        )
        container = manifest.spec.template.spec.containers[0]
        assert container.image == expected_image
        assert container.command == expected_cmd


class TestParseContainerStatus:
    @pytest.mark.parametrize(
        "reason,expected_status",
        [
            ("OOMKilled", "oom_killed"),
            ("Error", "failed"),
            ("DeadlineExceeded", "timeout"),
            (None, "failed"),
        ],
    )
    def test_termination_reason(self, reason, expected_status):
        from deep_agent.src.code_execution.k8s_job_runner import K8sJobRunner

        config = CodeExecutionConfig()
        runner = K8sJobRunner(config)
        exit_code, status = runner.parse_container_status(
            exit_code=1, termination_reason=reason
        )
        assert status == expected_status

    def test_success(self):
        from deep_agent.src.code_execution.k8s_job_runner import K8sJobRunner

        config = CodeExecutionConfig()
        runner = K8sJobRunner(config)
        exit_code, status = runner.parse_container_status(
            exit_code=0, termination_reason=None
        )
        assert status == "success"
        assert exit_code == 0
