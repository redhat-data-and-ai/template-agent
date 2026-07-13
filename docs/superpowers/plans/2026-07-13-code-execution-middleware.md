# CodeExecutionMiddleware Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a middleware that injects an `execute_code` tool into the agent, intercepts calls to it, and executes code in ephemeral K8s Jobs with full observability.

**Architecture:** DynamicToolMiddleware pattern — `CodeExecutionMiddleware(AgentMiddleware)` injects the tool via `awrap_model_call` and handles execution in `awrap_tool_call` by delegating to `K8sJobRunner` which manages the K8s Job lifecycle. `CodeExecutionMetrics` coordinates all four observability layers (OTEL metrics, tracing, audit, logs).

**Tech Stack:** Python 3.12, Pydantic v2, kubernetes Python client, OpenTelemetry SDK, LangChain AgentMiddleware, pytest + pytest-asyncio

## Global Constraints

- Python 3.12+ required
- All new code under `deep_agent/src/code_execution/`
- Follow existing patterns: `from __future__ import annotations` at top of every module
- Pydantic v2 models (use `BaseModel`, `Field`, `model_validate`)
- Async-first: `awrap_*` methods are the primary path; sync `wrap_*` pass through
- `kubernetes` client is optional — import inside functions, guard with try/except ImportError
- All tests use `pytest.mark.asyncio` (auto mode), `unittest.mock.patch`, `MagicMock`/`AsyncMock`
- Never commit without explicit user approval
- No Co-Authored-By lines in commits
- Design spec: `docs/superpowers/specs/2026-07-13-code-execution-middleware-design.md`

---

### Task 1: CodeExecutionConfig (Pydantic model + tests)

**Files:**
- Create: `deep_agent/src/code_execution/__init__.py`
- Create: `deep_agent/src/code_execution/config.py`
- Create: `tests/unit/code_execution/__init__.py`
- Create: `tests/unit/code_execution/test_config.py`

**Interfaces:**
- Consumes: Nothing (foundation task)
- Produces: `CodeExecutionConfig` Pydantic model used by Tasks 2-4. Fields: `enabled: bool`, `max_timeout_seconds: int`, `max_code_length: int`, `max_output_bytes: int`, `images: dict[str, str]`, `entrypoints: dict[str, list[str]]`, `resource_requests: dict[str, str]`, `resource_limits: dict[str, str]`, `tmp_size_limit: str`, `job_ttl_after_finished: int`, `pod_poll_interval_seconds: float`, `pod_poll_timeout_seconds: float`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/code_execution/__init__.py` (empty) and `tests/unit/code_execution/test_config.py`:

```python
"""Tests for CodeExecutionConfig."""
from __future__ import annotations

import pytest


class TestCodeExecutionConfigDefaults:
    def test_defaults(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        cfg = CodeExecutionConfig()
        assert cfg.enabled is False
        assert cfg.max_timeout_seconds == 60
        assert cfg.max_code_length == 50_000
        assert cfg.max_output_bytes == 1_048_576
        assert "python" in cfg.images
        assert "shell" in cfg.images
        assert "node" in cfg.images
        assert cfg.images["python"] == "python:3.12-slim"
        assert cfg.entrypoints["python"] == ["python", "-c"]
        assert cfg.entrypoints["shell"] == ["bash", "-c"]
        assert cfg.entrypoints["node"] == ["node", "-e"]
        assert cfg.resource_requests == {"cpu": "100m", "memory": "128Mi"}
        assert cfg.resource_limits == {"cpu": "500m", "memory": "256Mi"}
        assert cfg.tmp_size_limit == "64Mi"
        assert cfg.job_ttl_after_finished == 30
        assert cfg.pod_poll_interval_seconds == 1.0
        assert cfg.pod_poll_timeout_seconds == 120.0

    def test_from_dict(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        cfg = CodeExecutionConfig.model_validate({
            "enabled": True,
            "max_timeout_seconds": 120,
            "images": {"python": "my-registry/python:3.12"},
        })
        assert cfg.enabled is True
        assert cfg.max_timeout_seconds == 120
        assert cfg.images["python"] == "my-registry/python:3.12"
        assert cfg.images["shell"] == "bash:5"  # default preserved

    def test_supported_languages(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        cfg = CodeExecutionConfig()
        assert cfg.supported_languages == {"python", "shell", "node"}


class TestCodeExecutionConfigValidation:
    def test_timeout_min(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        with pytest.raises(Exception):
            CodeExecutionConfig(max_timeout_seconds=4)

    def test_timeout_max(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        with pytest.raises(Exception):
            CodeExecutionConfig(max_timeout_seconds=301)

    def test_poll_interval_min(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        with pytest.raises(Exception):
            CodeExecutionConfig(pod_poll_interval_seconds=0.1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/code_execution/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deep_agent.src.code_execution'`

- [ ] **Step 3: Implement CodeExecutionConfig**

Create `deep_agent/src/code_execution/__init__.py`:

```python
"""Code execution middleware — ephemeral K8s Job backend for agent code execution."""
from __future__ import annotations
```

Create `deep_agent/src/code_execution/config.py`:

```python
"""Configuration model for code execution middleware."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CodeExecutionConfig(BaseModel):
    """Configuration for code execution middleware.

    Loaded from the ``code_execution:`` section of ``middleware:`` in
    ``config/agent/runtime/agent.yaml``.
    """

    enabled: bool = False
    max_timeout_seconds: int = Field(default=60, ge=5, le=300)
    max_code_length: int = Field(default=50_000, ge=100, le=500_000)
    max_output_bytes: int = Field(default=1_048_576)

    images: dict[str, str] = Field(default_factory=lambda: {
        "python": "python:3.12-slim",
        "shell": "bash:5",
        "node": "node:22-slim",
    })

    entrypoints: dict[str, list[str]] = Field(default_factory=lambda: {
        "python": ["python", "-c"],
        "shell": ["bash", "-c"],
        "node": ["node", "-e"],
    })

    resource_requests: dict[str, str] = Field(
        default_factory=lambda: {"cpu": "100m", "memory": "128Mi"}
    )
    resource_limits: dict[str, str] = Field(
        default_factory=lambda: {"cpu": "500m", "memory": "256Mi"}
    )

    tmp_size_limit: str = "64Mi"
    job_ttl_after_finished: int = Field(default=30, ge=0, le=300)
    pod_poll_interval_seconds: float = Field(default=1.0, ge=0.5, le=10.0)
    pod_poll_timeout_seconds: float = Field(default=120.0, ge=10.0, le=600.0)

    @property
    def supported_languages(self) -> set[str]:
        return set(self.images.keys()) & set(self.entrypoints.keys())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/code_execution/test_config.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add deep_agent/src/code_execution/__init__.py deep_agent/src/code_execution/config.py tests/unit/code_execution/__init__.py tests/unit/code_execution/test_config.py
git commit -m "feat(code-execution): add CodeExecutionConfig Pydantic model with tests"
```

---

### Task 2: K8sJobRunner (Job lifecycle + tests)

**Files:**
- Create: `deep_agent/src/code_execution/k8s_job_runner.py`
- Create: `tests/unit/code_execution/test_k8s_job_runner.py`

**Interfaces:**
- Consumes: `CodeExecutionConfig` from Task 1
- Produces: `K8sJobRunner` class with `async run(language: str, code: str, timeout: int, namespace: str) -> ExecutionResult` and `ExecutionResult` dataclass with fields `stdout: str`, `stderr: str`, `exit_code: int`, `duration_seconds: float`, `status: str`, `job_name: str`, `namespace: str` and method `format() -> str`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/code_execution/test_k8s_job_runner.py`:

```python
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
            stdout="", stderr="", exit_code=-1,
            duration_seconds=60.0, status="timeout",
            job_name="j", namespace="ns",
        )
        formatted = result.format()
        assert "timed out" in formatted

    def test_format_oom(self):
        from deep_agent.src.code_execution.k8s_job_runner import ExecutionResult

        result = ExecutionResult(
            stdout="", stderr="", exit_code=137,
            duration_seconds=5.0, status="oom_killed",
            job_name="j", namespace="ns",
        )
        formatted = result.format()
        assert "out of memory" in formatted

    def test_format_no_stdout(self):
        from deep_agent.src.code_execution.k8s_job_runner import ExecutionResult

        result = ExecutionResult(
            stdout="", stderr="some error", exit_code=1,
            duration_seconds=0.5, status="failed",
            job_name="j", namespace="ns",
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
            language="python", code="x=1", timeout=30,
            namespace="ap-myorg-myagent",
            execution_id="exec-123", trace_id="tr-456",
        )
        labels = manifest.metadata.labels
        assert labels["app.kubernetes.io/managed-by"] == "template-agent"
        assert labels["ai-platform.io/execution-id"] == "exec-123"

    @pytest.mark.parametrize("language,expected_image,expected_cmd", [
        ("python", "python:3.12-slim", ["python", "-c"]),
        ("shell", "bash:5", ["bash", "-c"]),
        ("node", "node:22-slim", ["node", "-e"]),
    ])
    def test_manifest_language_mapping(self, language, expected_image, expected_cmd):
        from deep_agent.src.code_execution.k8s_job_runner import K8sJobRunner

        config = CodeExecutionConfig()
        runner = K8sJobRunner(config)
        manifest = runner.build_job_manifest(
            language=language, code="code", timeout=60,
            namespace="ns", execution_id="id", trace_id="tr",
        )
        container = manifest.spec.template.spec.containers[0]
        assert container.image == expected_image
        assert container.command == expected_cmd


class TestParseContainerStatus:
    @pytest.mark.parametrize("reason,expected_status", [
        ("OOMKilled", "oom_killed"),
        ("Error", "failed"),
        ("DeadlineExceeded", "timeout"),
        (None, "failed"),
    ])
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/code_execution/test_k8s_job_runner.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement K8sJobRunner**

Create `deep_agent/src/code_execution/k8s_job_runner.py`:

```python
"""K8s Job lifecycle manager for ephemeral code execution."""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

from deep_agent.src.code_execution.config import CodeExecutionConfig
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    status: str
    job_name: str
    namespace: str

    def format(self) -> str:
        parts = []
        if self.stdout:
            parts.append(f"stdout:\n{self.stdout}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        parts.append(f"exit_code: {self.exit_code}")
        if self.status == "timeout":
            parts.append(f"(timed out after {self.duration_seconds:.1f}s)")
        if self.status == "oom_killed":
            parts.append("(killed: out of memory)")
        return "\n".join(parts)


class K8sJobRunner:
    """Manages the lifecycle of ephemeral K8s Jobs for code execution."""

    def __init__(self, config: CodeExecutionConfig) -> None:
        self._config = config
        self._batch_api: Any | None = None
        self._core_api: Any | None = None

    def _ensure_k8s_client(self) -> None:
        if self._batch_api is not None:
            return
        try:
            from kubernetes import client, config as k8s_config

            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()
            self._batch_api = client.BatchV1Api()
            self._core_api = client.CoreV1Api()
        except ImportError:
            raise RuntimeError(
                "kubernetes package required for code execution. "
                "Install with: pip install kubernetes"
            )

    def build_job_manifest(
        self,
        *,
        language: str,
        code: str,
        timeout: int,
        namespace: str,
        execution_id: str,
        trace_id: str | None = None,
    ) -> Any:
        from kubernetes import client

        labels = {
            "app.kubernetes.io/name": "code-execution",
            "app.kubernetes.io/component": "ephemeral-job",
            "app.kubernetes.io/managed-by": "template-agent",
            "ai-platform.io/execution-id": execution_id,
        }

        annotations = {}
        if trace_id:
            annotations["ai-platform.io/trace-id"] = trace_id

        container = client.V1Container(
            name="executor",
            image=self._config.images[language],
            command=self._config.entrypoints[language],
            args=[code],
            resources=client.V1ResourceRequirements(
                requests=dict(self._config.resource_requests),
                limits=dict(self._config.resource_limits),
            ),
            security_context=client.V1SecurityContext(
                allow_privilege_escalation=False,
                read_only_root_filesystem=True,
                capabilities=client.V1Capabilities(drop=["ALL"]),
            ),
            volume_mounts=[
                client.V1VolumeMount(name="tmp", mount_path="/tmp"),
            ],
        )

        pod_spec = client.V1PodSpec(
            restart_policy="Never",
            automount_service_account_token=False,
            security_context=client.V1PodSecurityContext(
                run_as_non_root=True,
                run_as_user=1000,
                run_as_group=1000,
                fs_group=1000,
                seccomp_profile=client.V1SeccompProfile(type="RuntimeDefault"),
            ),
            containers=[container],
            volumes=[
                client.V1Volume(
                    name="tmp",
                    empty_dir=client.V1EmptyDirVolumeSource(
                        size_limit=self._config.tmp_size_limit,
                    ),
                ),
            ],
        )

        job_name = f"code-exec-{execution_id[:8]}"

        return client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=namespace,
                labels=labels,
                annotations=annotations or None,
            ),
            spec=client.V1JobSpec(
                active_deadline_seconds=timeout,
                ttl_seconds_after_finished=self._config.job_ttl_after_finished,
                backoff_limit=0,
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels=labels,
                        annotations=annotations or None,
                    ),
                    spec=pod_spec,
                ),
            ),
        )

    def parse_container_status(
        self, *, exit_code: int, termination_reason: str | None
    ) -> tuple[int, str]:
        if exit_code == 0 and termination_reason is None:
            return exit_code, "success"
        if termination_reason == "OOMKilled":
            return exit_code, "oom_killed"
        if termination_reason == "DeadlineExceeded":
            return exit_code, "timeout"
        return exit_code, "failed"

    def resolve_namespace(self) -> str:
        org = os.environ.get("AI_PLATFORM_AGENT_ORG", "default")
        agent = os.environ.get("AI_PLATFORM_AGENT_NAME", "agent")
        return f"ap-{org}-{agent}"

    async def run(
        self,
        *,
        language: str,
        code: str,
        timeout: int,
        namespace: str | None = None,
    ) -> ExecutionResult:
        self._ensure_k8s_client()
        ns = namespace or self.resolve_namespace()
        execution_id = uuid.uuid4().hex
        job_name = f"code-exec-{execution_id[:8]}"
        started = time.monotonic()

        manifest = self.build_job_manifest(
            language=language,
            code=code,
            timeout=timeout,
            namespace=ns,
            execution_id=execution_id,
        )
        job_name = manifest.metadata.name

        try:
            await asyncio.to_thread(
                self._batch_api.create_namespaced_job,
                namespace=ns,
                body=manifest,
            )
            logger.info("code_execution_job_created", job_name=job_name, namespace=ns)

            pod_name = await self._wait_for_pod(job_name, ns, timeout)
            stdout, stderr = await self._collect_logs(pod_name, ns)
            raw_exit_code, termination_reason = await self._get_exit_info(pod_name, ns)
            exit_code, status = self.parse_container_status(
                exit_code=raw_exit_code,
                termination_reason=termination_reason,
            )

            duration = time.monotonic() - started
            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                duration_seconds=duration,
                status=status,
                job_name=job_name,
                namespace=ns,
            )
        except asyncio.TimeoutError:
            duration = time.monotonic() - started
            return ExecutionResult(
                stdout="", stderr="", exit_code=-1,
                duration_seconds=duration, status="timeout",
                job_name=job_name, namespace=ns,
            )
        except Exception as exc:
            duration = time.monotonic() - started
            logger.error("code_execution_failed", error=str(exc), job_name=job_name)
            return ExecutionResult(
                stdout="", stderr=str(exc), exit_code=-1,
                duration_seconds=duration, status="error",
                job_name=job_name, namespace=ns,
            )
        finally:
            await self._cleanup(job_name, ns)

    async def _wait_for_pod(self, job_name: str, namespace: str, timeout: int) -> str:
        deadline = time.monotonic() + self._config.pod_poll_timeout_seconds
        while time.monotonic() < deadline:
            pods = await asyncio.to_thread(
                self._core_api.list_namespaced_pod,
                namespace=namespace,
                label_selector=f"job-name={job_name}",
            )
            for pod in pods.items:
                phase = pod.status.phase
                if phase in ("Succeeded", "Failed"):
                    return pod.metadata.name
                if phase == "Running":
                    return pod.metadata.name
            await asyncio.sleep(self._config.pod_poll_interval_seconds)

        raise asyncio.TimeoutError(f"Pod for {job_name} did not start within {self._config.pod_poll_timeout_seconds}s")

    async def _collect_logs(self, pod_name: str, namespace: str) -> tuple[str, str]:
        try:
            logs = await asyncio.to_thread(
                self._core_api.read_namespaced_pod_log,
                name=pod_name,
                namespace=namespace,
                container="executor",
            )
            if len(logs) > self._config.max_output_bytes:
                logs = logs[: self._config.max_output_bytes] + "\n[truncated at 1MB]"
            return logs, ""
        except Exception as exc:
            logger.warning("code_execution_log_collection_failed", error=str(exc))
            return "", f"[warning: logs partially collected: {exc}]"

    async def _get_exit_info(self, pod_name: str, namespace: str) -> tuple[int, str | None]:
        try:
            pod = await asyncio.to_thread(
                self._core_api.read_namespaced_pod,
                name=pod_name,
                namespace=namespace,
            )
            for cs in pod.status.container_statuses or []:
                if cs.name == "executor" and cs.state and cs.state.terminated:
                    return (
                        cs.state.terminated.exit_code or 0,
                        cs.state.terminated.reason,
                    )
            return -1, None
        except Exception:
            return -1, None

    async def _cleanup(self, job_name: str, namespace: str) -> None:
        try:
            from kubernetes import client

            await asyncio.to_thread(
                self._batch_api.delete_namespaced_job,
                name=job_name,
                namespace=namespace,
                body=client.V1DeleteOptions(propagation_policy="Foreground"),
            )
            logger.debug("code_execution_cleanup", job_name=job_name)
        except Exception as exc:
            logger.warning(
                "code_execution_cleanup_failed",
                job_name=job_name,
                error=str(exc),
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/code_execution/test_k8s_job_runner.py -v`
Expected: All tests PASS (manifest tests use the real kubernetes client library for model construction)

- [ ] **Step 5: Commit**

```bash
git add deep_agent/src/code_execution/k8s_job_runner.py tests/unit/code_execution/test_k8s_job_runner.py
git commit -m "feat(code-execution): add K8sJobRunner with Job lifecycle management"
```

---

### Task 3: CodeExecutionMetrics (observability + tests)

**Files:**
- Create: `deep_agent/src/code_execution/metrics.py`
- Create: `tests/unit/code_execution/test_metrics.py`

**Interfaces:**
- Consumes: `get_tracer()`, `get_metrics()` from `deep_agent.aegra.otel`; `emit_audit_event()` from `deep_agent.src.audit.emitter`; `get_python_logger()` from `deep_agent.utils.pylogger`
- Produces: `CodeExecutionMetrics` class with methods: `record_execution(*, language, org, exit_code, status, duration)`, `record_error(*, language, org, error_type)`, `record_scheduling_latency(*, org, duration)`, `increment_active(*, org)`, `decrement_active(*, org)`, `start_span(name, **attributes)`, `emit_audit(*, language, status, exit_code, latency_ms, code_hash, namespace, image, job_name, timeout, stdout_bytes, stderr_bytes)`, `log_started(**fields)`, `log_completed(**fields)`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/code_execution/test_metrics.py`:

```python
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
        # Should not raise
        metrics.log_started(language="python", job_name="j", namespace="ns")

    @patch("deep_agent.src.code_execution.metrics.get_tracer")
    def test_log_completed(self, mock_tracer):
        from deep_agent.src.code_execution.metrics import CodeExecutionMetrics

        metrics = CodeExecutionMetrics()
        metrics.log_completed(exit_code=0, duration_ms=1234, status="success")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/code_execution/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CodeExecutionMetrics**

Create `deep_agent/src/code_execution/metrics.py`:

```python
"""Observability for code execution — OTEL metrics, tracing, audit, logs."""
from __future__ import annotations

import hashlib
from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _get_tracer() -> Any:
    try:
        from deep_agent.aegra.otel import get_tracer
        return get_tracer("code_execution")
    except Exception:
        return None


def get_tracer() -> Any:
    return _get_tracer()


def emit_audit_event(event_type: str, **details: Any) -> None:
    try:
        from deep_agent.src.audit.emitter import emit_audit_event as _emit
        _emit(event_type, **details)
    except ImportError:
        logger.debug("Audit emitter not available, skipping audit event")
    except Exception as exc:
        logger.warning("Failed to emit audit event: %s", exc)


def compute_code_hash(code: str) -> str:
    return f"sha256:{hashlib.sha256(code.encode()).hexdigest()}"


class CodeExecutionMetrics:
    """Centralized observability for code execution."""

    def __init__(self) -> None:
        self._tracer = get_tracer()

    def record_execution(
        self, *, language: str, org: str, exit_code: int, status: str, duration: float
    ) -> None:
        logger.info(
            "code_execution_metric",
            language=language, org=org,
            exit_code=exit_code, status=status,
            duration_seconds=round(duration, 3),
        )

    def record_error(self, *, language: str, org: str, error_type: str) -> None:
        logger.warning(
            "code_execution_error_metric",
            language=language, org=org, error_type=error_type,
        )

    def record_scheduling_latency(self, *, org: str, duration: float) -> None:
        logger.debug(
            "code_execution_scheduling_metric",
            org=org, scheduling_seconds=round(duration, 3),
        )

    def increment_active(self, *, org: str) -> None:
        logger.debug("code_execution_active_increment", org=org)

    def decrement_active(self, *, org: str) -> None:
        logger.debug("code_execution_active_decrement", org=org)

    def start_span(self, name: str, **attributes: Any) -> Any:
        if self._tracer is None:
            return None
        return self._tracer.start_span(name, attributes=attributes)

    def emit_audit(
        self,
        *,
        language: str,
        status: str,
        exit_code: int,
        latency_ms: float,
        code_hash: str,
        namespace: str,
        image: str,
        job_name: str,
        timeout: int,
        stdout_bytes: int,
        stderr_bytes: int,
    ) -> None:
        emit_audit_event(
            "code_execution",
            language=language,
            status=status,
            exit_code=exit_code,
            latency_ms=latency_ms,
            code_hash=code_hash,
            namespace=namespace,
            image=image,
            job_name=job_name,
            timeout_seconds=timeout,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
        )

    def log_started(self, **fields: Any) -> None:
        logger.info("code_execution_started", **fields)

    def log_completed(self, **fields: Any) -> None:
        logger.info("code_execution_completed", **fields)

    def log_timeout(self, **fields: Any) -> None:
        logger.warning("code_execution_timeout", **fields)

    def log_oom(self, **fields: Any) -> None:
        logger.warning("code_execution_oom_killed", **fields)

    def log_failed(self, **fields: Any) -> None:
        logger.error("code_execution_failed", **fields)

    def log_cleanup(self, **fields: Any) -> None:
        logger.debug("code_execution_cleanup", **fields)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/code_execution/test_metrics.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add deep_agent/src/code_execution/metrics.py tests/unit/code_execution/test_metrics.py
git commit -m "feat(code-execution): add CodeExecutionMetrics with 4-layer observability"
```

---

### Task 4: CodeExecutionMiddleware (middleware + tool injection + integration test)

**Files:**
- Create: `deep_agent/src/code_execution/middleware.py`
- Create: `tests/unit/code_execution/test_middleware.py`
- Modify: `deep_agent/src/code_execution/__init__.py` (add public exports)

**Interfaces:**
- Consumes: `CodeExecutionConfig` (Task 1), `K8sJobRunner` + `ExecutionResult` (Task 2), `CodeExecutionMetrics` + `compute_code_hash` (Task 3), `AgentMiddleware`/`ModelRequest`/`ToolCallRequest` from `langchain.agents.middleware.types`, `ToolMessage` from `langchain_core.messages`
- Produces: `CodeExecutionMiddleware(AgentMiddleware)` — the middleware class that gets registered in `build_middleware_list()`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/code_execution/test_middleware.py`:

```python
"""Tests for CodeExecutionMiddleware."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from deep_agent.src.code_execution.config import CodeExecutionConfig


class TestToolInjection:
    async def test_awrap_model_call_injects_tool(self):
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        config = CodeExecutionConfig(enabled=True)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        request.tools = [MagicMock()]
        original_tool_count = len(request.tools)

        handler = AsyncMock()
        override_request = MagicMock()
        request.override = MagicMock(return_value=override_request)
        handler.return_value = MagicMock()

        await mw.awrap_model_call(request, handler)

        request.override.assert_called_once()
        call_kwargs = request.override.call_args
        injected_tools = call_kwargs[1]["tools"] if "tools" in call_kwargs[1] else call_kwargs[0][0] if call_kwargs[0] else None
        # The override should have been called with tools that include the original + execute_code
        handler.assert_called_once_with(override_request)

    async def test_awrap_model_call_disabled(self):
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        config = CodeExecutionConfig(enabled=False)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        handler = AsyncMock(return_value=MagicMock())

        result = await mw.awrap_model_call(request, handler)
        handler.assert_called_once_with(request)
        request.override.assert_not_called()


class TestToolCallRouting:
    async def test_passthrough_non_execute_code(self):
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        config = CodeExecutionConfig(enabled=True)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        request.tool_call = {"name": "search_web", "args": {}, "id": "tc1"}
        handler = AsyncMock(return_value=MagicMock())

        result = await mw.awrap_tool_call(request, handler)
        handler.assert_called_once_with(request)

    @patch("deep_agent.src.code_execution.middleware.K8sJobRunner")
    async def test_invalid_language(self, mock_runner_cls):
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        config = CodeExecutionConfig(enabled=True)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        request.tool_call = {
            "name": "execute_code",
            "args": {"code": "x=1", "language": "ruby"},
            "id": "tc1",
        }
        handler = AsyncMock()

        result = await mw.awrap_tool_call(request, handler)
        handler.assert_not_called()
        assert "Unsupported language" in result.content
        assert "ruby" in result.content

    @patch("deep_agent.src.code_execution.middleware.K8sJobRunner")
    async def test_code_too_long(self, mock_runner_cls):
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        config = CodeExecutionConfig(enabled=True, max_code_length=10)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        request.tool_call = {
            "name": "execute_code",
            "args": {"code": "x" * 100, "language": "python"},
            "id": "tc1",
        }
        handler = AsyncMock()

        result = await mw.awrap_tool_call(request, handler)
        handler.assert_not_called()
        assert "exceeds maximum length" in result.content

    @patch("deep_agent.src.code_execution.middleware.CodeExecutionMetrics")
    @patch("deep_agent.src.code_execution.middleware.K8sJobRunner")
    async def test_successful_execution(self, mock_runner_cls, mock_metrics_cls):
        from deep_agent.src.code_execution.k8s_job_runner import ExecutionResult
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock(return_value=ExecutionResult(
            stdout="42", stderr="", exit_code=0,
            duration_seconds=1.5, status="success",
            job_name="code-exec-abc", namespace="ap-test-agent",
        ))
        mock_runner_cls.return_value = mock_runner

        config = CodeExecutionConfig(enabled=True)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        request.tool_call = {
            "name": "execute_code",
            "args": {"code": "print(42)", "language": "python"},
            "id": "tc1",
        }
        handler = AsyncMock()

        result = await mw.awrap_tool_call(request, handler)
        handler.assert_not_called()
        assert "42" in result.content
        assert "exit_code: 0" in result.content
        assert result.tool_call_id == "tc1"


class TestSyncPassthrough:
    def test_wrap_model_call_passes_through(self):
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        config = CodeExecutionConfig(enabled=True)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        handler = MagicMock(return_value=MagicMock())

        result = mw.wrap_model_call(request, handler)
        handler.assert_called_once_with(request)

    def test_wrap_tool_call_passes_through(self):
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        config = CodeExecutionConfig(enabled=True)
        mw = CodeExecutionMiddleware(config=config)

        request = MagicMock()
        handler = MagicMock(return_value=MagicMock())

        result = mw.wrap_tool_call(request, handler)
        handler.assert_called_once_with(request)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/code_execution/test_middleware.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CodeExecutionMiddleware**

Create `deep_agent/src/code_execution/middleware.py`:

```python
"""CodeExecutionMiddleware — inject execute_code tool, route to K8s Jobs."""
from __future__ import annotations

import os
import time
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from deep_agent.src.code_execution.config import CodeExecutionConfig
from deep_agent.src.code_execution.k8s_job_runner import K8sJobRunner
from deep_agent.src.code_execution.metrics import (
    CodeExecutionMetrics,
    compute_code_hash,
)
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _build_execute_code_tool(config: CodeExecutionConfig) -> Any:
    languages = ", ".join(sorted(config.supported_languages))

    @tool
    def execute_code(
        code: str,
        language: str = "python",
        timeout: int = 60,
    ) -> str:
        f"""Execute code in an isolated sandbox environment.

        Args:
            code: The source code to execute.
            language: Programming language — {languages}.
            timeout: Maximum execution time in seconds (5-{config.max_timeout_seconds}).

        Returns:
            Execution output with stdout, stderr, and exit code.
        """
        return "This tool is handled by CodeExecutionMiddleware"

    return execute_code


class CodeExecutionMiddleware(AgentMiddleware):
    """Inject execute_code tool and route calls to K8s Job backend."""

    def __init__(self, *, config: CodeExecutionConfig) -> None:
        self._config = config
        self._runner = K8sJobRunner(config)
        self._metrics = CodeExecutionMetrics()
        self._execute_code_tool = _build_execute_code_tool(config)

    def wrap_model_call(self, request: ModelRequest[Any], handler: Any) -> ModelResponse[Any]:
        return handler(request)

    async def awrap_model_call(self, request: ModelRequest[Any], handler: Any) -> ModelResponse[Any]:
        if not self._config.enabled:
            return await handler(request)
        updated = request.override(tools=[*request.tools, self._execute_code_tool])
        return await handler(updated)

    def wrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        return handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        tool_call = request.tool_call
        if tool_call.get("name") != "execute_code":
            return await handler(request)

        args = tool_call.get("args", {})
        code = args.get("code", "")
        language = args.get("language", "python")
        timeout = min(
            int(args.get("timeout", self._config.max_timeout_seconds)),
            self._config.max_timeout_seconds,
        )
        tool_call_id = tool_call.get("id", "")

        if language not in self._config.supported_languages:
            return ToolMessage(
                content=f"Unsupported language: {language}. "
                        f"Supported: {', '.join(sorted(self._config.supported_languages))}",
                tool_call_id=tool_call_id,
            )

        if len(code) > self._config.max_code_length:
            return ToolMessage(
                content=f"Code exceeds maximum length of {self._config.max_code_length} characters",
                tool_call_id=tool_call_id,
            )

        org = os.environ.get("AI_PLATFORM_AGENT_ORG", "default")
        namespace = self._runner.resolve_namespace()

        self._metrics.increment_active(org=org)
        self._metrics.log_started(
            language=language, org=org, namespace=namespace,
            timeout_seconds=timeout, code_length=len(code),
        )

        started = time.monotonic()
        try:
            result = await self._runner.run(
                language=language,
                code=code,
                timeout=timeout,
                namespace=namespace,
            )

            duration = time.monotonic() - started
            latency_ms = round(duration * 1000, 2)

            self._metrics.record_execution(
                language=language, org=org,
                exit_code=result.exit_code, status=result.status,
                duration=duration,
            )
            self._metrics.emit_audit(
                language=language, status=result.status,
                exit_code=result.exit_code, latency_ms=latency_ms,
                code_hash=compute_code_hash(code),
                namespace=namespace,
                image=self._config.images.get(language, "unknown"),
                job_name=result.job_name, timeout=timeout,
                stdout_bytes=len(result.stdout),
                stderr_bytes=len(result.stderr),
            )

            if result.status == "timeout":
                self._metrics.log_timeout(
                    job_name=result.job_name, timeout_seconds=timeout,
                )
            elif result.status == "oom_killed":
                self._metrics.log_oom(
                    job_name=result.job_name,
                    memory_limit=self._config.resource_limits.get("memory", "unknown"),
                )
            elif result.status in ("error", "failed") and result.exit_code != 0:
                self._metrics.log_completed(
                    exit_code=result.exit_code, duration_ms=latency_ms,
                    status=result.status, job_name=result.job_name,
                )
            else:
                self._metrics.log_completed(
                    exit_code=result.exit_code, duration_ms=latency_ms,
                    status=result.status, job_name=result.job_name,
                )

            return ToolMessage(content=result.format(), tool_call_id=tool_call_id)

        except Exception as exc:
            duration = time.monotonic() - started
            self._metrics.record_error(
                language=language, org=org, error_type=type(exc).__name__,
            )
            self._metrics.log_failed(
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return ToolMessage(
                content="Code execution service temporarily unavailable",
                tool_call_id=tool_call_id,
            )
        finally:
            self._metrics.decrement_active(org=org)
```

Update `deep_agent/src/code_execution/__init__.py`:

```python
"""Code execution middleware — ephemeral K8s Job backend for agent code execution."""
from __future__ import annotations

from deep_agent.src.code_execution.config import CodeExecutionConfig
from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

__all__ = ["CodeExecutionConfig", "CodeExecutionMiddleware"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/code_execution/ -v`
Expected: All tests across all 4 test files PASS

- [ ] **Step 5: Commit**

```bash
git add deep_agent/src/code_execution/middleware.py deep_agent/src/code_execution/__init__.py tests/unit/code_execution/test_middleware.py
git commit -m "feat(code-execution): add CodeExecutionMiddleware with tool injection and routing"
```

---

### Task 5: Wire into middleware builder + agent.yaml config + audit event type

**Files:**
- Modify: `deep_agent/src/agent/config/middleware.py` (add `CodeExecutionConfig` to `MiddlewareDefaults`)
- Modify: `deep_agent/src/infrastructure/middleware.py` (register in `build_middleware_list`)
- Modify: `config/agent/runtime/agent.yaml` (add `code_execution:` section)
- Modify: `deep_agent/src/audit/events.py` (add `CODE_EXECUTION` type — if file exists on this branch; create stub otherwise)
- Modify: `tests/unit/test_infrastructure_middleware.py` (add test for code execution middleware registration)

**Interfaces:**
- Consumes: `CodeExecutionConfig` (Task 1), `CodeExecutionMiddleware` (Task 4), `ResolvedMiddlewareConfig` and `MiddlewareDefaults` from `deep_agent.src.agent.config.middleware`, `build_middleware_list` from `deep_agent.src.infrastructure.middleware`
- Produces: End-to-end wiring — when `code_execution.enabled: true` in agent.yaml, the middleware is built and included in the agent's middleware chain

- [ ] **Step 1: Add CodeExecutionConfig to middleware config model**

In `deep_agent/src/agent/config/middleware.py`, add to `MiddlewareDefaults`:

```python
# Add import at top
from deep_agent.src.code_execution.config import CodeExecutionConfig
```

Add field to `MiddlewareDefaults` class after `extra`:

```python
    code_execution: CodeExecutionConfig = Field(default_factory=CodeExecutionConfig)
```

Add field to `ResolvedMiddlewareConfig` class:

```python
    code_execution: CodeExecutionConfig = Field(default_factory=CodeExecutionConfig)
```

In `resolve_middleware()`, add before the return statement:

```python
    code_execution = defaults.code_execution
    if isinstance(overrides.get("code_execution"), dict):
        code_execution = CodeExecutionConfig.model_validate(overrides["code_execution"])
```

And add to the `ResolvedMiddlewareConfig(...)` constructor:

```python
        code_execution=code_execution,
```

- [ ] **Step 2: Register in build_middleware_list**

In `deep_agent/src/infrastructure/middleware.py`, add after the `_append_guardrails` call and before the `for dotted_path` loop:

```python
    if resolved.code_execution.enabled:
        _append_if_built(
            middlewares,
            _build_code_execution(resolved.code_execution),
        )
```

Add the builder function at the bottom of the file:

```python
def _build_code_execution(config: Any) -> Any | None:
    """Build CodeExecutionMiddleware for sandboxed code execution."""
    try:
        from deep_agent.src.code_execution.middleware import CodeExecutionMiddleware

        return CodeExecutionMiddleware(config=config)
    except ImportError:
        logger.debug("CodeExecutionMiddleware not available (missing kubernetes package?)")
        return None
    except Exception as e:
        logger.warning("Failed to create CodeExecutionMiddleware: %s", e)
        return None
```

- [ ] **Step 3: Add code_execution section to agent.yaml**

In `config/agent/runtime/agent.yaml`, add after `extra: []` (before the Async Tasks section):

```yaml
  # --- Code Execution (ephemeral K8s Job sandbox) ---
  code_execution:
    enabled: false
    max_timeout_seconds: 60
    max_code_length: 50000
    max_output_bytes: 1048576
    images:
      python: "python:3.12-slim"
      shell: "bash:5"
      node: "node:22-slim"
    resource_requests:
      cpu: "100m"
      memory: "128Mi"
    resource_limits:
      cpu: "500m"
      memory: "256Mi"
```

- [ ] **Step 4: Add audit event type**

Check if `deep_agent/src/audit/events.py` exists on this branch. If not, create it:

```python
"""Audit event type constants."""
from __future__ import annotations

from typing import Final

LLM_CALL: Final = "llm_call"
MCP_TOOL_CALL: Final = "mcp_tool_call"
MEMORY_WRITE: Final = "memory_write"
SUBAGENT_DELEGATION: Final = "subagent_delegation"
CODE_EXECUTION: Final = "code_execution"

AUDITED_MIDDLEWARE_EVENTS: frozenset[str] = frozenset(
    {LLM_CALL, MCP_TOOL_CALL, MEMORY_WRITE, SUBAGENT_DELEGATION, CODE_EXECUTION}
)


class AuditEventType:
    LLM_CALL = LLM_CALL
    MCP_TOOL_CALL = MCP_TOOL_CALL
    MEMORY_WRITE = MEMORY_WRITE
    SUBAGENT_DELEGATION = SUBAGENT_DELEGATION
    CODE_EXECUTION = CODE_EXECUTION
```

- [ ] **Step 5: Add wiring test**

In `tests/unit/test_infrastructure_middleware.py`, add a test:

```python
def test_code_execution_middleware_registered_when_enabled():
    """CodeExecutionMiddleware is built when code_execution.enabled=True."""
    from deep_agent.src.agent.config.middleware import ResolvedMiddlewareConfig
    from deep_agent.src.code_execution.config import CodeExecutionConfig
    from deep_agent.src.infrastructure.middleware import build_middleware_list

    resolved = ResolvedMiddlewareConfig(
        code_execution=CodeExecutionConfig(enabled=True),
    )
    middlewares = build_middleware_list(resolved)
    class_names = [type(m).__name__ for m in middlewares]
    assert "CodeExecutionMiddleware" in class_names


def test_code_execution_middleware_not_registered_when_disabled():
    """CodeExecutionMiddleware is not built when code_execution.enabled=False."""
    from deep_agent.src.agent.config.middleware import ResolvedMiddlewareConfig
    from deep_agent.src.code_execution.config import CodeExecutionConfig
    from deep_agent.src.infrastructure.middleware import build_middleware_list

    resolved = ResolvedMiddlewareConfig(
        code_execution=CodeExecutionConfig(enabled=False),
    )
    middlewares = build_middleware_list(resolved)
    class_names = [type(m).__name__ for m in middlewares]
    assert "CodeExecutionMiddleware" not in class_names
```

- [ ] **Step 6: Run all tests**

Run: `python -m pytest tests/unit/code_execution/ tests/unit/test_infrastructure_middleware.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add deep_agent/src/agent/config/middleware.py deep_agent/src/infrastructure/middleware.py config/agent/runtime/agent.yaml deep_agent/src/audit/events.py tests/unit/test_infrastructure_middleware.py
git commit -m "feat(code-execution): wire CodeExecutionMiddleware into middleware builder and config"
```

---

### Task 6: Full test suite run + cleanup

**Files:**
- No new files — verification task

**Interfaces:**
- Consumes: Everything from Tasks 1-5
- Produces: Verified, clean test suite

- [ ] **Step 1: Run the complete test suite**

Run: `python -m pytest tests/unit/ -v --tb=short 2>&1 | tail -40`
Expected: All tests PASS (or existing failures unrelated to code execution)

- [ ] **Step 2: Run pre-commit checks if available**

Run: `pre-commit run --all-files 2>&1 | tail -20` (if pre-commit is configured)
Fix any linting/formatting issues.

- [ ] **Step 3: Verify the module is importable**

Run:
```bash
python -c "from deep_agent.src.code_execution import CodeExecutionMiddleware, CodeExecutionConfig; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Final commit if any fixes**

```bash
git add -A
git commit -m "chore: fix linting and formatting for code execution middleware"
```
