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
    """Result of a code execution job."""

    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    status: str
    job_name: str
    namespace: str

    def format(self) -> str:
        """Format execution result as a human-readable string."""
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
        """Initialize runner with execution configuration."""
        self._config = config
        self._batch_api: Any | None = None
        self._core_api: Any | None = None

    def _ensure_k8s_client(self) -> None:
        if self._batch_api is not None:
            return
        try:
            from kubernetes import client
            from kubernetes import config as k8s_config

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
        """Build a K8s Job manifest for code execution."""
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
        """Parse container exit code and reason into a status tuple."""
        if exit_code == 0 and termination_reason is None:
            return exit_code, "success"
        if termination_reason == "OOMKilled":
            return exit_code, "oom_killed"
        if termination_reason == "DeadlineExceeded":
            return exit_code, "timeout"
        return exit_code, "failed"

    def resolve_namespace(self) -> str:
        """Resolve the K8s namespace from environment variables."""
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
        """Execute code in an ephemeral K8s Job and return the result."""
        self._ensure_k8s_client()
        assert self._batch_api is not None
        assert self._core_api is not None
        ns = namespace or self.resolve_namespace()
        execution_id = uuid.uuid4().hex
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

            pod_name = await self._wait_for_pod(job_name, ns)
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
                stdout="",
                stderr="",
                exit_code=-1,
                duration_seconds=duration,
                status="timeout",
                job_name=job_name,
                namespace=ns,
            )
        except Exception as exc:
            duration = time.monotonic() - started
            logger.error("code_execution_failed", error=str(exc), job_name=job_name)
            return ExecutionResult(
                stdout="",
                stderr=str(exc),
                exit_code=-1,
                duration_seconds=duration,
                status="error",
                job_name=job_name,
                namespace=ns,
            )
        finally:
            await self._cleanup(job_name, ns)

    async def _wait_for_pod(self, job_name: str, namespace: str) -> str:
        assert self._core_api is not None
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
                    return str(pod.metadata.name)
                if phase == "Running":
                    return str(pod.metadata.name)
            await asyncio.sleep(self._config.pod_poll_interval_seconds)
        raise asyncio.TimeoutError(
            f"Pod for {job_name} did not start within {self._config.pod_poll_timeout_seconds}s"
        )

    async def _collect_logs(self, pod_name: str, namespace: str) -> tuple[str, str]:
        assert self._core_api is not None
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

    async def _get_exit_info(
        self, pod_name: str, namespace: str
    ) -> tuple[int, str | None]:
        assert self._core_api is not None
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
        assert self._batch_api is not None
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
                "code_execution_cleanup_failed", job_name=job_name, error=str(exc)
            )
