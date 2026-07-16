"""K8s Job lifecycle manager for ephemeral code execution."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from deep_agent.src.code_execution.config import CodeExecutionConfig
from deep_agent.src.code_execution.metrics import _log_json


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
    output_files: dict[str, str] = field(default_factory=dict)
    cpu_seconds: float = 0.0
    memory_mb_seconds: float = 0.0

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
        self._networking_api: Any | None = None
        self._custom_api: Any | None = None

    def _ensure_k8s_client(self) -> None:
        """Lazy-initialize K8s API clients."""
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
            self._networking_api = client.NetworkingV1Api()
            self._custom_api = client.CustomObjectsApi()
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
        allow_network: bool = False,
        input_configmap_name: str | None = None,
    ) -> Any:
        """Build a K8s Job manifest for code execution."""
        from kubernetes import client

        labels = {
            "app.kubernetes.io/name": "code-execution",
            "app.kubernetes.io/component": "ephemeral-job",
            "app.kubernetes.io/managed-by": "template-agent",
            "ai-platform.io/execution-id": execution_id,
        }
        if allow_network:
            labels["ai-platform.io/allow-internet"] = "true"

        annotations = {}
        if trace_id:
            annotations["ai-platform.io/trace-id"] = trace_id

        volume_mounts = [
            client.V1VolumeMount(name="tmp", mount_path="/tmp"),
            client.V1VolumeMount(name="output", mount_path="/output"),
        ]
        volumes = [
            client.V1Volume(
                name="tmp",
                empty_dir=client.V1EmptyDirVolumeSource(
                    size_limit=self._config.tmp_size_limit,
                ),
            ),
            client.V1Volume(
                name="output",
                empty_dir=client.V1EmptyDirVolumeSource(size_limit="64Mi"),
            ),
        ]

        if input_configmap_name:
            volume_mounts.append(
                client.V1VolumeMount(name="input", mount_path="/input", read_only=True)
            )
            volumes.append(
                client.V1Volume(
                    name="input",
                    config_map=client.V1ConfigMapVolumeSource(
                        name=input_configmap_name
                    ),
                )
            )

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
            volume_mounts=volume_mounts,
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
            volumes=volumes,
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
        if termination_reason == "OOMKilled":
            return exit_code, "oom_killed"
        if termination_reason == "DeadlineExceeded":
            return exit_code, "timeout"
        if exit_code == 0:
            return exit_code, "success"
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
        allow_network: bool = False,
        input_files: dict[str, str] | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """Execute code in an ephemeral K8s Job and return the result."""
        self._ensure_k8s_client()
        assert self._batch_api is not None
        assert self._core_api is not None
        ns = namespace or self.resolve_namespace()
        execution_id = uuid.uuid4().hex
        started = time.monotonic()
        configmap_name: str | None = None
        network_policy_name: str | None = None
        job_name = f"code-exec-{execution_id[:8]}"

        try:
            if input_files:
                configmap_name = await self._create_input_configmap(
                    execution_id, ns, input_files
                )

            should_allow_network = (
                allow_network and self._config.network_access != "deny"
            ) or self._config.network_access == "allow_internet"
            network_policy_name = await self._create_network_policy(
                execution_id, ns, allow_internet=should_allow_network
            )

            manifest = self.build_job_manifest(
                language=language,
                code=code,
                timeout=timeout,
                namespace=ns,
                execution_id=execution_id,
                allow_network=allow_network,
                input_configmap_name=configmap_name,
            )
            job_name = manifest.metadata.name
            await asyncio.to_thread(
                self._batch_api.create_namespaced_job,
                namespace=ns,
                body=manifest,
            )
            _log_json(
                logging.INFO,
                "code_execution_job_created",
                job_name=job_name,
                namespace=ns,
            )

            if on_output and self._config.streaming_enabled:
                pod_name = await self._wait_for_pod(job_name, ns, wait_for_running=True)
                stdout, stderr = await self._collect_logs_streaming(
                    pod_name, ns, on_output
                )
                await self._wait_for_pod(job_name, ns, wait_for_running=False)
            else:
                pod_name = await self._wait_for_pod(job_name, ns)
                stdout, stderr = await self._collect_logs(pod_name, ns)

            raw_exit_code, termination_reason = await self._get_exit_info(pod_name, ns)
            exit_code, status = self.parse_container_status(
                exit_code=raw_exit_code,
                termination_reason=termination_reason,
            )

            cpu_seconds = 0.0
            memory_mb_seconds = 0.0
            if self._config.cost_tracking_enabled:
                cpu_seconds, memory_mb_seconds = await self._get_resource_usage(
                    pod_name, ns, time.monotonic() - started
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
                cpu_seconds=cpu_seconds,
                memory_mb_seconds=memory_mb_seconds,
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
            _log_json(
                logging.ERROR,
                "code_execution_failed",
                error=str(exc),
                job_name=job_name,
            )
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
            if configmap_name:
                await self._delete_configmap(configmap_name, ns)
            if network_policy_name:
                await self._delete_network_policy(network_policy_name, ns)

    async def _wait_for_pod(
        self,
        job_name: str,
        namespace: str,
        *,
        wait_for_running: bool = False,
    ) -> str:
        """Poll until pod reaches a target state."""
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
                if wait_for_running and phase == "Running":
                    return str(pod.metadata.name)
            await asyncio.sleep(self._config.pod_poll_interval_seconds)
        raise asyncio.TimeoutError(
            f"Pod for {job_name} did not reach target state "
            f"within {self._config.pod_poll_timeout_seconds}s"
        )

    async def _collect_logs(self, pod_name: str, namespace: str) -> tuple[str, str]:
        """Read stdout/stderr from pod logs after completion."""
        assert self._core_api is not None
        try:
            raw_logs = await asyncio.to_thread(
                self._core_api.read_namespaced_pod_log,
                name=pod_name,
                namespace=namespace,
                container="executor",
            )
            if isinstance(raw_logs, bytes):
                logs = raw_logs.decode("utf-8", errors="replace")
            else:
                logs = str(raw_logs)
                if logs.startswith("b'") or logs.startswith('b"'):
                    import ast

                    try:
                        logs = ast.literal_eval(logs).decode("utf-8", errors="replace")
                    except Exception:
                        pass
            if len(logs) > self._config.max_output_bytes:
                logs = logs[: self._config.max_output_bytes] + "\n[truncated at 1MB]"
            return logs, ""
        except Exception as exc:
            _log_json(
                logging.WARNING,
                "code_execution_log_collection_failed",
                error=str(exc),
            )
            return "", f"[warning: logs partially collected: {exc}]"

    async def _collect_logs_streaming(
        self,
        pod_name: str,
        namespace: str,
        callback: Callable[[str], None],
    ) -> tuple[str, str]:
        """Stream pod logs in real-time via callback, return full output."""
        assert self._core_api is not None
        _log_json(logging.INFO, "code_execution_streaming_started", pod=pod_name)
        accumulated: list[str] = []
        total_bytes = 0
        try:
            response = await asyncio.to_thread(
                self._core_api.read_namespaced_pod_log,
                name=pod_name,
                namespace=namespace,
                container="executor",
                follow=True,
                _preload_content=False,
            )

            def _read_stream() -> list[str]:
                chunks: list[str] = []
                nonlocal total_bytes
                for chunk in response.stream(512):
                    text = (
                        chunk.decode("utf-8", errors="replace")
                        if isinstance(chunk, bytes)
                        else str(chunk)
                    )
                    total_bytes += len(text)
                    if total_bytes <= self._config.max_output_bytes:
                        chunks.append(text)
                    else:
                        chunks.append("\n[truncated at 1MB]")
                        break
                response.release_conn()
                return chunks

            stream_chunks = await asyncio.to_thread(_read_stream)
            for text in stream_chunks:
                accumulated.append(text)
                callback(text)
            _log_json(
                logging.INFO,
                "code_execution_streaming_completed",
                pod=pod_name,
                total_bytes=total_bytes,
            )
            return "".join(accumulated), ""
        except Exception as exc:
            _log_json(
                logging.WARNING,
                "code_execution_streaming_failed",
                error=str(exc),
            )
            return "".join(accumulated), f"[streaming error: {exc}]"

    async def _get_exit_info(
        self, pod_name: str, namespace: str
    ) -> tuple[int, str | None]:
        """Extract exit code and termination reason from container status."""
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
            phase = pod.status.phase
            if phase == "Succeeded":
                return 0, None
            if phase == "Failed":
                return 1, None
            return -1, None
        except Exception:
            return -1, None

    async def _get_resource_usage(
        self, pod_name: str, namespace: str, duration: float
    ) -> tuple[float, float]:
        """Query K8s Metrics API for pod resource usage."""
        assert self._custom_api is not None
        try:
            metrics = await asyncio.to_thread(
                self._custom_api.get_namespaced_custom_object,
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=namespace,
                plural="pods",
                name=pod_name,
            )
            containers = metrics.get("containers", [])
            for c in containers:
                if c.get("name") == "executor":
                    usage = c.get("usage", {})
                    cpu_nano = self._parse_cpu(usage.get("cpu", "0"))
                    memory_bytes = self._parse_memory(usage.get("memory", "0"))
                    cpu_seconds = (cpu_nano / 1e9) * duration
                    memory_mb_seconds = (memory_bytes / (1024 * 1024)) * duration
                    return cpu_seconds, memory_mb_seconds
            return 0.0, 0.0
        except Exception:
            cpu_req = self._parse_cpu(self._config.resource_requests.get("cpu", "100m"))
            mem_req = self._parse_memory(
                self._config.resource_requests.get("memory", "128Mi")
            )
            cpu_seconds = (cpu_req / 1e9) * duration
            memory_mb_seconds = (mem_req / (1024 * 1024)) * duration
            return cpu_seconds, memory_mb_seconds

    @staticmethod
    def _parse_cpu(value: str) -> float:
        """Parse K8s CPU value to nanocores."""
        value = str(value).strip()
        if value.endswith("n"):
            return float(value[:-1])
        if value.endswith("m"):
            return float(value[:-1]) * 1e6
        return float(value) * 1e9

    @staticmethod
    def _parse_memory(value: str) -> float:
        """Parse K8s memory value to bytes."""
        value = str(value).strip()
        suffixes = {
            "Ki": 1024,
            "Mi": 1024**2,
            "Gi": 1024**3,
            "Ti": 1024**4,
            "k": 1000,
            "M": 1000**2,
            "G": 1000**3,
        }
        for suffix, multiplier in suffixes.items():
            if value.endswith(suffix):
                return float(value[: -len(suffix)]) * multiplier
        return float(value)

    async def _create_network_policy(
        self,
        execution_id: str,
        namespace: str,
        *,
        allow_internet: bool = False,
    ) -> str:
        """Create an ephemeral NetworkPolicy for this execution."""
        assert self._networking_api is not None
        from kubernetes import client

        policy_name = f"code-exec-{execution_id[:8]}"
        selector = {
            "matchLabels": {
                "ai-platform.io/execution-id": execution_id,
            }
        }

        if allow_internet:
            egress = [
                client.V1NetworkPolicyEgressRule(
                    ports=[
                        client.V1NetworkPolicyPort(port=443, protocol="TCP"),
                        client.V1NetworkPolicyPort(port=80, protocol="TCP"),
                    ],
                    to=[
                        client.V1NetworkPolicyPeer(
                            ip_block=client.V1IPBlock(
                                cidr="0.0.0.0/0",
                                _except=[
                                    "10.0.0.0/8",
                                    "172.16.0.0/12",
                                    "192.168.0.0/16",
                                ],
                            )
                        )
                    ],
                )
            ]
        else:
            egress = []

        policy = client.V1NetworkPolicy(
            api_version="networking.k8s.io/v1",
            kind="NetworkPolicy",
            metadata=client.V1ObjectMeta(
                name=policy_name,
                namespace=namespace,
            ),
            spec=client.V1NetworkPolicySpec(
                pod_selector=client.V1LabelSelector(**selector),
                policy_types=["Egress"],
                egress=egress,
            ),
        )

        await asyncio.to_thread(
            self._networking_api.create_namespaced_network_policy,
            namespace=namespace,
            body=policy,
        )
        _log_json(
            logging.INFO,
            "code_execution_network_policy_created",
            policy=policy_name,
            allow_internet=allow_internet,
        )
        return policy_name

    async def _delete_network_policy(self, policy_name: str, namespace: str) -> None:
        """Delete the ephemeral NetworkPolicy."""
        assert self._networking_api is not None
        try:
            await asyncio.to_thread(
                self._networking_api.delete_namespaced_network_policy,
                name=policy_name,
                namespace=namespace,
            )
            _log_json(
                logging.DEBUG,
                "code_execution_network_policy_deleted",
                policy=policy_name,
            )
        except Exception as exc:
            _log_json(
                logging.WARNING,
                "code_execution_network_policy_delete_failed",
                policy=policy_name,
                error=str(exc),
            )

    async def _create_input_configmap(
        self,
        execution_id: str,
        namespace: str,
        files: dict[str, str],
    ) -> str:
        """Create a ConfigMap with input files for the execution pod."""
        assert self._core_api is not None
        from kubernetes import client

        cm_name = f"code-exec-input-{execution_id[:8]}"
        configmap = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            metadata=client.V1ObjectMeta(
                name=cm_name,
                namespace=namespace,
                labels={
                    "app.kubernetes.io/managed-by": "template-agent",
                    "ai-platform.io/execution-id": execution_id,
                },
            ),
            data=files,
        )
        await asyncio.to_thread(
            self._core_api.create_namespaced_config_map,
            namespace=namespace,
            body=configmap,
        )
        _log_json(
            logging.INFO,
            "code_execution_configmap_created",
            configmap=cm_name,
            file_count=len(files),
        )
        return cm_name

    async def _delete_configmap(self, cm_name: str, namespace: str) -> None:
        """Delete the input ConfigMap."""
        assert self._core_api is not None
        try:
            await asyncio.to_thread(
                self._core_api.delete_namespaced_config_map,
                name=cm_name,
                namespace=namespace,
            )
            _log_json(
                logging.DEBUG,
                "code_execution_configmap_deleted",
                configmap=cm_name,
            )
        except Exception as exc:
            _log_json(
                logging.WARNING,
                "code_execution_configmap_delete_failed",
                configmap=cm_name,
                error=str(exc),
            )

    async def _cleanup(self, job_name: str, namespace: str) -> None:
        """Delete Job with Foreground propagation (cascades to pods)."""
        assert self._batch_api is not None
        try:
            from kubernetes import client

            await asyncio.to_thread(
                self._batch_api.delete_namespaced_job,
                name=job_name,
                namespace=namespace,
                body=client.V1DeleteOptions(propagation_policy="Foreground"),
            )
            _log_json(logging.DEBUG, "code_execution_cleanup", job_name=job_name)
        except Exception as exc:
            _log_json(
                logging.WARNING,
                "code_execution_cleanup_failed",
                job_name=job_name,
                error=str(exc),
            )
