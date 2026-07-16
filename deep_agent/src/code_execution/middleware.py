"""CodeExecutionMiddleware — inject execute_code tool, route to K8s Jobs."""

from __future__ import annotations

import asyncio
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


def _build_execute_code_tool(config: CodeExecutionConfig) -> Any:
    """Build the execute_code tool definition for LLM tool binding."""

    @tool
    def execute_code(
        code: str,
        language: str = "python",
        timeout: int = 60,
        network: bool = False,
        input_files: dict[str, str] | None = None,
    ) -> str:
        """Execute code in an isolated sandbox environment.

        Args:
            code: The source code to execute.
            language: Programming language (python, python-ds, python-ml, shell, node).
            timeout: Maximum execution time in seconds.
            network: Whether to allow internet access from the sandbox.
            input_files: Optional dict of filename to content, mounted at /input/.

        Returns:
            Execution output with stdout, stderr, and exit code.
        """
        return "This tool is handled by CodeExecutionMiddleware"

    return execute_code


class CodeExecutionMiddleware(AgentMiddleware):
    """Inject execute_code tool and route calls to K8s Job backend."""

    def __init__(self, *, config: CodeExecutionConfig) -> None:
        """Initialize middleware with execution configuration."""
        self._config = config
        self._runner = K8sJobRunner(config)
        self._metrics = CodeExecutionMetrics()
        self._execute_code_tool = _build_execute_code_tool(config)
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def _get_semaphore(self, org: str) -> asyncio.Semaphore:
        """Get or create a per-org execution semaphore."""
        if org not in self._semaphores:
            self._semaphores[org] = asyncio.Semaphore(
                self._config.max_concurrent_per_org
            )
        return self._semaphores[org]

    def wrap_model_call(
        self, request: ModelRequest[Any], handler: Any
    ) -> ModelResponse[Any]:
        """Synchronous model call pass-through."""
        return handler(request)

    async def awrap_model_call(
        self, request: ModelRequest[Any], handler: Any
    ) -> ModelResponse[Any]:
        """Inject the execute_code tool into model requests when enabled."""
        if not self._config.enabled:
            return await handler(request)
        updated = request.override(tools=[*request.tools, self._execute_code_tool])
        return await handler(updated)

    def wrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        """Synchronous tool call pass-through."""
        return handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        """Intercept execute_code tool calls and route to K8s backend."""
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
        network = bool(args.get("network", False))
        input_files = args.get("input_files")
        tool_call_id = tool_call.get("id", "")

        if not code.strip():
            return ToolMessage(
                content="No code provided to execute",
                tool_call_id=tool_call_id,
            )

        if language not in self._config.supported_languages:
            return ToolMessage(
                content=f"Unsupported language: {language}. "
                f"Supported: {', '.join(sorted(self._config.supported_languages))}",
                tool_call_id=tool_call_id,
            )

        if len(code) > self._config.max_code_length:
            return ToolMessage(
                content=f"Code exceeds maximum length of "
                f"{self._config.max_code_length} characters",
                tool_call_id=tool_call_id,
            )

        if input_files:
            total_size = sum(len(v) for v in input_files.values())
            if total_size > self._config.max_input_file_size:
                return ToolMessage(
                    content=f"Input files exceed maximum size of "
                    f"{self._config.max_input_file_size} bytes",
                    tool_call_id=tool_call_id,
                )

        if network and self._config.network_access == "deny":
            network = False

        org = os.environ.get("AI_PLATFORM_AGENT_ORG", "default")
        namespace = self._runner.resolve_namespace()
        semaphore = self._get_semaphore(org)

        self._metrics.log_queued(org=org)
        queue_start = time.monotonic()
        acquired = False
        try:
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=self._config.queue_timeout_seconds,
            )
            acquired = True
        except asyncio.TimeoutError:
            self._metrics.record_rejected(org=org)
            return ToolMessage(
                content="Code execution queue full, try again later",
                tool_call_id=tool_call_id,
            )
        queue_wait = time.monotonic() - queue_start
        self._metrics.record_queue_wait(org=org, duration=queue_wait)
        self._metrics.log_dequeued(org=org, wait_seconds=queue_wait)

        self._metrics.increment_active(org=org)
        self._metrics.log_started(
            language=language,
            org=org,
            namespace=namespace,
            timeout_seconds=timeout,
            code_length=len(code),
            network=network,
            input_file_count=len(input_files) if input_files else 0,
        )

        on_output = None
        if self._config.streaming_enabled:
            try:
                from langgraph.config import get_stream_writer

                stream_writer = get_stream_writer()

                def on_output(chunk: str) -> None:
                    stream_writer({"type": "code_output", "content": chunk})
            except Exception:
                pass

        started = time.monotonic()
        try:
            result = await self._runner.run(
                language=language,
                code=code,
                timeout=timeout,
                namespace=namespace,
                allow_network=network,
                input_files=input_files,
                on_output=on_output,
            )

            duration = time.monotonic() - started
            latency_ms = round(duration * 1000, 2)

            self._metrics.record_execution(
                language=language,
                org=org,
                exit_code=result.exit_code,
                status=result.status,
                duration=duration,
            )

            if self._config.cost_tracking_enabled:
                self._metrics.record_resource_usage(
                    org=org,
                    language=language,
                    cpu_seconds=result.cpu_seconds,
                    memory_mb_seconds=result.memory_mb_seconds,
                    duration=duration,
                )

            self._metrics.emit_audit(
                language=language,
                status=result.status,
                exit_code=result.exit_code,
                latency_ms=latency_ms,
                code_hash=compute_code_hash(code),
                namespace=namespace,
                image=self._config.images.get(language, "unknown"),
                job_name=result.job_name,
                timeout=timeout,
                stdout_bytes=len(result.stdout),
                stderr_bytes=len(result.stderr),
            )

            if result.status == "timeout":
                self._metrics.log_timeout(
                    job_name=result.job_name,
                    timeout_seconds=timeout,
                )
            elif result.status == "oom_killed":
                self._metrics.log_oom(
                    job_name=result.job_name,
                    memory_limit=self._config.resource_limits.get("memory", "unknown"),
                )
            else:
                self._metrics.log_completed(
                    exit_code=result.exit_code,
                    duration_ms=latency_ms,
                    status=result.status,
                    job_name=result.job_name,
                )

            return ToolMessage(content=result.format(), tool_call_id=tool_call_id)

        except Exception as exc:
            duration = time.monotonic() - started
            self._metrics.record_error(
                language=language,
                org=org,
                error_type=type(exc).__name__,
            )
            self._metrics.log_failed(
                error_type=type(exc).__name__,
                error_message=str(exc),
                duration_ms=round(duration * 1000, 2),
            )
            return ToolMessage(
                content="Code execution service temporarily unavailable",
                tool_call_id=tool_call_id,
            )
        finally:
            self._metrics.decrement_active(org=org)
            if acquired:
                semaphore.release()
