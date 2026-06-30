"""EventTriggerMiddleware — orchestrates trigger sources, graph invocation, and output sinks."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from deep_agent.src.triggers.config import HeadlessConfig
from deep_agent.src.triggers.sinks.protocol import OutputSink, TriggerResult
from deep_agent.src.triggers.sources.protocol import TriggerEvent, TriggerSource
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


class EventTriggerMiddleware:
    """Owns the lifecycle of trigger sources and output sinks in headless mode.

    Consumes events from all enabled trigger sources, invokes the agent graph
    per event, and fans out results to all configured output sinks.
    """

    def __init__(
        self,
        config: HeadlessConfig,
        graph: Any,
        redis_url: str = "redis://redis:6379/0",
    ) -> None:
        """Initialize the event trigger middleware with config, graph, and Redis URL."""
        self._config = config
        self._graph = graph
        self._redis_url = redis_url
        self._sources: list[TriggerSource] = []
        self._sinks: list[OutputSink] = []
        self._stop_event = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None

        from deep_agent.src.triggers.task_store import TaskStore

        self._task_store: TaskStore | None = (
            TaskStore(redis_url=self._redis_url) if self._redis_url else None
        )

    def _build_sources(self) -> list[TriggerSource]:
        """Build trigger sources from configuration."""
        sources: list[TriggerSource] = []
        tc = self._config.triggers

        if tc.webhook.enabled:
            from deep_agent.src.triggers.sources.webhook import WebhookTriggerSource

            sources.append(WebhookTriggerSource(tc.webhook))

        if tc.cron.enabled:
            from deep_agent.src.triggers.sources.cron import CronTriggerSource

            sources.append(CronTriggerSource(tc.cron))

        if tc.queue.enabled:
            from deep_agent.src.triggers.sources.queue import QueueTriggerSource

            sources.append(QueueTriggerSource(tc.queue, redis_url=self._redis_url))

        return sources

    def _build_sinks(self) -> list[OutputSink]:
        """Build output sinks from configuration."""
        sinks: list[OutputSink] = []

        if not self._config.output_sinks:
            from deep_agent.src.triggers.sinks.stdout import StdoutSink

            return [StdoutSink()]

        for sc in self._config.output_sinks:
            if sc.type == "stdout":
                from deep_agent.src.triggers.sinks.stdout import StdoutSink

                sinks.append(StdoutSink())
            elif sc.type == "file":
                from deep_agent.src.triggers.sinks.file import FileSink

                sinks.append(FileSink(path=sc.path or "output.jsonl"))
            elif sc.type == "webhook":
                from deep_agent.src.triggers.sinks.webhook import WebhookSink

                sinks.append(WebhookSink(url=sc.url or "", headers=sc.headers or None))
            elif sc.type == "redis":
                from deep_agent.src.triggers.sinks.redis import RedisSink

                sinks.append(
                    RedisSink(
                        stream=sc.stream or "agent-results", redis_url=self._redis_url
                    )
                )
            else:
                logger.warning("Unknown output sink type: %s", sc.type)

        return sinks

    async def start(self) -> None:
        """Start all trigger sources and begin the event processing loop."""
        logger.info("EventTriggerMiddleware starting")
        self._sources = self._build_sources()
        self._sinks = self._build_sinks()

        for source in self._sources:
            await source.start()

        self._stop_event.clear()
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info(
            "EventTriggerMiddleware started: %d source(s), %d sink(s)",
            len(self._sources),
            len(self._sinks),
        )

    async def _run_loop(self) -> None:
        async def _consume_source(source: TriggerSource) -> None:
            async for event in source:
                if self._stop_event.is_set():
                    return
                logger.info("Event received: %s (source=%s)", event.name, event.source)
                await self._process_event(event)

        try:
            async with asyncio.TaskGroup() as tg:
                for source in self._sources:
                    tg.create_task(_consume_source(source))

                async def _wait_for_stop() -> None:
                    await self._stop_event.wait()
                    raise _StopSentinel()

                tg.create_task(_wait_for_stop())
        except* _StopSentinel:
            pass
        except* Exception as eg:
            for exc in eg.exceptions:
                logger.exception("Source consumer error: %s", exc)

    async def _process_event(self, event: TriggerEvent) -> None:
        store = self._task_store
        task_id = event.payload.get("task_id") or event.metadata.get("task_id")

        if store and not task_id:
            record = await store.create_task(
                task_name=event.name,
                payload=event.payload,
                user_id=event.payload.get("user_id"),
            )
            task_id = record.task_id
            logger.info(
                "Auto-created task record for %s event",
                event.source,
                task_id=task_id,
                event_name=event.name,
            )

        if store and task_id:
            await store.update_status(task_id, "processing")

        graph_timeout = self._config.drain_timeout * 4
        t0 = time.monotonic()
        try:
            try:
                output = await asyncio.wait_for(
                    self._graph.ainvoke(
                        {
                            "messages": [
                                {"role": "user", "content": json.dumps(event.payload)}
                            ]
                        }
                    ),
                    timeout=graph_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "graph_invocation_timeout",
                    task_id=task_id,
                    timeout_seconds=graph_timeout,
                )
                output = None

            duration_ms = (time.monotonic() - t0) * 1000
            result = TriggerResult(
                event=event,
                output=output,
                duration_ms=duration_ms,
                success=output is not None,
            )
            if store and task_id:
                if output is not None:
                    await store.update_status(
                        task_id, "completed", result=_extract_result(output)
                    )
                else:
                    await store.update_status(
                        task_id, "failed", error="graph invocation timed out"
                    )
        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            result = TriggerResult(
                event=event,
                output=None,
                duration_ms=duration_ms,
                success=False,
                error=str(exc),
            )
            if store and task_id:
                await store.update_status(task_id, "failed", error=str(exc))
            logger.exception("Graph invocation failed for event: %s", event.name)

        # Ack queue message after processing (B3 fix: ack-after-processing).
        queue_msg = event.metadata.get("_queue_message")
        queue_consumer = event.metadata.get("_consumer")
        if queue_msg and queue_consumer:
            try:
                await queue_consumer.ack(queue_msg)
            except Exception:
                logger.warning("failed_to_ack_message", task_id=task_id)

        await self._emit_result(result)
        logger.info(
            "Event processed: %s (success=%s, duration=%.1fms)",
            event.name,
            result.success,
            result.duration_ms,
        )

    async def _emit_result(self, result: TriggerResult) -> None:
        for sink in self._sinks:
            try:
                await sink.emit(result)
            except Exception:
                logger.exception("Sink error (%s)", type(sink).__name__)

    async def stop(self) -> None:
        """Stop all trigger sources and close output sinks."""
        logger.info("EventTriggerMiddleware stopping")
        self._stop_event.set()

        if self._loop_task is not None:
            try:
                await asyncio.wait_for(
                    self._loop_task, timeout=self._config.drain_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Drain timeout (%.1fs) exceeded, cancelling",
                    self._config.drain_timeout,
                )
                self._loop_task.cancel()
                try:
                    await self._loop_task
                except asyncio.CancelledError:
                    pass
            self._loop_task = None

        for source in self._sources:
            try:
                await source.stop()
            except Exception:
                logger.exception("Error stopping source %s", type(source).__name__)

        for sink in self._sinks:
            try:
                await sink.close()
            except Exception:
                logger.exception("Error closing sink %s", type(sink).__name__)

        if self._task_store:
            try:
                await self._task_store.close()
            except Exception:
                logger.exception("Error closing task store")
            self._task_store = None

        logger.info("EventTriggerMiddleware stopped")


def _extract_result(output: Any) -> str:
    """Extract human-readable result text from graph output."""
    if isinstance(output, dict):
        messages = output.get("messages", [])
        for msg in reversed(messages):
            content = getattr(msg, "content", None) or msg.get("content", "")
            if isinstance(content, list):
                texts = [
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("text")
                ]
                if texts:
                    return "\n".join(texts)
            elif isinstance(content, str) and content.strip():
                role = getattr(msg, "type", None) or msg.get("type", "")
                if role in ("ai", "assistant"):
                    return content.strip()
    return str(output)[:2000]


class _StopSentinel(BaseException):
    """Raised to break out of the TaskGroup when stop is requested."""
