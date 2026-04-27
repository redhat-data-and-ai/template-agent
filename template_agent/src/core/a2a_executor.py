"""A2A (Agent2Agent) executor bridging the template agent to the A2A protocol.

Supports both non-streaming (SendMessage) and streaming (SendStreamingMessage)
callers. The same ``execute`` method pushes incremental events to the
EventQueue; the SDK's DefaultRequestHandler decides whether to collect them
into a single response or yield them as SSE.

The SDK enforces two mutually exclusive modes:
  - **Message mode**: emit a single ``Message`` (no Task)
  - **Task mode**: emit ``Task`` first, then ``TaskArtifactUpdateEvent`` /
    ``TaskStatusUpdateEvent`` only (no ``Message``)

We use task mode so streaming callers receive incremental SSE events.
Non-streaming callers still get the final completed Task with artifacts.
"""

from __future__ import annotations

import uuid
from typing import Any

from a2a.helpers import new_text_artifact_update_event, new_text_status_update_event
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Task, TaskState, TaskStatus

from template_agent.src.a2a.context import a2a_request_ctx
from template_agent.src.core.manager import AgentManager
from template_agent.src.schema import StreamRequest
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def _meta_str(metadata: dict[str, Any], key: str, default: str) -> str:
    raw = metadata.get(key)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return default


def _optional_thread_id(metadata: dict[str, Any]) -> str | None:
    raw = metadata.get("thread_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


class TemplateAgentA2AExecutor(AgentExecutor):
    """Runs ``AgentManager`` for each A2A task, streaming incremental events.

    Event flow:
    1. ``Task(WORKING)`` to establish the task in the SDK
    2. ``TaskArtifactUpdateEvent(append=True)`` for each token chunk
    3. ``TaskArtifactUpdateEvent(last_chunk=True)`` with the complete response
    4. ``TaskStatusUpdateEvent(COMPLETED)`` to signal the task is done
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        prompt = context.get_user_input().strip()
        task_id = context.task_id
        context_id = context.context_id
        if not prompt:
            await event_queue.enqueue_event(
                Task(
                    id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                )
            )
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task_id,
                    context_id=context_id,
                    state=TaskState.TASK_STATE_COMPLETED,
                    text="No text message was provided.",
                )
            )
            return

        ctx = a2a_request_ctx.get()
        token = ctx.access_token

        metadata = context.metadata
        user_id = _meta_str(metadata, "user_id", "a2a")
        session_id = _meta_str(metadata, "session_id", context_id or "a2a")
        thread_id = _optional_thread_id(metadata)

        if ctx.correlation_id:
            logger.info(
                "a2a_execute",
                correlation_id=ctx.correlation_id,
                calling_agent=ctx.calling_agent_id,
            )

        stream_request = StreamRequest(
            message=prompt,
            thread_id=thread_id,
            session_id=session_id,
            user_id=user_id,
            stream_tokens=True,
        )

        try:
            manager = AgentManager(redhat_sso_token=token)
        except Exception as exc:
            logger.exception("A2A: AgentManager initialization failed: %s", exc)
            await event_queue.enqueue_event(
                Task(
                    id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_FAILED),
                )
            )
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task_id,
                    context_id=context_id,
                    state=TaskState.TASK_STATE_FAILED,
                    text=f"Agent initialization failed: {exc}",
                )
            )
            return

        artifact_id = str(uuid.uuid4())

        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            )
        )

        last_ai = ""
        error_text: str | None = None

        try:
            async for event in manager.stream_response(stream_request):
                et = event.get("type")

                if et == "error":
                    content = event.get("content") or {}
                    error_text = str(
                        content.get("message", "The agent returned an error.")
                    )
                    break

                if et == "token":
                    chunk = str(event.get("content", ""))
                    if chunk:
                        await event_queue.enqueue_event(
                            new_text_artifact_update_event(
                                task_id=task_id,
                                context_id=context_id,
                                name="response",
                                text=chunk,
                                append=True,
                                last_chunk=False,
                                artifact_id=artifact_id,
                            )
                        )

                if et == "message":
                    content = event.get("content") or {}
                    if content.get("type") == "ai":
                        piece = content.get("content")
                        if piece is not None and str(piece).strip():
                            last_ai = str(piece)

        except Exception as exc:
            logger.exception("A2A: stream_response failed: %s", exc)
            error_text = str(exc)

        if error_text is not None:
            out = error_text
            final_state = TaskState.TASK_STATE_FAILED
        elif last_ai.strip():
            out = last_ai
            final_state = TaskState.TASK_STATE_COMPLETED
        else:
            out = "(no assistant response)"
            final_state = TaskState.TASK_STATE_COMPLETED

        await event_queue.enqueue_event(
            new_text_artifact_update_event(
                task_id=task_id,
                context_id=context_id,
                name="response",
                text=out,
                append=False,
                last_chunk=True,
                artifact_id=artifact_id,
            )
        )

        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id=task_id,
                context_id=context_id,
                state=final_state,
                text=out,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Best-effort cancel; LangGraph stream cancellation is not wired here."""
        del context, event_queue
