"""A2A (Agent2Agent) executor bridging the template agent to the A2A protocol."""

from __future__ import annotations

from typing import Any

from a2a.helpers import new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue

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


async def _stream_to_final_text(
    manager: AgentManager,
    stream_request: StreamRequest,
) -> tuple[str | None, str]:
    """Return (error_message, assistant_text)."""
    last_ai = ""
    async for event in manager.stream_response(stream_request):
        et = event.get("type")
        if et == "error":
            content = event.get("content") or {}
            msg = str(content.get("message", "The agent returned an error."))
            return msg, ""
        if et == "message":
            content = event.get("content") or {}
            if content.get("type") == "ai":
                piece = content.get("content")
                if piece is not None and str(piece).strip():
                    last_ai = str(piece)
    return None, last_ai


class TemplateAgentA2AExecutor(AgentExecutor):
    """Runs ``AgentManager`` for each A2A task and returns the final assistant text."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        prompt = context.get_user_input().strip()
        task_id = context.task_id
        context_id = context.context_id
        if not prompt:
            await event_queue.enqueue_event(
                new_text_message(
                    "No text message was provided.",
                    task_id=task_id,
                    context_id=context_id,
                )
            )
            return

        # Token comes from the auth middleware via ContextVar (not metadata)
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
            stream_tokens=False,
        )

        try:
            manager = AgentManager(redhat_sso_token=token)
        except Exception as exc:
            logger.exception("A2A: AgentManager initialization failed: %s", exc)
            await event_queue.enqueue_event(
                new_text_message(
                    f"Agent initialization failed: {exc}",
                    task_id=task_id,
                    context_id=context_id,
                )
            )
            return

        try:
            error_text, last_ai = await _stream_to_final_text(manager, stream_request)
        except Exception as exc:
            logger.exception("A2A: stream_response failed: %s", exc)
            error_text, last_ai = str(exc), ""

        if error_text is not None:
            out = error_text
        elif last_ai.strip():
            out = last_ai
        else:
            out = "(no assistant response)"

        await event_queue.enqueue_event(
            new_text_message(out, task_id=task_id, context_id=context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Best-effort cancel; LangGraph stream cancellation is not wired here."""
        del context, event_queue
