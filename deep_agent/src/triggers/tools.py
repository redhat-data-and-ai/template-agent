"""Tools for interacting with the headless worker from the server agent."""

from __future__ import annotations

import json
from typing import Any

from deep_agent.src.settings import settings
from deep_agent.src.triggers.task_store import TaskStore
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_DEFAULT_STREAM = "agent-tasks"
_store = TaskStore()


async def queue_task(
    task_name: str,
    payload: dict[str, Any],
    thread_id: str = "",
    user_id: str = "",
    stream: str = _DEFAULT_STREAM,
) -> str:
    """Queue a background task for the headless worker.

    Use this for long-running or bulk work that doesn't need an
    immediate response. The headless worker picks it up from Redis
    and processes it asynchronously.

    Args:
        task_name: A descriptive name (e.g. "generate-report").
        payload: The task data sent to the headless worker.
        thread_id: Current conversation thread ID (for tracking).
        user_id: Current user ID (for result delivery).
        stream: Redis Stream name. Defaults to "agent-tasks".

    Returns:
        Confirmation with the task ID that can be used to check status.
    """
    record = await _store.create_task(
        task_name=task_name,
        payload=payload,
        thread_id=thread_id or None,
        user_id=user_id or None,
    )

    import redis.asyncio as aioredis

    client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        fields = {
            "name": task_name,
            "task_id": record.task_id,
            "payload": json.dumps(payload),
        }
        await client.xadd(stream, fields)
        logger.info(
            "task queued for headless worker",
            task_id=record.task_id,
            task_name=task_name,
            stream=stream,
        )
    finally:
        await client.aclose()

    return (
        f"Task '{task_name}' queued with ID {record.task_id}. "
        f"The headless worker will process it in the background. "
        f"Use check_task_status with this ID to check progress."
    )


async def check_task_status(task_id: str) -> str:
    """Check the status of a background task.

    Args:
        task_id: The task ID returned by queue_task.

    Returns:
        Task status and result if completed.
    """
    record = await _store.get_task(task_id)
    if record is None:
        return f"Task '{task_id}' not found. It may have expired (tasks are kept for 24 hours)."

    if record.status == "completed":
        result_text = str(record.result) if record.result else "No output"
        return (
            f"Task '{record.task_name}' (ID: {task_id}) is COMPLETED.\n"
            f"Result:\n{result_text}"
        )
    elif record.status == "failed":
        return (
            f"Task '{record.task_name}' (ID: {task_id}) FAILED.\nError: {record.error}"
        )
    else:
        return f"Task '{record.task_name}' (ID: {task_id}) is {record.status.upper()}."


async def get_pending_results(user_id: str) -> str:
    """Get completed background tasks that haven't been delivered yet.

    Call this at the start of a conversation to check if any
    background tasks have completed since the user's last visit.

    Args:
        user_id: The user's identity.

    Returns:
        Summary of pending results, or a message saying there are none.
    """
    pending = await _store.get_pending_results(user_id)
    if not pending:
        return "No pending background task results."

    lines = [f"{len(pending)} background task(s) completed since your last visit:\n"]
    for record in pending:
        if record.status == "completed":
            result_text = str(record.result) if record.result else "No output"
            lines.append(
                f"- **{record.task_name}** (ID: {record.task_id}): COMPLETED\n  Result:\n  {result_text}\n"
            )
        elif record.status == "failed":
            lines.append(
                f"- **{record.task_name}** (ID: {record.task_id}): FAILED\n  Error: {record.error}\n"
            )
        await _store.mark_delivered(record.task_id)

    return "\n".join(lines)


def get_builtin_tools() -> list[Any]:
    """Return LangChain-compatible tool objects for the headless worker tools."""
    from langchain_core.tools import StructuredTool

    return [
        StructuredTool.from_function(
            coroutine=queue_task,
            name="queue_task",
            description=(
                "Queue a background task for the headless worker. Use for long-running, "
                "bulk, or fire-and-forget work. Returns a task ID for status tracking."
            ),
        ),
        StructuredTool.from_function(
            coroutine=check_task_status,
            name="check_task_status",
            description="Check the status of a background task by its task ID.",
        ),
        StructuredTool.from_function(
            coroutine=get_pending_results,
            name="get_pending_results",
            description=(
                "Get completed background tasks not yet delivered to the user. "
                "Call at the start of every conversation to check for results."
            ),
        ),
    ]
