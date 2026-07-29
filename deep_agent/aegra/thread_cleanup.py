"""Thread deletion with full data cleanup.

Overrides Aegra's default DELETE /threads/{thread_id} to also purge:
- LangGraph checkpoint history (checkpoints, blobs, writes)
- User feedback (message_feedback table)
- Token usage records (MongoDB, if configured)

Aegra's built-in delete only removes the thread row and cascades to runs.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from deep_agent.aegra.auth_helpers import authenticated_user_id
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

thread_cleanup_router = APIRouter(tags=["threads"])


async def _delete_checkpoints(thread_id: str) -> int:
    """Delete all LangGraph checkpoint data for a thread."""
    try:
        import psycopg

        async with await psycopg.AsyncConnection.connect(settings.database_uri) as conn:
            deleted = 0
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                cur = await conn.execute(
                    f"DELETE FROM {table} WHERE thread_id = %s",
                    (thread_id,),
                )
                deleted += cur.rowcount
            await conn.commit()
        return deleted
    except Exception:
        logger.warning("checkpoint_cleanup_failed", thread_id=thread_id, exc_info=True)
        return 0


async def _delete_feedback(thread_id: str) -> int:
    """Delete all feedback entries for a thread."""
    try:
        import psycopg

        async with await psycopg.AsyncConnection.connect(settings.database_uri) as conn:
            cur = await conn.execute(
                "DELETE FROM message_feedback WHERE thread_id = %s",
                (thread_id,),
            )
            await conn.commit()
            count: int = cur.rowcount
            return count
    except Exception:
        logger.warning("feedback_cleanup_failed", thread_id=thread_id, exc_info=True)
        return 0


async def _delete_token_usage(thread_id: str) -> bool:
    """Delete token usage records for a thread from MongoDB."""
    try:
        if not settings.MONGODB_URI:
            return False

        from deep_agent.src.token_budget.service import _mongo_repo

        repo = _mongo_repo()
        result = await repo._thread_collection().delete_many({"thread_id": thread_id})
        return bool(result.deleted_count > 0)
    except Exception:
        logger.warning("token_usage_cleanup_failed", thread_id=thread_id, exc_info=True)
        return False


@thread_cleanup_router.delete("/threads/{thread_id}")
async def delete_thread_with_cleanup(
    thread_id: str, request: Request
) -> dict[str, Any]:
    """Delete a thread and purge all associated data.

    Goes beyond Aegra's default delete by also cleaning up:
    - Checkpoint history (conversation messages)
    - Feedback records
    - Token usage records
    """
    from uuid import UUID

    try:
        UUID(thread_id)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid thread_id format"
        ) from None

    user_id = await authenticated_user_id(request)

    if not settings.database_uri:
        raise HTTPException(status_code=503, detail="Database unavailable")

    import psycopg
    from psycopg.rows import dict_row

    async with await psycopg.AsyncConnection.connect(
        settings.database_uri, row_factory=dict_row
    ) as conn:
        cur = await conn.execute(
            "SELECT thread_id FROM thread WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        thread = await cur.fetchone()

    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")

    checkpoint_count = await _delete_checkpoints(thread_id)
    feedback_count = await _delete_feedback(thread_id)
    token_usage_deleted = await _delete_token_usage(thread_id)

    async with await psycopg.AsyncConnection.connect(settings.database_uri) as conn:
        await conn.execute(
            "DELETE FROM runs WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        await conn.execute(
            "DELETE FROM thread WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        await conn.commit()

    logger.info(
        "thread_deleted_with_cleanup",
        thread_id=thread_id,
        user_id=user_id[:8],
        checkpoints_deleted=checkpoint_count,
        feedback_deleted=feedback_count,
        token_usage_deleted=token_usage_deleted,
    )

    return {"status": "deleted"}
