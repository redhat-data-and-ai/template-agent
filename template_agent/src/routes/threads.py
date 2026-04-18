"""Threads route for the template agent API.

This module provides endpoints for managing conversation threads,
including listing threads for specific users.
"""

from typing import List

from fastapi import APIRouter, HTTPException

from template_agent.src.core.checkpointer import get_checkpointer
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

router = APIRouter()

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


@router.get("/v1/users/{user_id}/threads")
async def list_threads(user_id: str) -> List[str]:
    """Get a list of all thread IDs for a specific user.

    Args:
        user_id: The unique identifier of the user whose threads to retrieve.

    Returns:
        A list of thread IDs (strings) sorted by creation time (newest first).

    Raises:
        HTTPException: If there's a database connection error or query failure.
    """
    logger.info(f"Retrieving threads for user_id={user_id}")

    try:
        async with get_checkpointer() as checkpointer:
            async with checkpointer.conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT thread_id
                    FROM checkpoints
                    WHERE metadata->>'user_id' = %s
                    GROUP BY thread_id
                    ORDER BY MIN((checkpoint->>'ts')::timestamp) DESC
                    """,
                    (user_id,),
                )
                rows = await cur.fetchall()
                thread_ids = [row["thread_id"] for row in rows]

                logger.info(
                    f"Retrieved {len(thread_ids)} thread(s) for user_id={user_id}"
                )
                return thread_ids

    except Exception as e:
        logger.error(
            f"Database error while fetching threads for user {user_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
