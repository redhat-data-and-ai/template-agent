"""Threads route for the template agent API.

This module provides endpoints for managing conversation threads,
including listing threads for specific users.
"""

from typing import List

import psycopg2
from fastapi import APIRouter, HTTPException

from template_agent.src.core.storage import get_user_threads
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

router = APIRouter()

app_logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


@router.get("/v1/threads/{user_id}")
async def list_threads(user_id: str) -> List[str]:
    """Get a list of all thread IDs for a specific user.

    Args:
        user_id: The unique identifier of the user whose threads to retrieve.

    Returns:
        A list of thread IDs (strings) associated with the user.

    Raises:
        HTTPException: If there's a database connection error or query failure.
    """
    if settings.USE_INMEMORY_SAVER:
        app_logger.info(
            f"Using in-memory storage - retrieving threads from registry for user_id: {user_id}"
        )
        try:
            thread_ids = get_user_threads(user_id)
            app_logger.info(
                f"Found {len(thread_ids)} threads in registry for user_id: {user_id}"
            )
            return thread_ids
        except Exception as e:
            app_logger.error(f"Error accessing thread registry for user {user_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve threads from registry: {str(e)}",
            )

    try:
        with psycopg2.connect(settings.database_uri) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT DISTINCT thread_id FROM checkpoints WHERE metadata->>'user_id' = %s",
                (user_id,),
            )
            rows = cur.fetchall()
            thread_ids = [row[0] for row in rows]

            app_logger.info(f"Found {len(thread_ids)} threads for user_id: {user_id}")
            return thread_ids

    except Exception as e:
        app_logger.error(
            f"Database error while fetching threads for user {user_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
