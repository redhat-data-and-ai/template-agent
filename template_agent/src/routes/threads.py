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

    This endpoint queries the PostgreSQL database to retrieve all unique
    thread IDs associated with a given user_id from the checkpoints table.
    When using in-memory storage, returns an empty list since threads
    are not persisted.

    Args:
        user_id: The unique identifier of the user whose threads to retrieve.

    Returns:
        A list of thread IDs (strings) associated with the user.
        Returns empty list when using in-memory storage.

    Raises:
        HTTPException: If there's a database connection error or query failure.
            Status code 500 with error details.

    Note:
        This function uses raw SQL queries to extract thread_id from the
        checkpoints table where metadata contains the specified user_id.
        In-memory storage mode returns empty list as threads are not persisted.
    """
    # When using in-memory storage, get threads from thread registry
    if settings.USE_INMEMORY_SAVER:
        app_logger.info(
            f"Using in-memory storage - retrieving threads from registry for user_id: {user_id}"
        )
        try:
            # Use the thread registry for fast lookup
            thread_ids = get_user_threads(user_id)
            app_logger.info(
                f"Found {len(thread_ids)} threads in registry for user_id: {user_id}: {thread_ids}"
            )
            return thread_ids
        except Exception as e:
            app_logger.error(f"Error accessing thread registry for user {user_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve threads from registry: {str(e)}",
            )

    try:
        # Connect to the PostgreSQL database
        with psycopg2.connect(settings.database_uri) as conn:
            cur = conn.cursor()

            # Query for distinct thread IDs where metadata contains the user_id
            cur.execute(
                f"SELECT distinct thread_id FROM checkpoints where metadata->>'user_id'='{user_id}'"
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
