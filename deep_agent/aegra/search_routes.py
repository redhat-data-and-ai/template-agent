"""Search a user's own conversations.

Scoped to the caller at the SQL level, not by filtering afterwards: a
search endpoint that can be made to return someone else's conversation
text is a worse leak than a list endpoint, because the attacker chooses
the query.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from deep_agent.aegra.auth_helpers import authenticated_user_id
from deep_agent.src.search.index import refresh_user, search
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/threads")
async def search_threads(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    refresh: bool = Query(
        True,
        description="Reindex this user's changed threads first. Disable for paging.",
    ),
) -> dict[str, Any]:
    """Full-text search across the caller's threads."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)

    if refresh:
        try:
            # Best-effort: a stale index returns slightly old results,
            # which is a better failure than no search at all.
            await refresh_user(user_id)
        except Exception:
            logger.exception("Search index refresh failed for %s", user_id)

    try:
        results = await search(user_id, q, limit=limit, offset=offset)
    except Exception:
        logger.exception("Search failed for %s", user_id)
        raise HTTPException(status_code=500, detail="Search failed") from None

    return {"query": q, "count": len(results), "results": results}


@router.post("/reindex")
async def reindex(request: Request) -> dict[str, Any]:
    """Force a reindex of the caller's threads.

    The read path does this automatically; this exists for the case
    where someone needs to confirm the index is not the problem.
    """
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    return {"indexed": await refresh_user(user_id)}
