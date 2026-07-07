"""REST API for user memories and rules (per-user, JWT-authenticated)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from deep_agent.src.personalization.repository import PersonalizationRepository
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

personalization_router = APIRouter(tags=["personalization"])


class MemoryCreateRequest(BaseModel):
    """Request body for creating a user memory."""

    content: str = Field(min_length=1, max_length=2000)


class RuleCreateRequest(BaseModel):
    """Request body for creating a user rule."""

    content: str = Field(min_length=1, max_length=2000)
    is_active: bool = True


async def _authenticated_user_id(request: Request) -> str:
    from deep_agent.aegra.auth import DEV_USER_ID, ENABLE_AUTH, _decode_token

    if not ENABLE_AUTH:
        return DEV_USER_ID

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )

    payload = _decode_token(auth_header[7:])
    return str(payload["sub"])


def _get_repo() -> PersonalizationRepository:
    if not settings.database_uri:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return PersonalizationRepository(settings.database_uri)


@personalization_router.get("/memories")
async def list_memories(request: Request) -> dict[str, Any]:
    """List all memories for the authenticated user."""
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    memories = await repo.list_memories(user_id)
    return {
        "memories": [
            {
                "id": str(m.id),
                "content": m.content,
                "score": m.score,
                "created_at": m.created_at.isoformat(),
            }
            for m in memories
        ]
    }


@personalization_router.post("/memories", status_code=201)
async def create_memory(body: MemoryCreateRequest, request: Request) -> dict[str, Any]:
    """Create a new memory for the authenticated user."""
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    mem = await repo.create_memory(user_id, body.content)
    from deep_agent.src.cache.personalization_cache import invalidate

    await invalidate(user_id)
    return {"id": str(mem.id), "content": mem.content}


@personalization_router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, request: Request) -> dict[str, Any]:
    """Delete a specific memory for the authenticated user."""
    try:
        mid = UUID(memory_id)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid memory_id format"
        ) from None
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    deleted = await repo.delete_memory(user_id, mid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True}


@personalization_router.delete("/memories")
async def delete_all_memories(request: Request) -> dict[str, Any]:
    """Delete all memories for the authenticated user."""
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    count = await repo.delete_all_memories(user_id)
    return {"deleted_count": count}


@personalization_router.get("/rules")
async def list_rules(request: Request) -> dict[str, Any]:
    """List all rules for the authenticated user."""
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    rules = await repo.list_rules(user_id, active_only=False)
    return {
        "rules": [
            {
                "id": str(r.id),
                "content": r.content,
                "is_active": r.is_active,
                "created_at": r.created_at.isoformat(),
            }
            for r in rules
        ]
    }


@personalization_router.post("/rules", status_code=201)
async def create_rule(body: RuleCreateRequest, request: Request) -> dict[str, Any]:
    """Create a new rule for the authenticated user."""
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    rule = await repo.upsert_rule(user_id, body.content, is_active=body.is_active)
    from deep_agent.src.cache.personalization_cache import invalidate

    await invalidate(user_id)
    return {"id": str(rule.id), "content": rule.content, "is_active": rule.is_active}


@personalization_router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str, request: Request) -> dict[str, Any]:
    """Delete a specific rule for the authenticated user."""
    try:
        rid = UUID(rule_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid rule_id format") from None
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    deleted = await repo.delete_rule(user_id, rid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"deleted": True}


@personalization_router.delete("/rules")
async def delete_all_rules(request: Request) -> dict[str, Any]:
    """Delete all rules for the authenticated user."""
    user_id = await _authenticated_user_id(request)
    repo = _get_repo()
    count = await repo.delete_all_rules(user_id)
    return {"deleted_count": count}
