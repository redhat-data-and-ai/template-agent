"""REST API for user memories and custom rules.

Provides CRUD endpoints so the UI can persist personalization data
immediately when the user adds/removes items in Settings, without
waiting for a chat message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from deep_agent.aegra.auth_helpers import authenticated_user_id
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

if TYPE_CHECKING:
    from deep_agent.src.personalization.repository import PersonalizationRepository

logger = get_python_logger()

personalization_router = APIRouter(tags=["personalization"])


def _get_repo() -> PersonalizationRepository:
    from deep_agent.src.personalization.repository import PersonalizationRepository

    if not settings.database_uri:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return PersonalizationRepository(settings.database_uri)


class CreateMemoryRequest(BaseModel):
    """Request body for creating a memory."""

    id: UUID | None = None
    content: str


class CreateRuleRequest(BaseModel):
    """Request body for creating a rule."""

    id: UUID | None = None
    content: str
    is_active: bool = True


# ── Memories ──────────────────────────────────────────────


@personalization_router.get("/personalization/memories")
async def list_memories(request: Request) -> dict[str, Any]:
    """Return all memories for the authenticated user."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    repo = _get_repo()
    memories = await repo.list_memories(user_id)
    return {
        "memories": [
            {
                "id": str(m.id),
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in memories
        ]
    }


@personalization_router.post("/personalization/memories", status_code=201)
async def create_memory(request: Request, body: CreateMemoryRequest) -> dict[str, Any]:
    """Create a new memory for the authenticated user."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    repo = _get_repo()
    mem = await repo.create_memory(user_id, body.content, memory_id=body.id)
    logger.info("memory_created", user_id=user_id[:8], memory_id=str(mem.id))
    return {
        "id": str(mem.id),
        "content": mem.content,
        "created_at": mem.created_at.isoformat(),
    }


@personalization_router.delete("/personalization/memories/{memory_id}")
async def delete_memory(memory_id: str, request: Request) -> dict[str, str]:
    """Delete a memory by ID for the authenticated user."""
    try:
        mid = UUID(memory_id)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid memory_id format"
        ) from None
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    repo = _get_repo()
    deleted = await repo.delete_memory(user_id, mid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    logger.info("memory_deleted", user_id=user_id[:8], memory_id=memory_id)
    return {"status": "deleted"}


# ── Rules ─────────────────────────────────────────────────


@personalization_router.get("/personalization/rules")
async def list_rules(request: Request) -> dict[str, Any]:
    """Return all rules for the authenticated user."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)
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


@personalization_router.post("/personalization/rules", status_code=201)
async def create_rule(request: Request, body: CreateRuleRequest) -> dict[str, Any]:
    """Create a new rule for the authenticated user."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    repo = _get_repo()
    try:
        rule = await repo.upsert_rule(
            user_id, body.content, rule_id=body.id, is_active=body.is_active
        )
    except PermissionError:
        raise HTTPException(
            status_code=409, detail="Rule belongs to another user"
        ) from None
    logger.info("rule_created", user_id=user_id[:8], rule_id=str(rule.id))
    return {
        "id": str(rule.id),
        "content": rule.content,
        "is_active": rule.is_active,
        "created_at": rule.created_at.isoformat(),
    }


@personalization_router.delete("/personalization/rules/{rule_id}")
async def delete_rule(rule_id: str, request: Request) -> dict[str, str]:
    """Delete a rule by ID for the authenticated user."""
    try:
        rid = UUID(rule_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid rule_id format") from None
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    repo = _get_repo()
    deleted = await repo.delete_rule(user_id, rid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Rule not found")
    logger.info("rule_deleted", user_id=user_id[:8], rule_id=rule_id)
    return {"status": "deleted"}
