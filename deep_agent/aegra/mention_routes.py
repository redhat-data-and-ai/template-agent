"""Catalog of subagents the chat box can offer after ``/``."""

from __future__ import annotations

from fastapi import APIRouter, Request

from deep_agent.aegra.auth_helpers import authenticated_user_id
from deep_agent.src.agent.config import agent_config
from deep_agent.src.infrastructure.requested_subagents import mention_catalog

router = APIRouter(tags=["mentions"])


@router.get("/mentions")
async def list_mentions(request: Request) -> dict[str, list[dict[str, str]]]:
    """Return subagent names and descriptions for the slash menu."""
    await authenticated_user_id(request, reject_anonymous=True)
    return mention_catalog(agent_config.get_all_subagent_configs())
