"""Deep research plan management routes.

Provides endpoints for saving, retrieving, and managing deep research plans.
"""

from fastapi import APIRouter, HTTPException, Query

from template_agent.src.schema import DeepResearchPlanRequest, DeepResearchPlanResponse
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(prefix="/v1/deep-research", tags=["deep-research"])

_GENERIC_500 = "An internal error occurred. Please try again later."


async def _verify_thread_ownership(thread_id: str, user_id: str | None) -> None:
    """Raise 403 if user_id does not match the registered owner of thread_id."""
    if not user_id:
        raise HTTPException(status_code=403, detail="user_id is required")
    from template_agent.src.core.deep_research.plan_store import get_thread_owner

    owner = await get_thread_owner(thread_id)
    if owner is not None and owner != user_id:
        raise HTTPException(
            status_code=403, detail="Not authorized to access this thread"
        )


@router.post("/plan", response_model=DeepResearchPlanResponse)
async def save_plan(request: DeepResearchPlanRequest) -> DeepResearchPlanResponse:
    """Save or update a deep research plan for a thread."""
    try:
        from template_agent.src.core.deep_research.plan_store import (
            register_thread_owner,
            set_plan,
        )

        await _verify_thread_ownership(request.thread_id, request.user_id)
        await set_plan(request.thread_id, request.plan, user_id=request.user_id)

        if request.user_id:
            await register_thread_owner(request.thread_id, request.user_id)

        logger.info(
            "deep_research_plan_saved",
            thread_id=request.thread_id,
            plan_count=len(request.plan),
        )

        return DeepResearchPlanResponse(
            thread_id=request.thread_id,
            plan_count=len(request.plan),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save deep research plan: {e}")
        raise HTTPException(status_code=500, detail=_GENERIC_500)


@router.get("/plan/{thread_id}")
async def get_plan(
    thread_id: str,
    user_id: str | None = Query(
        default=None, description="User ID for ownership verification"
    ),
) -> dict:
    """Get the deep research plan for a thread."""
    try:
        from template_agent.src.core.deep_research.plan_store import (
            get_plan as get_plan_fn,
        )

        await _verify_thread_ownership(thread_id, user_id)

        plan = await get_plan_fn(thread_id)
        if plan is None:
            raise HTTPException(status_code=404, detail="No plan found for thread")

        return {"thread_id": thread_id, "plan": plan, "plan_count": len(plan)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get deep research plan: {e}")
        raise HTTPException(status_code=500, detail=_GENERIC_500)


@router.delete("/plan/{thread_id}")
async def delete_plan(
    thread_id: str,
    user_id: str | None = Query(
        default=None, description="User ID for ownership verification"
    ),
) -> dict:
    """Delete the deep research plan for a thread."""
    try:
        from template_agent.src.core.deep_research.plan_store import clear_plan

        await _verify_thread_ownership(thread_id, user_id)

        await clear_plan(thread_id)
        return {"status": "success", "thread_id": thread_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete deep research plan: {e}")
        raise HTTPException(status_code=500, detail=_GENERIC_500)
