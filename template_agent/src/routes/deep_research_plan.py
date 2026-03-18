"""Deep research plan management routes.

Provides endpoints for saving, retrieving, and managing deep research plans.
"""

from fastapi import APIRouter, HTTPException

from template_agent.src.schema import DeepResearchPlanRequest, DeepResearchPlanResponse
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(prefix="/v1/deep-research", tags=["deep-research"])


@router.post("/plan", response_model=DeepResearchPlanResponse)
async def save_plan(request: DeepResearchPlanRequest) -> DeepResearchPlanResponse:
    """Save or update a deep research plan for a thread."""
    try:
        from template_agent.src.core.deep_research.plan_store import set_plan

        await set_plan(request.thread_id, request.plan)

        logger.info(
            "deep_research_plan_saved",
            thread_id=request.thread_id,
            plan_count=len(request.plan),
        )

        return DeepResearchPlanResponse(
            thread_id=request.thread_id,
            plan_count=len(request.plan),
        )
    except Exception as e:
        logger.error(f"Failed to save deep research plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plan/{thread_id}")
async def get_plan(thread_id: str) -> dict:
    """Get the deep research plan for a thread."""
    try:
        from template_agent.src.core.deep_research.plan_store import (
            get_plan as get_plan_fn,
        )

        plan = await get_plan_fn(thread_id)
        if plan is None:
            raise HTTPException(status_code=404, detail="No plan found for thread")

        return {"thread_id": thread_id, "plan": plan, "plan_count": len(plan)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get deep research plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/plan/{thread_id}")
async def delete_plan(thread_id: str) -> dict:
    """Delete the deep research plan for a thread."""
    try:
        from template_agent.src.core.deep_research.plan_store import clear_plan

        await clear_plan(thread_id)
        return {"status": "success", "thread_id": thread_id}
    except Exception as e:
        logger.error(f"Failed to delete deep research plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))
