"""Health check and admin routes for the template agent API.

This module provides health check endpoints to monitor the status
and availability of the template agent service, and admin endpoints
for runtime configuration management.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health_check() -> JSONResponse:
    """Perform a health check on the template agent service.

    This endpoint is used to verify that the service is running and
    responding to requests. It returns a simple JSON response indicating
    the service status.

    Returns:
        A JSONResponse containing the service status and name.
    """
    return JSONResponse(content={"status": "healthy", "service": "Template Agent"})


@router.post("/reload")
async def reload_config() -> JSONResponse:
    """Reload agent configuration from disk without restarting the container.

    Resets the AgentConfig singleton so the next request picks up
    any changes to orchestrator, subagent, or skill files on the
    bind-mounted volume.

    Returns:
        A JSONResponse with reload status and loaded config counts.
    """
    from template_agent.src.agent.config.loader import agent_config

    agent_config._configs_loaded = False
    agent_config._ensure_loaded()

    return JSONResponse(content={
        "status": "reloaded",
        "subagents": len(agent_config.get_all_subagent_configs()),
        "skills": len(agent_config._available_skills),
    })
