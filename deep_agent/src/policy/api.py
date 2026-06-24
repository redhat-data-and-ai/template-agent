"""API endpoints for managing user policy settings."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from deep_agent.src.policy.repository import PolicySettingsRepository
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(prefix="/api/v1/policy", tags=["policy"])


class UpdateSettingsRequest(BaseModel):
    """Request model for updating user policy settings."""

    settings: dict[str, Any]


class SettingsResponse(BaseModel):
    """Response model for policy settings."""

    user_id: str
    settings: dict[str, Any]
    updated_at: str | None = None


@router.get("/settings/{user_id}", response_model=SettingsResponse)
async def get_user_settings(user_id: str):
    """Get policy settings for a specific user.

    Returns user-specific settings if they exist, otherwise returns OPA defaults.

    Args:
        user_id: User identifier

    Returns:
        SettingsResponse with current settings
    """
    repo = PolicySettingsRepository(settings.database_uri)
    user_settings = await repo.get_user_settings(user_id)

    if user_settings:
        return SettingsResponse(
            user_id=user_id,
            settings=user_settings.values,
            updated_at=user_settings.updated_at.isoformat(),
        )

    # Return indication that user has no custom settings
    # Frontend should show OPA defaults from the policy file
    return SettingsResponse(
        user_id=user_id,
        settings={},  # Empty means using OPA defaults
        updated_at=None,
    )


@router.put("/settings/{user_id}", response_model=SettingsResponse)
async def update_user_settings(user_id: str, request: UpdateSettingsRequest):
    """Update policy settings for a specific user.

    Settings are saved to the database and the middleware cache is invalidated.

    Args:
        user_id: User identifier
        request: Settings to save

    Returns:
        Updated SettingsResponse
    """
    from pathlib import Path
    from deep_agent.src.policy.schema_parser import (
        extract_schema_from_template,
        validate_user_settings,
    )

    # Validate against schema
    template_dir = Path(__file__).parents[3] / "config" / "compliance" / "policy_templates"
    template_path = template_dir / "agent_authz.rego.tmpl"

    try:
        schema = extract_schema_from_template(template_path)
        is_valid, errors = validate_user_settings(request.settings, schema)

        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail={"message": "Validation failed", "errors": errors}
            )
    except ValueError as exc:
        logger.error(f"Schema extraction failed: {exc}")
        # Continue without validation (graceful degradation)

    repo = PolicySettingsRepository(settings.database_uri)

    # Save settings to database
    saved = await repo.save_user_settings(user_id, request.settings)

    # Invalidate middleware cache
    try:
        from deep_agent.src.infrastructure.rego_trajectory_middleware import (
            get_middleware_instance,
        )

        # Get the global middleware instance and invalidate cache for this user
        middleware = get_middleware_instance()
        if middleware:
            middleware.invalidate_cache(user_id)
            logger.info(f"Invalidated policy settings cache for user {user_id}")
        else:
            logger.warning("No middleware instance found, cache not invalidated")
    except Exception as exc:
        logger.warning(f"Could not invalidate cache: {exc}")

    return SettingsResponse(
        user_id=user_id,
        settings=saved.values,
        updated_at=saved.updated_at.isoformat(),
    )


@router.delete("/settings/{user_id}")
async def reset_user_settings(user_id: str):
    """Reset user to OPA default settings.

    Deletes custom settings from database, causing the user to fall back
    to the defaults defined in the Rego policy.

    Args:
        user_id: User identifier

    Returns:
        Status message
    """
    repo = PolicySettingsRepository(settings.database_uri)
    deleted = await repo.delete_user_settings(user_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No custom settings found for user {user_id}"
        )

    logger.info(f"Reset policy settings for user {user_id}")

    return {
        "status": "reset",
        "user_id": user_id,
        "message": "User will now use OPA default settings"
    }


@router.get("/settings")
async def list_all_settings():
    """List all users with custom policy settings.

    Returns:
        List of users and their settings
    """
    repo = PolicySettingsRepository(settings.database_uri)
    all_settings = await repo.list_all_settings()

    return {
        "count": len(all_settings),
        "users": [
            {
                "user_id": s.user_id,
                "settings": s.values,
                "updated_at": s.updated_at.isoformat(),
            }
            for s in all_settings
        ],
    }


@router.get("/defaults")
async def get_default_settings():
    """Get the default policy settings from the OPA policy.

    These are the settings that will be used when a user has no custom settings.

    Returns:
        Default settings as defined in agent_authz.rego
    """
    # These should match the defaults in agent_authz.rego
    return {
        "max_trajectory_length": 100,
        "enable_trajectory_limits": True,
    }


@router.get("/schema/{template_id}")
async def get_policy_schema(template_id: str):
    """Get JSON Schema for a policy template.

    This schema is used by the frontend to dynamically generate the settings form.

    Args:
        template_id: Template identifier (e.g., 'agent_authz')

    Returns:
        JSON Schema with UI hints for dynamic form rendering
    """
    from pathlib import Path
    from deep_agent.src.policy.schema_parser import (
        extract_schema_from_template,
        validate_schema,
    )

    template_dir = Path(__file__).parents[3] / "config" / "compliance" / "policy_templates"
    template_path = template_dir / f"{template_id}.rego.tmpl"

    if not template_path.exists():
        raise HTTPException(404, f"Template {template_id} not found")

    try:
        schema = extract_schema_from_template(template_path)
        validate_schema(schema)
        return schema
    except ValueError as exc:
        logger.error(f"Failed to extract schema from {template_id}: {exc}")
        raise HTTPException(500, f"Invalid schema: {exc}")
    except Exception as exc:
        logger.error(f"Unexpected error extracting schema: {exc}")
        raise HTTPException(500, f"Failed to load schema: {exc}")


@router.post("/validate/{template_id}")
async def validate_policy_settings(template_id: str, request: UpdateSettingsRequest):
    """Validate policy settings against the template schema.

    Args:
        template_id: Template identifier
        request: Settings to validate

    Returns:
        Validation result with errors if any
    """
    from pathlib import Path
    from deep_agent.src.policy.schema_parser import (
        extract_schema_from_template,
        validate_user_settings,
    )

    template_dir = Path(__file__).parents[3] / "config" / "compliance" / "policy_templates"
    template_path = template_dir / f"{template_id}.rego.tmpl"

    if not template_path.exists():
        raise HTTPException(404, f"Template {template_id} not found")

    try:
        schema = extract_schema_from_template(template_path)
        is_valid, errors = validate_user_settings(request.settings, schema)

        return {
            "valid": is_valid,
            "errors": errors,
        }
    except Exception as exc:
        logger.error(f"Validation error: {exc}")
        raise HTTPException(500, f"Validation failed: {exc}")
