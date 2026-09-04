"""REST API for user projects.

Provides CRUD endpoints so the UI can organize conversations into named
groups, including assign/unassign and delete.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from psycopg.errors import UniqueViolation

from deep_agent.aegra.auth_helpers import authenticated_user_id
from deep_agent.src.projects import (
    ProjectNotFoundError,
    ProjectsRepository,
    ProjectsStorageError,
    ThreadNotFoundError,
)
from deep_agent.src.schema import (
    Project,
    ProjectAssignRequest,
    ProjectCreate,
    ProjectDeletionResponse,
    ProjectListResponse,
    ProjectUnassignAllResponse,
    ProjectUpdate,
)
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

projects_router = APIRouter(tags=["projects"])


def _get_repo() -> ProjectsRepository:
    """Return the projects repository, or 503 if the database URI is missing."""
    if not settings.database_uri:
        raise HTTPException(status_code=503, detail="Database not configured")
    return ProjectsRepository(settings.database_uri)


def _row_to_project(row: dict[str, Any], thread_count: int = 0) -> Project:
    raw_count = row.get("thread_count", thread_count)
    try:
        count = int(raw_count or 0)
    except (TypeError, ValueError):
        count = thread_count
    return Project(
        project_id=row["project_id"],
        project_name=row["project_name"],
        project_description=row.get("project_description"),
        username=row["username"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        thread_count=count,
    )


def _is_duplicate_name(exc: BaseException) -> bool:
    """Return True if *exc* (or its cause) is a unique-name violation."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, UniqueViolation):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


# ── CRUD ──────────────────────────────────────────────────────


@projects_router.get("/projects", response_model=ProjectListResponse)
async def list_projects(request: Request) -> ProjectListResponse:
    """Return all projects for the authenticated user."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    repo = _get_repo()
    rows = await repo.list_projects_with_thread_counts(user_id)
    return ProjectListResponse(projects=[_row_to_project(r) for r in rows])


@projects_router.post("/projects", response_model=Project, status_code=201)
async def create_project(body: ProjectCreate, request: Request) -> Project:
    """Create a project; 409 if the name is already used by this user."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    repo = _get_repo()
    try:
        row = await repo.create_project(
            user_id, body.project_name, body.project_description
        )
    except Exception as exc:
        if _is_duplicate_name(exc):
            raise HTTPException(
                status_code=409, detail="A project with that name already exists"
            ) from exc
        raise
    return _row_to_project(row)


@projects_router.post("/projects/assign")
async def assign_thread(body: ProjectAssignRequest, request: Request) -> dict[str, Any]:
    """Assign or unassign a thread to a project."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    repo = _get_repo()

    if body.project_id:
        if not await repo.verify_project_ownership(body.project_id, user_id):
            raise HTTPException(status_code=404, detail="Project not found")

    try:
        await repo.assign_thread_to_project(body.thread_id, body.project_id, user_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found") from None
    except ThreadNotFoundError:
        raise HTTPException(status_code=404, detail="Thread not found") from None
    return {
        "status": "success",
        "thread_id": body.thread_id,
        "project_id": body.project_id,
    }


@projects_router.patch("/projects/{project_id}", response_model=Project)
async def update_project(
    project_id: str, body: ProjectUpdate, request: Request
) -> Project:
    """Rename or update a project's description."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    if body.project_name is None and body.project_description is None:
        raise HTTPException(
            status_code=422,
            detail="At least one field (project_name or project_description) must be provided",
        )
    repo = _get_repo()
    try:
        row = await repo.update_project(
            project_id, user_id, body.project_name, body.project_description
        )
    except Exception as exc:
        if _is_duplicate_name(exc):
            raise HTTPException(
                status_code=409, detail="A project with that name already exists"
            ) from exc
        raise
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    tc = await repo.count_project_threads(project_id, user_id)
    return _row_to_project(row, thread_count=tc)


@projects_router.delete(
    "/projects/{project_id}", response_model=ProjectDeletionResponse
)
async def delete_project(
    project_id: str, request: Request, keep_threads: bool = False
) -> ProjectDeletionResponse:
    """Delete a project, optionally keeping its conversations.

    When *keep_threads* is true, conversations are unassigned (moved to Chats)
    before the project row is removed. Otherwise conversations are deleted.
    """
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    repo = _get_repo()

    if not await repo.verify_project_ownership(project_id, user_id):
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        deleted_thread_ids = await repo.delete_project_with_threads(
            project_id, user_id, keep_threads=keep_threads
        )
    except ProjectNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found") from None
    except ProjectsStorageError:
        logger.exception("project_delete_failed")
        raise HTTPException(
            status_code=500, detail="Failed to delete project"
        ) from None

    if keep_threads:
        message = "Project deleted. Conversations were moved to Chats."
    else:
        message = f"Project and {len(deleted_thread_ids)} conversation(s) permanently deleted."

    return ProjectDeletionResponse(
        status="success",
        message=message,
        project_id=project_id,
        deleted_thread_ids=deleted_thread_ids,
    )


@projects_router.get("/projects/{project_id}/threads")
async def get_project_threads(
    project_id: str, request: Request
) -> list[dict[str, Any]]:
    """Return threads belonging to a project."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    repo = _get_repo()

    if not await repo.verify_project_ownership(project_id, user_id):
        raise HTTPException(status_code=404, detail="Project not found")

    return await repo.get_project_threads(project_id, user_id)


@projects_router.post(
    "/projects/{project_id}/unassign-all", response_model=ProjectUnassignAllResponse
)
async def unassign_all_threads(
    project_id: str, request: Request
) -> ProjectUnassignAllResponse:
    """Unassign every conversation from a project, leaving the project in place."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    repo = _get_repo()

    if not await repo.verify_project_ownership(project_id, user_id):
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        thread_ids = await repo.unassign_all_threads(project_id, user_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found") from None
    return ProjectUnassignAllResponse(
        status="success",
        message=f"{len(thread_ids)} conversation(s) moved to Chats.",
        project_id=project_id,
        threads_unassigned=len(thread_ids),
        unassigned_thread_ids=thread_ids,
    )
