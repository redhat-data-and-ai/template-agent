"""Unit tests for project management HTTP endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from psycopg.errors import UniqueViolation
from starlette.testclient import TestClient

from deep_agent.aegra.projects_routes import projects_router
from deep_agent.src.projects import (
    ProjectNotFoundError,
    ProjectsStorageError,
    ThreadNotFoundError,
)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(projects_router)
    return TestClient(app)


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.list_projects_with_thread_counts = AsyncMock(return_value=[])
    repo.create_project = AsyncMock(
        return_value={
            "project_id": "p1",
            "project_name": "Alpha",
            "project_description": None,
            "username": "dev-user",
            "created_at": "2026-01-01",
            "updated_at": "2026-01-01",
            "thread_count": 0,
        }
    )
    repo.update_project = AsyncMock(return_value=None)
    repo.verify_project_ownership = AsyncMock(return_value=True)
    repo.delete_project_with_threads = AsyncMock(return_value=["t1"])
    repo.count_project_threads = AsyncMock(return_value=1)
    repo.assign_thread_to_project = AsyncMock()
    repo.unassign_all_threads = AsyncMock(return_value=["t1", "t2"])
    repo.get_project_threads = AsyncMock(return_value=[])
    return repo


def _auth_and_repo(mock_repo):
    return (
        patch(
            "deep_agent.aegra.projects_routes.authenticated_user_id",
            new_callable=AsyncMock,
            return_value="dev-user",
        ),
        patch("deep_agent.aegra.projects_routes._get_repo", return_value=mock_repo),
    )


class TestListAndCreate:
    def test_list_projects(self, client, mock_repo):
        mock_repo.list_projects_with_thread_counts = AsyncMock(
            return_value=[
                {
                    "project_id": "p1",
                    "project_name": "Alpha",
                    "project_description": None,
                    "username": "dev-user",
                    "created_at": "2026-01-01",
                    "updated_at": "2026-01-01",
                    "thread_count": 3,
                }
            ]
        )
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.get("/projects")
        assert res.status_code == 200
        assert res.json()["projects"][0]["project_name"] == "Alpha"
        assert res.json()["projects"][0]["thread_count"] == 3

    def test_create_project(self, client, mock_repo):
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.post("/projects", json={"project_name": "Alpha"})
        assert res.status_code == 201
        assert res.json()["project_id"] == "p1"

    def test_create_duplicate_name_returns_409(self, client, mock_repo):
        mock_repo.create_project = AsyncMock(side_effect=UniqueViolation("duplicate"))
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.post("/projects", json={"project_name": "Alpha"})
        assert res.status_code == 409

    def test_create_duplicate_name_wrapped_returns_409(self, client, mock_repo):
        inner = UniqueViolation("duplicate")
        outer = RuntimeError("insert failed")
        outer.__cause__ = inner
        mock_repo.create_project = AsyncMock(side_effect=outer)
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.post("/projects", json={"project_name": "Alpha"})
        assert res.status_code == 409


class TestUpdateDeleteAssign:
    def test_update_not_found(self, client, mock_repo):
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.patch("/projects/p1", json={"project_name": "Beta"})
        assert res.status_code == 404

    def test_update_success(self, client, mock_repo):
        mock_repo.update_project = AsyncMock(
            return_value={
                "project_id": "p1",
                "project_name": "Beta",
                "project_description": "d",
                "username": "dev-user",
                "created_at": "t",
                "updated_at": "t",
            }
        )
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.patch("/projects/p1", json={"project_name": "Beta"})
        assert res.status_code == 200
        assert res.json()["project_name"] == "Beta"
        assert res.json()["thread_count"] == 1

    def test_delete_not_owned(self, client, mock_repo):
        mock_repo.verify_project_ownership = AsyncMock(return_value=False)
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.delete("/projects/p1")
        assert res.status_code == 404

    def test_delete_success(self, client, mock_repo):
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.delete("/projects/p1")
        assert res.status_code == 200
        body = res.json()
        assert body["project_id"] == "p1"
        assert body["deleted_thread_ids"] == ["t1"]

    def test_assign_unknown_project(self, client, mock_repo):
        mock_repo.verify_project_ownership = AsyncMock(return_value=False)
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.post(
                "/projects/assign",
                json={"thread_id": "th1", "project_id": "missing"},
            )
        assert res.status_code == 404

    def test_assign_success(self, client, mock_repo):
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.post(
                "/projects/assign",
                json={"thread_id": "th1", "project_id": "p1"},
            )
        assert res.status_code == 200
        assert res.json()["status"] == "success"
        mock_repo.assign_thread_to_project.assert_awaited_once_with(
            "th1", "p1", "dev-user"
        )

    def test_update_empty_body_returns_422(self, client, mock_repo):
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.patch("/projects/p1", json={})
        assert res.status_code == 422
        mock_repo.update_project.assert_not_called()

    def test_assign_unknown_thread_returns_404(self, client, mock_repo):
        mock_repo.assign_thread_to_project = AsyncMock(
            side_effect=ThreadNotFoundError("missing")
        )
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.post(
                "/projects/assign",
                json={"thread_id": "th1", "project_id": "p1"},
            )
        assert res.status_code == 404
        assert res.json()["detail"] == "Thread not found"

    def test_assign_missing_project_from_repo_returns_404(self, client, mock_repo):
        mock_repo.assign_thread_to_project = AsyncMock(
            side_effect=ProjectNotFoundError("gone")
        )
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.post(
                "/projects/assign",
                json={"thread_id": "th1", "project_id": "p1"},
            )
        assert res.status_code == 404
        assert res.json()["detail"] == "Project not found"

    def test_delete_race_returns_404(self, client, mock_repo):
        mock_repo.delete_project_with_threads = AsyncMock(
            side_effect=ProjectNotFoundError("gone")
        )
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.delete("/projects/p1")
        assert res.status_code == 404

    def test_unassign_all_success(self, client, mock_repo):
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.post("/projects/p1/unassign-all")
        assert res.status_code == 200
        body = res.json()
        assert body["project_id"] == "p1"
        assert body["threads_unassigned"] == 2
        assert body["unassigned_thread_ids"] == ["t1", "t2"]
        mock_repo.unassign_all_threads.assert_awaited_once_with("p1", "dev-user")

    def test_unassign_all_not_owned(self, client, mock_repo):
        mock_repo.verify_project_ownership = AsyncMock(return_value=False)
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.post("/projects/p1/unassign-all")
        assert res.status_code == 404
        mock_repo.unassign_all_threads.assert_not_called()

    def test_delete_keep_threads_unassigns_then_deletes_project(
        self, client, mock_repo
    ):
        mock_repo.delete_project_with_threads = AsyncMock(return_value=[])
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.delete("/projects/p1?keep_threads=true")
        assert res.status_code == 200
        body = res.json()
        assert body["project_id"] == "p1"
        assert body["deleted_thread_ids"] == []
        mock_repo.unassign_all_threads.assert_not_called()
        mock_repo.delete_project_with_threads.assert_awaited_once_with(
            "p1", "dev-user", keep_threads=True
        )

    def test_delete_storage_error_returns_generic_500(self, client, mock_repo):
        mock_repo.delete_project_with_threads = AsyncMock(
            side_effect=ProjectsStorageError(
                "duplicate key value violates unique constraint"
            )
        )
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.delete("/projects/p1")
        assert res.status_code == 500
        assert res.json()["detail"] == "Failed to delete project"

    def test_list_project_threads(self, client, mock_repo):
        mock_repo.get_project_threads = AsyncMock(
            return_value=[{"thread_id": "th1", "thread_title": "Hi"}]
        )
        auth, repo = _auth_and_repo(mock_repo)
        with auth, repo:
            res = client.get("/projects/p1/threads")
        assert res.status_code == 200
        assert res.json()[0]["thread_id"] == "th1"
