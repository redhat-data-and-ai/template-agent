"""Tests for the read-only admin config endpoint."""

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from deep_agent.aegra.http_app import app

ADMIN_CLAIMS = {"sub": "admin-1", "preferred_username": "admin-1"}
USER_CLAIMS = {"sub": "user-1", "preferred_username": "user-1"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestAuthorization:
    def test_requires_a_bearer_token(self, client: TestClient):
        with patch("deep_agent.aegra.auth.ENABLE_AUTH", True):
            assert client.get("/admin/config").status_code == 401

    def test_rejects_an_invalid_token(self, client: TestClient):
        import jwt

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                side_effect=jwt.PyJWTError("bad"),
            ),
        ):
            response = client.get(
                "/admin/config", headers={"Authorization": "Bearer nope"}
            )
        assert response.status_code == 401

    def test_unprivileged_role_is_forbidden(self, client: TestClient):
        """Config introspection is privileged — it names every tool and skill."""
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("deep_agent.aegra.auth._decode_token", return_value=USER_CLAIMS),
            patch(
                "deep_agent.src.ldap.resolve_user_role",
                return_value="users",
            ),
        ):
            response = client.get(
                "/admin/config", headers={"Authorization": "Bearer t"}
            )
        assert response.status_code == 403


class TestConfigReport:
    @staticmethod
    def _get(client: TestClient):
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("deep_agent.aegra.auth._decode_token", return_value=ADMIN_CLAIMS),
            patch("deep_agent.src.ldap.resolve_user_role", return_value="admins"),
        ):
            return client.get("/admin/config", headers={"Authorization": "Bearer t"})

    def test_reports_resolved_approval_decisions(self, client: TestClient):
        """The point of the endpoint: the effect of the globs, not the globs."""
        body = self._get(client).json()
        tools = body["approval"]["tools"]

        assert tools["ls"]["decision"] == "allow"
        assert tools["execute"]["decision"] == "ask"
        assert tools["execute"]["risk"] == "high"

    def test_reports_skills_and_their_paths(self, client: TestClient):
        # Asserting a specific skill name made this a test of which
        # skills ship, which changes; the report's job is to map every
        # shipped skill to a real path.
        skills = self._get(client).json()["skills"]
        assert skills, "config report lists no skills"
        assert all(isinstance(path, str) and path for path in skills.values())

    def test_reports_whether_opa_fails_open(self, client: TestClient):
        """An engine that is enabled but failing open enforces nothing."""
        engine = self._get(client).json()["policy_engine"]
        assert "opa_enabled" in engine
        assert "fail_open" in engine

    def test_does_not_leak_mcp_urls_or_credentials(self, client: TestClient):
        response = self._get(client)
        for server in response.json()["mcp_servers"].values():
            assert "url" not in server
            assert "auth_env_var" not in server
        assert "localhost:5001" not in response.text
