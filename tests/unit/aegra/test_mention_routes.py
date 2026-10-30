"""Unit tests for the slash-menu mentions catalog."""

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from deep_agent.aegra.mention_routes import router as mention_router
from deep_agent.src.infrastructure.requested_subagents import mention_catalog


def test_mention_catalog_is_name_and_description_only():
    configs = {
        "publisher": {
            "description": "Sends   email\nreports.",
            "body": "You are a publisher. Do not leak this.",
            "tools": ["template_send_email"],
            "skills": ["email-formatter"],
        },
        "analyst": {"description": "Calculates BMI"},
    }
    catalog = mention_catalog(configs)
    assert catalog == {
        "subagents": [
            {"name": "analyst", "description": "Calculates BMI"},
            {"name": "publisher", "description": "Sends email reports."},
        ]
    }
    dumped = str(catalog)
    assert "Do not leak" not in dumped
    assert "template_send_email" not in dumped
    assert "email-formatter" not in dumped


def test_list_mentions_route():
    app = FastAPI()
    app.include_router(mention_router)
    client = TestClient(app)
    payload = {
        "subagents": [{"name": "analyst", "description": "Calculates BMI"}],
    }
    with (
        patch(
            "deep_agent.aegra.mention_routes.authenticated_user_id",
            new=AsyncMock(return_value="dev"),
        ) as auth,
        patch(
            "deep_agent.aegra.mention_routes.agent_config.get_all_subagent_configs",
            return_value={
                "analyst": {"description": "Calculates BMI", "body": "secret"}
            },
        ),
    ):
        response = client.get("/mentions")
    assert response.status_code == 200
    assert response.json() == payload
    assert auth.await_args is not None
    assert auth.await_args.kwargs.get("reject_anonymous") is True


def test_list_mentions_rejects_anonymous_when_auth_enabled():
    app = FastAPI()
    app.include_router(mention_router)
    client = TestClient(app)
    with (
        patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
        patch(
            "deep_agent.aegra.mention_routes.agent_config.get_all_subagent_configs",
        ) as configs,
    ):
        response = client.get("/mentions")
    assert response.status_code == 401
    configs.assert_not_called()
