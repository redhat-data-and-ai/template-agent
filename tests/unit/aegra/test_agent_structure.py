"""Tests for the sanitized agent-structure endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from deep_agent.aegra import http_app


def test_agent_structure_returns_sanitized_configuration(monkeypatch) -> None:
    """The endpoint exposes graph metadata without prompt or auth internals."""
    orchestrator = {
        "name": "orchestrator",
        "skills": ["client-intake"],
        "body": "This system prompt must not be exposed.",
        "skill_paths": {"client-intake": "/app/config/agent/skills/client-intake"},
    }
    subagents = {
        "analyst": {
            "name": "analyst",
            "skills": ["bmi-report"],
            "body": "This subagent prompt must not be exposed.",
            "skill_paths": {"bmi-report": "/app/config/agent/skills/bmi-report"},
        }
    }
    mcp_servers = {
        "search": {
            "display_name": "Search",
            "transport": "streamable_http",
            "ssl_verify": True,
            "auth_env_var": "SEARCH_API_KEY",
            "oauth": {"client_secret_env": "SEARCH_CLIENT_SECRET"},
        }
    }

    monkeypatch.setattr(http_app.agent_config, "get_name", lambda: "orchestrator")
    monkeypatch.setattr(
        http_app.agent_config,
        "get_orchestrator_config",
        lambda: orchestrator,
    )
    monkeypatch.setattr(
        http_app.agent_config,
        "get_all_subagent_configs",
        lambda: subagents,
    )
    monkeypatch.setattr(http_app.agent_config, "get_mcp_servers", lambda: mcp_servers)
    monkeypatch.setattr(
        http_app.agent_config,
        "get_available_skills",
        lambda: ["client-intake", "bmi-report"],
    )

    response = TestClient(http_app.app).get("/config/agent-structure")

    assert response.status_code == 200
    assert response.json() == {
        "name": "orchestrator",
        "orchestrator": {"name": "orchestrator", "skills": ["client-intake"]},
        "subagents": {
            "analyst": {"name": "analyst", "skills": ["bmi-report"]},
        },
        "mcp_servers": {
            "search": {
                "display_name": "Search",
                "transport": "streamable_http",
            },
        },
        "available_skills": ["client-intake", "bmi-report"],
    }

    # The endpoint works from copies and must not mutate the cached config.
    assert "body" in orchestrator
    assert "skill_paths" in subagents["analyst"]
    assert "oauth" in mcp_servers["search"]
