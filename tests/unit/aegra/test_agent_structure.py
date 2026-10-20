"""Tests for the developer-facing agent-structure endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from deep_agent.aegra import http_app


def test_agent_structure_returns_prompt_and_skill_details_without_auth_internals(
    monkeypatch,
) -> None:
    """Developer settings may view instructions, but never MCP auth configuration."""
    orchestrator = {
        "name": "orchestrator",
        "skills": ["client-intake"],
        "body": "This system prompt is visible in developer settings.",
        "skill_paths": {"client-intake": "/app/config/agent/skills/client-intake"},
    }
    subagents = {
        "analyst": {
            "name": "analyst",
            "skills": ["bmi-report"],
            "body": "This subagent prompt is visible in developer settings.",
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
        lambda: {
            "client-intake": "# Client intake\n\nGather requirements.",
            "bmi-report": "# BMI report\n\nCalculate BMI.",
        },
    )

    response = TestClient(http_app.app).get("/config/agent-structure")

    assert response.status_code == 200
    assert response.json() == {
        "name": "orchestrator",
        "orchestrator": {
            "name": "orchestrator",
            "skills": ["client-intake"],
            "body": "This system prompt is visible in developer settings.",
        },
        "subagents": {
            "analyst": {
                "name": "analyst",
                "skills": ["bmi-report"],
                "body": "This subagent prompt is visible in developer settings.",
            },
        },
        "mcp_servers": {
            "search": {
                "display_name": "Search",
                "transport": "streamable_http",
            },
        },
        "skills": {
            "client-intake": "# Client intake\n\nGather requirements.",
            "bmi-report": "# BMI report\n\nCalculate BMI.",
        },
    }

    # The endpoint works from copies and must not mutate the cached config.
    assert "body" in orchestrator
    assert "skill_paths" in subagents["analyst"]
    assert "oauth" in mcp_servers["search"]
