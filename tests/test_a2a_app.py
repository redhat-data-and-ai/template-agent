"""Tests for a2a/app.py - A2A Starlette app builder and agent card."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from template_agent.src.a2a.app import (
    _get_downstream_skills,
    build_a2a_starlette_app,
    build_agent_card,
    resolve_a2a_public_base_url,
)
from template_agent.src.a2a.models import A2ATargetAgent
from template_agent.src.a2a.registry import A2AAgentRegistry
from template_agent.src.settings import Settings


def _settings(**overrides) -> Settings:
    defaults = {
        "USE_INMEMORY_SAVER": True,
        "A2A_ENABLED": True,
        "A2A_AUTH_REQUIRED": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestResolveA2APublicBaseUrl:
    """Tests for resolve_a2a_public_base_url function."""

    def test_uses_explicit_public_base_url(self):
        """Uses AGENT_PUBLIC_BASE_URL when set."""
        cfg = _settings(AGENT_PUBLIC_BASE_URL="https://my-agent.example.com")
        url = resolve_a2a_public_base_url(cfg)
        assert url == "https://my-agent.example.com/a2a/"

    def test_uses_agent_url_env_fallback(self, monkeypatch):
        """Falls back to AGENT_URL env var when AGENT_PUBLIC_BASE_URL not set."""
        monkeypatch.setenv("AGENT_URL", "https://env-agent.example.com")
        cfg = _settings(AGENT_PUBLIC_BASE_URL="")
        url = resolve_a2a_public_base_url(cfg)
        assert url == "https://env-agent.example.com/a2a/"

    def test_constructs_from_host_and_port(self, monkeypatch):
        """Constructs URL from AGENT_HOST and AGENT_PORT when no explicit URL."""
        monkeypatch.delenv("AGENT_URL", raising=False)
        cfg = _settings(
            AGENT_PUBLIC_BASE_URL="",
            AGENT_HOST="myhost",
            AGENT_PORT=9090,
        )
        url = resolve_a2a_public_base_url(cfg)
        assert url == "http://myhost:9090/a2a/"

    def test_replaces_0000_with_localhost(self, monkeypatch):
        """Replaces 0.0.0.0 host with 127.0.0.1."""
        monkeypatch.delenv("AGENT_URL", raising=False)
        cfg = _settings(
            AGENT_PUBLIC_BASE_URL="",
            AGENT_HOST="0.0.0.0",
            AGENT_PORT=8080,
        )
        url = resolve_a2a_public_base_url(cfg)
        assert "127.0.0.1" in url

    def test_replaces_ipv6_any_with_localhost(self, monkeypatch):
        """Replaces :: (IPv6 any) host with 127.0.0.1."""
        monkeypatch.delenv("AGENT_URL", raising=False)
        cfg = _settings(
            AGENT_PUBLIC_BASE_URL="",
            AGENT_HOST="::",
            AGENT_PORT=8080,
        )
        url = resolve_a2a_public_base_url(cfg)
        assert "127.0.0.1" in url

    def test_custom_path_prefix(self):
        """Uses custom A2A_PATH_PREFIX."""
        cfg = _settings(
            AGENT_PUBLIC_BASE_URL="https://my-agent.example.com",
            A2A_PATH_PREFIX="/custom-a2a",
        )
        url = resolve_a2a_public_base_url(cfg)
        assert url == "https://my-agent.example.com/custom-a2a/"

    def test_path_prefix_without_leading_slash(self):
        """Handles path prefix without leading slash."""
        cfg = _settings(
            AGENT_PUBLIC_BASE_URL="https://my-agent.example.com",
            A2A_PATH_PREFIX="custom-a2a",
        )
        url = resolve_a2a_public_base_url(cfg)
        assert url == "https://my-agent.example.com/custom-a2a/"

    def test_empty_path_prefix_defaults_to_a2a(self):
        """Empty path prefix defaults to /a2a."""
        cfg = _settings(
            AGENT_PUBLIC_BASE_URL="https://my-agent.example.com",
            A2A_PATH_PREFIX="",
        )
        url = resolve_a2a_public_base_url(cfg)
        assert url == "https://my-agent.example.com/a2a/"


class TestGetDownstreamSkills:
    """Tests for _get_downstream_skills function."""

    def test_returns_empty_when_no_agents(self, monkeypatch):
        """Returns empty list when no downstream agents."""
        import template_agent.src.a2a.registry as reg_mod

        registry = A2AAgentRegistry.__new__(A2AAgentRegistry)
        registry._agents = {}
        monkeypatch.setattr(reg_mod, "_registry", registry)

        skills = _get_downstream_skills()
        assert skills == []

    def test_collects_skills_from_agents(self, monkeypatch):
        """Collects skills from all registered agents."""
        import template_agent.src.a2a.registry as reg_mod

        registry = A2AAgentRegistry.__new__(A2AAgentRegistry)
        registry._agents = {
            "agent1": A2ATargetAgent(
                agent_id="agent1",
                base_url="http://agent1:8080",
                description="Agent 1",
                skills=["skill1", "skill2"],
            ),
            "agent2": A2ATargetAgent(
                agent_id="agent2",
                base_url="http://agent2:8080",
                description="Agent 2",
                skills=["skill3"],
            ),
        }
        monkeypatch.setattr(reg_mod, "_registry", registry)

        skills = _get_downstream_skills()
        assert len(skills) == 3
        skill_ids = [s.id for s in skills]
        assert "downstream:agent1:skill1" in skill_ids
        assert "downstream:agent1:skill2" in skill_ids
        assert "downstream:agent2:skill3" in skill_ids

    def test_handles_registry_exception(self, monkeypatch):
        """Returns empty list when registry raises exception."""
        import template_agent.src.a2a.registry as reg_mod

        monkeypatch.setattr(reg_mod, "_registry", None)

        def _raise():
            raise RuntimeError("Registry not initialized")

        monkeypatch.setattr(reg_mod, "get_registry", _raise)

        skills = _get_downstream_skills()
        assert skills == []


class TestBuildAgentCard:
    """Tests for build_agent_card function."""

    def test_basic_card_structure(self, monkeypatch):
        """Agent card has correct basic structure."""
        import template_agent.src.a2a.registry as reg_mod

        registry = A2AAgentRegistry.__new__(A2AAgentRegistry)
        registry._agents = {}
        monkeypatch.setattr(reg_mod, "_registry", registry)

        cfg = _settings()
        card = build_agent_card(cfg)

        assert card.name == "Template Agent"
        assert len(card.supported_interfaces) == 1
        assert card.capabilities.streaming is True

    def test_security_schemes_when_auth_required(self, monkeypatch):
        """Includes security schemes when auth is required."""
        import template_agent.src.a2a.registry as reg_mod

        registry = A2AAgentRegistry.__new__(A2AAgentRegistry)
        registry._agents = {}
        monkeypatch.setattr(reg_mod, "_registry", registry)

        cfg = _settings(A2A_AUTH_REQUIRED=True)
        card = build_agent_card(cfg)

        assert card.security_schemes is not None
        assert "bearer" in card.security_schemes
        assert card.security_requirements is not None

    def test_no_security_schemes_when_auth_not_required(self, monkeypatch):
        """No security schemes when auth is not required."""
        import template_agent.src.a2a.registry as reg_mod

        registry = A2AAgentRegistry.__new__(A2AAgentRegistry)
        registry._agents = {}
        monkeypatch.setattr(reg_mod, "_registry", registry)

        cfg = _settings(A2A_AUTH_REQUIRED=False)
        card = build_agent_card(cfg)

        assert not card.security_schemes or card.security_schemes == {}
        assert not card.security_requirements

    def test_provider_info_when_configured(self, monkeypatch):
        """Includes provider info when configured."""
        import template_agent.src.a2a.registry as reg_mod

        registry = A2AAgentRegistry.__new__(A2AAgentRegistry)
        registry._agents = {}
        monkeypatch.setattr(reg_mod, "_registry", registry)

        cfg = _settings(
            A2A_PROVIDER_NAME="My Org",
            A2A_PROVIDER_URL="https://myorg.com",
        )
        card = build_agent_card(cfg)

        assert card.provider is not None
        assert card.provider.organization == "My Org"
        assert card.provider.url == "https://myorg.com"

    def test_no_provider_when_not_configured(self, monkeypatch):
        """No provider when not configured."""
        import template_agent.src.a2a.registry as reg_mod

        registry = A2AAgentRegistry.__new__(A2AAgentRegistry)
        registry._agents = {}
        monkeypatch.setattr(reg_mod, "_registry", registry)

        cfg = _settings()
        card = build_agent_card(cfg)

        assert not card.provider.organization if card.provider else True

    def test_includes_downstream_skills(self, monkeypatch):
        """Includes skills from downstream agents."""
        import template_agent.src.a2a.registry as reg_mod

        registry = A2AAgentRegistry.__new__(A2AAgentRegistry)
        registry._agents = {
            "downstream-agent": A2ATargetAgent(
                agent_id="downstream-agent",
                base_url="http://downstream:8080",
                description="Downstream",
                skills=["echo"],
            ),
        }
        monkeypatch.setattr(reg_mod, "_registry", registry)

        cfg = _settings()
        card = build_agent_card(cfg)

        skill_ids = [s.id for s in card.skills]
        assert "template-agent-mcp" in skill_ids
        assert "downstream:downstream-agent:echo" in skill_ids


class TestBuildA2AStarletteApp:
    """Tests for build_a2a_starlette_app function."""

    def test_creates_starlette_app(self, monkeypatch):
        """Creates a valid Starlette app."""
        import template_agent.src.a2a.registry as reg_mod

        registry = A2AAgentRegistry.__new__(A2AAgentRegistry)
        registry._agents = {}
        monkeypatch.setattr(reg_mod, "_registry", registry)

        cfg = _settings()
        app = build_a2a_starlette_app(cfg)

        assert app is not None

    def test_agent_card_endpoint(self, monkeypatch):
        """Agent card endpoint is accessible."""
        import template_agent.src.a2a.registry as reg_mod

        registry = A2AAgentRegistry.__new__(A2AAgentRegistry)
        registry._agents = {}
        monkeypatch.setattr(reg_mod, "_registry", registry)

        cfg = _settings()
        app = build_a2a_starlette_app(cfg)
        client = TestClient(app)

        resp = client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        card = resp.json()
        assert card["name"] == "Template Agent"

    def test_uses_default_settings_when_none_provided(self, monkeypatch):
        """Uses default settings when cfg is None."""
        import template_agent.src.a2a.registry as reg_mod

        registry = A2AAgentRegistry.__new__(A2AAgentRegistry)
        registry._agents = {}
        monkeypatch.setattr(reg_mod, "_registry", registry)

        monkeypatch.setenv("USE_INMEMORY_SAVER", "true")
        monkeypatch.setenv("A2A_ENABLED", "true")

        app = build_a2a_starlette_app(None)
        assert app is not None
