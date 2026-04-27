"""Tests for AgentCard enterprise compliance (Phase 1)."""

from __future__ import annotations

from template_agent.src.a2a.app import build_agent_card
from template_agent.src.settings import Settings


def _settings(**overrides) -> Settings:
    defaults = {
        "USE_INMEMORY_SAVER": True,
        "A2A_ENABLED": True,
        "A2A_AUTH_REQUIRED": True,
        "A2A_AGENT_VERSION": "2.0.0",
        "A2A_PROVIDER_NAME": "Acme Corp",
        "A2A_PROVIDER_URL": "https://acme.example.com",
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestAgentCardSecurity:
    def test_security_schemes_present_when_auth_required(self):
        card = build_agent_card(_settings(A2A_AUTH_REQUIRED=True))
        assert card.security_schemes is not None
        assert "bearer" in card.security_schemes
        scheme = card.security_schemes["bearer"].http_auth_security_scheme
        assert scheme.scheme == "bearer"

    def test_security_requirements_present(self):
        card = build_agent_card(_settings(A2A_AUTH_REQUIRED=True))
        assert card.security_requirements is not None
        assert len(card.security_requirements) == 1
        req = card.security_requirements[0]
        assert "bearer" in req.schemes

    def test_no_security_when_auth_not_required(self):
        card = build_agent_card(_settings(A2A_AUTH_REQUIRED=False))
        assert not card.security_schemes
        assert not card.security_requirements

    def test_capabilities_declared(self):
        card = build_agent_card(_settings())
        assert card.capabilities is not None
        assert card.capabilities.streaming is True
        assert card.capabilities.push_notifications is False

    def test_provider_info(self):
        card = build_agent_card(_settings())
        assert card.provider is not None
        assert card.provider.organization == "Acme Corp"
        assert card.provider.url == "https://acme.example.com"

    def test_version_configurable(self):
        card = build_agent_card(_settings(A2A_AGENT_VERSION="3.1.0"))
        assert card.version == "3.1.0"

    def test_primary_skill_present(self):
        card = build_agent_card(_settings())
        ids = [s.id for s in card.skills]
        assert "template-agent-mcp" in ids

    def test_default_modes(self):
        card = build_agent_card(_settings())
        assert "text" in card.default_input_modes
        assert "text" in card.default_output_modes

    def test_supported_interfaces(self):
        card = build_agent_card(_settings())
        assert len(card.supported_interfaces) == 1
        iface = card.supported_interfaces[0]
        assert iface.protocol_binding == "JSONRPC"
        assert iface.url.endswith("/")
