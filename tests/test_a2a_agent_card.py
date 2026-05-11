"""Tests for a2a/agent_card.py -- Agent Card builder."""

from __future__ import annotations

from unittest.mock import patch

from template_agent.src.a2a.agent_card import build_agent_card


class TestBuildAgentCard:
    """Tests for build_agent_card()."""

    def test_card_has_correct_name_and_description(self):
        card = build_agent_card()
        assert card.name == "Template Agent"
        assert card.description

    def test_card_version_matches_settings(self):
        card = build_agent_card()
        assert card.version == "1.0.0"

    def test_card_advertises_three_interfaces(self):
        card = build_agent_card()
        assert len(card.supported_interfaces) == 3

        bindings = [
            (i.protocol_binding, i.protocol_version)
            for i in card.supported_interfaces
        ]
        assert ("JSONRPC", "1.0") in bindings
        assert ("JSONRPC", "0.3") in bindings
        assert ("HTTP+JSON", "1.0") in bindings

    def test_all_interface_urls_end_with_a2a(self):
        card = build_agent_card()
        for iface in card.supported_interfaces:
            assert iface.url.endswith("/a2a")

    def test_card_has_streaming_capability(self):
        card = build_agent_card()
        assert card.capabilities.streaming is True

    def test_card_has_correct_input_output_modes(self):
        card = build_agent_card()
        assert "text/plain" in list(card.default_input_modes)
        assert "text/plain" in list(card.default_output_modes)
        assert "application/json" in list(card.default_output_modes)

    def test_card_has_general_assistant_skill(self):
        card = build_agent_card()
        assert len(card.skills) == 1
        skill = card.skills[0]
        assert skill.id == "general-assistant"
        assert skill.name == "General Assistant"
        assert len(skill.tags) > 0
        assert len(skill.examples) > 0

    def test_card_has_bearer_security_scheme(self):
        card = build_agent_card()
        assert "bearer" in card.security_schemes
        scheme = card.security_schemes["bearer"]
        assert scheme.http_auth_security_scheme.scheme == "Bearer"
        assert scheme.http_auth_security_scheme.bearer_format == "JWT"

    def test_card_has_security_requirement(self):
        card = build_agent_card()
        assert len(card.security_requirements) == 1
        assert "bearer" in card.security_requirements[0].schemes

    def test_card_no_provider_when_not_configured(self):
        card = build_agent_card()
        assert not card.HasField("provider")

    def test_card_includes_provider_when_both_set(self):
        with patch(
            "template_agent.src.a2a.agent_card.settings"
        ) as mock_settings:
            mock_settings.a2a_base_url = "http://localhost:8081"
            mock_settings.A2A_AGENT_NAME = "Test Agent"
            mock_settings.A2A_AGENT_DESCRIPTION = "desc"
            mock_settings.A2A_AGENT_VERSION = "1.0.0"
            mock_settings.A2A_PROVIDER_ORG = "TestOrg"
            mock_settings.A2A_PROVIDER_URL = "https://testorg.example.com"

            card = build_agent_card()
            assert card.provider is not None
            assert card.provider.organization == "TestOrg"
            assert card.provider.url == "https://testorg.example.com"

    def test_card_base_url_uses_settings(self):
        with patch(
            "template_agent.src.a2a.agent_card.settings"
        ) as mock_settings:
            mock_settings.a2a_base_url = "https://my-agent.example.com:9090"
            mock_settings.A2A_AGENT_NAME = "Agent"
            mock_settings.A2A_AGENT_DESCRIPTION = "desc"
            mock_settings.A2A_AGENT_VERSION = "2.0.0"
            mock_settings.A2A_PROVIDER_ORG = ""
            mock_settings.A2A_PROVIDER_URL = ""

            card = build_agent_card()
            for iface in card.supported_interfaces:
                assert iface.url.startswith("https://my-agent.example.com:9090")
