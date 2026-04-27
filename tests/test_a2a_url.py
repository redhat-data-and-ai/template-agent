"""Tests for A2A public URL resolution."""

from template_agent.src.a2a.app import resolve_a2a_public_base_url
from template_agent.src.settings import Settings


class TestResolveA2APublicBaseUrl:
    def test_uses_agent_url_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_URL", "https://agent.example.com")
        monkeypatch.delenv("AGENT_PUBLIC_BASE_URL", raising=False)
        cfg = Settings()
        assert resolve_a2a_public_base_url(cfg) == "https://agent.example.com/a2a/"

    def test_prefers_agent_public_base_url(self, monkeypatch):
        monkeypatch.delenv("AGENT_URL", raising=False)
        monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "https://pub.example")
        monkeypatch.setenv("A2A_PATH_PREFIX", "/wx")
        cfg = Settings()
        assert resolve_a2a_public_base_url(cfg) == "https://pub.example/wx/"
