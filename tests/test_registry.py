"""Tests for A2A Agent Registry (Phase 2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from template_agent.src.a2a.models import A2ATargetAgent
from template_agent.src.a2a.registry import A2AAgentRegistry


@pytest.fixture
def sample_card():
    return {
        "name": "Data Agent",
        "url": "http://data-agent:8082/a2a/",
        "version": "1.0.0",
        "skills": [
            {"id": "query-data", "name": "Query Data", "description": "Run queries"},
            {"id": "summarize", "name": "Summarize", "description": "Summarize data"},
        ],
        "capabilities": {"streaming": False},
    }


class TestA2ATargetAgent:
    def test_minimal(self):
        a = A2ATargetAgent(agent_id="x", base_url="http://x:8080")
        assert a.agent_id == "x"
        assert a.skills == []
        assert a.card is None

    def test_full(self):
        a = A2ATargetAgent(
            agent_id="y",
            base_url="http://y:8080",
            description="desc",
            card={"name": "Y"},
            skills=["s1", "s2"],
            capabilities={"streaming": True},
        )
        assert a.skills == ["s1", "s2"]


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    return resp


class TestA2AAgentRegistry:
    async def test_discover_success(self, sample_card):
        registry = A2AAgentRegistry(timeout=5.0)
        registry._client = AsyncMock(spec=httpx.AsyncClient)
        registry._client.get = AsyncMock(return_value=_mock_response(200, sample_card))

        await registry.discover(
            {"data-agent": {"base_url": "http://data-agent:8082", "description": "DA"}}
        )
        assert "data-agent" in registry.list_agent_ids()
        agent = registry.get("data-agent")
        assert agent is not None
        assert agent.skills == ["query-data", "summarize"]
        assert agent.description == "DA"

    async def test_discover_card_fetch_failure(self):
        registry = A2AAgentRegistry(timeout=5.0)
        registry._client = AsyncMock(spec=httpx.AsyncClient)
        registry._client.get = AsyncMock(return_value=_mock_response(404))

        await registry.discover({"bad": {"base_url": "http://bad:9090"}})
        agent = registry.get("bad")
        assert agent is not None
        assert agent.card is None
        assert agent.skills == []

    async def test_list_agents(self, sample_card):
        registry = A2AAgentRegistry(timeout=5.0)
        registry._client = AsyncMock(spec=httpx.AsyncClient)
        registry._client.get = AsyncMock(return_value=_mock_response(200, sample_card))

        await registry.discover(
            {
                "a": {"base_url": "http://a:8080"},
                "b": {"base_url": "http://b:8080"},
            }
        )
        assert len(registry.list_agents()) == 2

    async def test_get_nonexistent(self):
        registry = A2AAgentRegistry(timeout=5.0)
        assert registry.get("nope") is None

    async def test_close(self):
        registry = A2AAgentRegistry(timeout=5.0)
        registry._client = AsyncMock(spec=httpx.AsyncClient)
        await registry.close()
        registry._client.aclose.assert_awaited_once()

    async def test_fetch_card_tries_multiple_paths(self):
        """_fetch_card tries multiple well-known paths."""
        registry = A2AAgentRegistry(timeout=5.0)
        registry._client = AsyncMock(spec=httpx.AsyncClient)
        call_count = 0

        async def _mock_get(url):
            nonlocal call_count
            call_count += 1
            if "/.well-known/agent.json" in url:
                return _mock_response(200, {"name": "Found"})
            return _mock_response(404)

        registry._client.get = _mock_get
        card = await registry._fetch_card("http://test:8080")
        assert card is not None
        assert card["name"] == "Found"
        assert call_count == 3

    async def test_fetch_card_handles_exception(self):
        """_fetch_card handles exceptions during HTTP request."""
        registry = A2AAgentRegistry(timeout=5.0)
        registry._client = AsyncMock(spec=httpx.AsyncClient)
        registry._client.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        card = await registry._fetch_card("http://unreachable:8080")
        assert card is None


class TestRegistrySingleton:
    """Tests for registry singleton functions."""

    async def test_get_registry_creates_singleton(self, monkeypatch):
        """get_registry creates singleton on first call."""
        import template_agent.src.a2a.registry as reg_mod

        monkeypatch.setattr(reg_mod, "_registry", None)
        registry = reg_mod.get_registry()
        assert registry is not None
        assert reg_mod._registry is registry

    async def test_get_registry_returns_same_instance(self, monkeypatch):
        """get_registry returns the same instance on subsequent calls."""
        import template_agent.src.a2a.registry as reg_mod

        monkeypatch.setattr(reg_mod, "_registry", None)
        r1 = reg_mod.get_registry()
        r2 = reg_mod.get_registry()
        assert r1 is r2

    async def test_initialize_registry_creates_new_instance(self, monkeypatch):
        """initialize_registry creates a new registry instance."""
        import template_agent.src.a2a.registry as reg_mod

        monkeypatch.setattr(reg_mod, "_registry", None)
        monkeypatch.setattr(reg_mod.settings, "A2A_TARGET_AGENTS", None)
        await reg_mod.initialize_registry()
        assert reg_mod._registry is not None

    async def test_initialize_registry_with_targets(self, monkeypatch):
        """initialize_registry discovers configured targets."""
        import template_agent.src.a2a.registry as reg_mod

        monkeypatch.setattr(reg_mod, "_registry", None)
        monkeypatch.setattr(
            reg_mod.settings,
            "A2A_TARGET_AGENTS",
            {"test-agent": {"base_url": "http://test:8080"}},
        )

        with patch.object(
            A2AAgentRegistry, "discover", new_callable=AsyncMock
        ) as mock_discover:
            await reg_mod.initialize_registry()
            mock_discover.assert_awaited_once()

    async def test_cleanup_registry(self, monkeypatch):
        """cleanup_registry closes and clears the singleton."""
        import template_agent.src.a2a.registry as reg_mod

        mock_registry = AsyncMock(spec=A2AAgentRegistry)
        monkeypatch.setattr(reg_mod, "_registry", mock_registry)
        await reg_mod.cleanup_registry()
        mock_registry.close.assert_awaited_once()
        assert reg_mod._registry is None

    async def test_cleanup_registry_when_none(self, monkeypatch):
        """cleanup_registry handles None registry gracefully."""
        import template_agent.src.a2a.registry as reg_mod

        monkeypatch.setattr(reg_mod, "_registry", None)
        await reg_mod.cleanup_registry()
        assert reg_mod._registry is None
