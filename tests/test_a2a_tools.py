"""Tests for a2a/tools.py -- dynamic downstream agent tool building."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.a2a.tools import _make_tool_for_agent, build_a2a_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_agent_card(
    name: str = "HelperAgent",
    description: str = "A helpful agent",
    skills: list[dict] | None = None,
) -> MagicMock:
    """Build a minimal AgentCard mock."""
    card = MagicMock()
    card.name = name
    card.description = description

    if skills is None:
        skills = [
            {
                "id": "skill-1",
                "description": "Does something useful",
                "examples": ["example1", "example2"],
            }
        ]

    mock_skills = []
    for s in skills:
        skill = MagicMock()
        skill.id = s["id"]
        skill.description = s["description"]
        skill.examples = s.get("examples", [])
        mock_skills.append(skill)

    card.skills = mock_skills
    return card


# ---------------------------------------------------------------------------
# _make_tool_for_agent
# ---------------------------------------------------------------------------
class TestMakeToolForAgent:
    def test_creates_one_tool_per_agent(self):
        card = _make_agent_card(
            skills=[
                {"id": "s1", "description": "Skill 1", "examples": []},
                {"id": "s2", "description": "Skill 2", "examples": ["ex"]},
            ]
        )
        tool = _make_tool_for_agent(card, "http://url", "token")
        assert isinstance(tool, MagicMock) is False
        assert hasattr(tool, "name")

    def test_tool_name_format(self):
        card = _make_agent_card(name="My Agent")
        tool = _make_tool_for_agent(card, "http://url", "token")
        assert tool.name == "a2a_my_agent"
        assert all(c.isalnum() or c == "_" for c in tool.name)

    def test_tool_name_truncated_to_64_chars(self):
        long_name = "A" * 100
        card = _make_agent_card(name=long_name)
        tool = _make_tool_for_agent(card, "http://url", "token")
        assert len(tool.name) <= 64

    def test_tool_name_sanitized(self):
        card = _make_agent_card(name="Agent With Spaces & Symbols!")
        tool = _make_tool_for_agent(card, "http://url", "token")
        assert all(c.isalnum() or c == "_" for c in tool.name)

    def test_tool_description_contains_agent_name(self):
        card = _make_agent_card(name="HelperBot")
        tool = _make_tool_for_agent(card, "http://url", "token")
        assert "HelperBot" in tool.description

    def test_tool_description_includes_examples(self):
        card = _make_agent_card(
            skills=[{"id": "s", "description": "d", "examples": ["ex1", "ex2"]}]
        )
        tool = _make_tool_for_agent(card, "http://url", "token")
        assert "ex1" in tool.description
        assert "ex2" in tool.description

    def test_tool_description_aggregates_multiple_skills(self):
        card = _make_agent_card(
            skills=[
                {"id": "s1", "description": "Skill 1", "examples": []},
                {"id": "s2", "description": "Skill 2", "examples": ["ex"]},
            ]
        )
        tool = _make_tool_for_agent(card, "http://url", "token")
        assert "Skill 1" in tool.description
        assert "Skill 2" in tool.description

    def test_tool_is_async_callable(self):
        card = _make_agent_card()
        tool = _make_tool_for_agent(card, "http://url", "token")
        assert tool.coroutine is not None

    def test_no_skills_still_returns_tool(self):
        card = _make_agent_card(skills=[])
        tool = _make_tool_for_agent(card, "http://url", "token")
        assert "general-purpose" in tool.description


# ---------------------------------------------------------------------------
# build_a2a_tools
# ---------------------------------------------------------------------------
class TestBuildA2aTools:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_downstream_urls(self):
        with patch(
            "template_agent.src.a2a.tools.settings"
        ) as mock_settings:
            mock_settings.a2a_downstream_urls = []
            result = await build_a2a_tools(access_token="tok")
            assert result == []

    @pytest.mark.asyncio
    async def test_discovers_agents_and_builds_tools(self):
        card = _make_agent_card(
            name="Downstream",
            skills=[{"id": "s1", "description": "desc", "examples": []}],
        )

        with patch(
            "template_agent.src.a2a.tools.settings"
        ) as mock_settings, patch(
            "template_agent.src.a2a.tools.A2ACardResolver"
        ) as MockResolver, patch(
            "template_agent.src.a2a.tools.httpx.AsyncClient"
        ) as MockHttpx:
            mock_settings.a2a_downstream_urls = ["http://agent1:8080"]

            mock_resolver_instance = MagicMock()
            mock_resolver_instance.get_agent_card = AsyncMock(return_value=card)
            MockResolver.return_value = mock_resolver_instance

            mock_http = AsyncMock()
            MockHttpx.return_value = mock_http
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)

            tools = await build_a2a_tools(access_token="tok")

        assert len(tools) == 1
        assert "Downstream" in tools[0].description

    @pytest.mark.asyncio
    async def test_one_tool_per_agent_not_per_skill(self):
        card = _make_agent_card(
            name="MultiSkill",
            skills=[
                {"id": "s1", "description": "Skill 1", "examples": []},
                {"id": "s2", "description": "Skill 2", "examples": []},
                {"id": "s3", "description": "Skill 3", "examples": []},
            ],
        )

        with patch(
            "template_agent.src.a2a.tools.settings"
        ) as mock_settings, patch(
            "template_agent.src.a2a.tools.A2ACardResolver"
        ) as MockResolver, patch(
            "template_agent.src.a2a.tools.httpx.AsyncClient"
        ) as MockHttpx:
            mock_settings.a2a_downstream_urls = ["http://agent:8080"]

            mock_resolver_instance = MagicMock()
            mock_resolver_instance.get_agent_card = AsyncMock(return_value=card)
            MockResolver.return_value = mock_resolver_instance

            mock_http = AsyncMock()
            MockHttpx.return_value = mock_http
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)

            tools = await build_a2a_tools(access_token="tok")

        assert len(tools) == 1
        assert tools[0].name == "a2a_multiskill"

    @pytest.mark.asyncio
    async def test_skips_failed_discovery_gracefully(self):
        with patch(
            "template_agent.src.a2a.tools.settings"
        ) as mock_settings, patch(
            "template_agent.src.a2a.tools.A2ACardResolver"
        ) as MockResolver, patch(
            "template_agent.src.a2a.tools.httpx.AsyncClient"
        ) as MockHttpx:
            mock_settings.a2a_downstream_urls = [
                "http://failing:8080",
                "http://ok:8080",
            ]

            good_card = _make_agent_card(
                name="OK",
                skills=[{"id": "s", "description": "d", "examples": []}],
            )

            call_count = 0

            async def _resolver_side_effect():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("unreachable")
                return good_card

            mock_resolver = MagicMock()
            mock_resolver.get_agent_card = AsyncMock(
                side_effect=_resolver_side_effect
            )
            MockResolver.return_value = mock_resolver

            mock_http = AsyncMock()
            MockHttpx.return_value = mock_http
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)

            tools = await build_a2a_tools(access_token="tok")

        assert len(tools) == 1

    @pytest.mark.asyncio
    async def test_passes_context_id_to_tool_closure(self):
        card = _make_agent_card(
            skills=[{"id": "s", "description": "d", "examples": []}]
        )

        mock_send = AsyncMock(return_value="response")

        with patch(
            "template_agent.src.a2a.tools.settings"
        ) as mock_settings, patch(
            "template_agent.src.a2a.tools.A2ACardResolver"
        ) as MockResolver, patch(
            "template_agent.src.a2a.tools.httpx.AsyncClient"
        ) as MockHttpx, patch(
            "template_agent.src.a2a.tools.send_to_downstream_agent",
            mock_send,
        ):
            mock_settings.a2a_downstream_urls = ["http://agent:8080"]

            mock_resolver = MagicMock()
            mock_resolver.get_agent_card = AsyncMock(return_value=card)
            MockResolver.return_value = mock_resolver

            mock_http = AsyncMock()
            MockHttpx.return_value = mock_http
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)

            tools = await build_a2a_tools(
                access_token="tok", context_id="ctx-99", correlation_id="corr-1"
            )

            assert len(tools) == 1
            result = await tools[0].coroutine("test query")

        mock_send.assert_awaited_once_with(
            agent_url="http://agent:8080",
            message_text="test query",
            access_token="tok",
            context_id="ctx-99",
            correlation_id="corr-1",
        )
        assert result == "response"
