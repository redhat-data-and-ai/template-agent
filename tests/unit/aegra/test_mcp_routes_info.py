"""Unit tests for the /info endpoint in mcp_routes."""

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.aegra.mcp_routes import get_agent_info


@pytest.mark.asyncio
async def test_info_includes_tool_access_when_flag_enabled():
    """When tool_access_control.enabled=True, /info includes tool_access."""
    mock_config = MagicMock()
    mock_config.get_name.return_value = "test-agent"
    mock_config.get_mcp_servers.return_value = {}
    mock_config.get_orchestrator_config.return_value = {
        "name": "orchestrator",
        "allowed_tools": ["tool_a"],
        "denied_tools": ["tool_b"],
        "tool_approval": [],
    }
    mock_config.get_all_subagent_configs.return_value = {
        "analyst": {
            "name": "analyst",
            "type": "compiled",
            "allowed_tools": ["tool_a"],
            "denied_tools": [],
            "tool_approval": ["tool_a"],
        }
    }
    mock_config.get_middleware_config.return_value.defaults.tool_access_control.enabled = True

    with patch("deep_agent.aegra.mcp_routes.agent_config", mock_config):
        result = await get_agent_info()

    assert "tool_access" in result
    assert result["tool_access"]["orchestrator"]["denied_tools"] == ["tool_b"]
    assert len(result["tool_access"]["subagents"]) == 1


@pytest.mark.asyncio
async def test_info_omits_tool_access_when_flag_disabled():
    """When tool_access_control.enabled=False, /info omits tool_access."""
    mock_config = MagicMock()
    mock_config.get_name.return_value = "test-agent"
    mock_config.get_mcp_servers.return_value = {}
    mock_config.get_middleware_config.return_value.defaults.tool_access_control.enabled = False

    with patch("deep_agent.aegra.mcp_routes.agent_config", mock_config):
        result = await get_agent_info()

    assert "tool_access" not in result
    assert result["name"] == "test-agent"
