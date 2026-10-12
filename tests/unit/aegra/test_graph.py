"""Unit tests for aegra graph factory."""

import inspect
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.src.capability.tool_proxy import CapabilityToolProxy

_runtime_mock = MagicMock()
if "langgraph_sdk.runtime" not in sys.modules:
    sys.modules["langgraph_sdk.runtime"] = _runtime_mock


def _reset_graph_state() -> None:
    """Clear the graph module's cache dicts so tests start from scratch."""
    from deep_agent.aegra import graph

    graph._graph_cache.clear()
    graph._graph_cache_ts.clear()


class TestAgentFactory:
    """Tests for the agent() graph factory function.

    The ``agent()`` function uses lazy imports inside its body, so
    patches must target the actual module where each symbol lives.

    The autouse fixture below disables Guardian and PII wrapping so
    every test can assert ``result is mock_compiled`` directly.
    """

    @pytest.fixture(autouse=True)
    def _no_guardian_wrapping(self):
        """Disable Guardian and PII so tests can assert on raw compiled graph."""
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = ""
        mock_settings.LIFECYCLE_PERSISTENCE_ENABLED = False
        mock_settings.MEMORY_ENABLED = False
        mock_settings.PYTHON_LOG_LEVEL = "WARNING"
        mock_settings.SAFETY_DANGEROUS_CONTENT = "BLOCK_MEDIUM_AND_ABOVE"
        mock_settings.SAFETY_HATE_SPEECH = "BLOCK_MEDIUM_AND_ABOVE"
        mock_settings.SAFETY_HARASSMENT = "BLOCK_LOW_AND_ABOVE"
        mock_settings.SAFETY_SEXUALLY_EXPLICIT = "BLOCK_LOW_AND_ABOVE"
        with (
            patch("deep_agent.src.settings.settings", mock_settings),
            patch("deep_agent.src.pii.get_scrubber", return_value=None),
            patch(
                "deep_agent.aegra.mcp_resource_tools._get_server_configs",
                return_value={},
            ),
        ):
            yield

    @pytest.mark.asyncio
    async def test_builds_agent_without_user(self):
        """Agent builds successfully when runtime.user is None."""
        mock_compiled = MagicMock()
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            "tools": [],
        }
        mock_config.resolve_tools.return_value = []
        mock_config.resolve_agent_middleware.return_value = MagicMock(
            skills_enabled=True
        )

        mock_runtime = MagicMock()
        mock_runtime.user = None

        _reset_graph_state()

        with (
            patch(
                "deep_agent.src.agent.config.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                return_value=None,
            ),
            patch(
                "deep_agent.src.agent.config.model.parse_model_config",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.cache.model_cache.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.resolve_memory_param",
                return_value=None,
            ),
            patch("deep_agent.aegra.graph._ensure_startup", new_callable=AsyncMock),
            patch("deepagents.create_deep_agent", return_value=mock_compiled),
            patch(
                "deep_agent.src.settings.settings.LIFECYCLE_PERSISTENCE_ENABLED",
                False,
            ),
        ):
            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)
            assert result is mock_compiled

    @pytest.mark.asyncio
    async def test_builds_agent_with_sso_token(self):
        """Agent refreshes SSO token when runtime.user provides one."""
        mock_compiled = MagicMock()
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            "tools": [],
        }
        mock_config.resolve_tools.return_value = []
        mock_config.resolve_agent_middleware.return_value = MagicMock(
            skills_enabled=True
        )

        mock_user = MagicMock()
        mock_user.access_token = "test_access_token"
        mock_user.refresh_token = "test_refresh_token"
        mock_user.identity = None
        mock_user.display_name = None
        mock_user.email = None

        mock_runtime = MagicMock()
        mock_runtime.user = mock_user

        _reset_graph_state()

        with (
            patch(
                "deep_agent.src.agent.config.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                return_value=None,
            ),
            patch(
                "deep_agent.src.agent.config.model.parse_model_config",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.cache.model_cache.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_agent.aegra.mcp.refresh_access_token",
                new_callable=AsyncMock,
                return_value="refreshed_token",
            ) as mock_refresh,
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.resolve_memory_param",
                return_value=None,
            ),
            patch("deep_agent.aegra.graph._ensure_startup", new_callable=AsyncMock),
            patch("deepagents.create_deep_agent", return_value=mock_compiled),
            patch(
                "deep_agent.src.settings.settings.LIFECYCLE_PERSISTENCE_ENABLED",
                False,
            ),
        ):
            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)
            assert result is mock_compiled
            mock_refresh.assert_awaited_once_with(
                "test_access_token", "test_refresh_token", user_id=None
            )

    @pytest.mark.asyncio
    async def test_exposes_all_mcp_tools_when_mcps_declared_without_tool_list(self):
        """Omitted tools key with declared MCPs grants all MCP tools via implicit manifest."""
        mock_compiled = MagicMock()
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            # 'tools:' key absent (not an explicit empty list) triggers
            # the implicit-all-mcp grant.
            "mcps": ["dataverse-mcp-prod1"],
        }
        mock_config.resolve_tools.return_value = []
        mock_config.resolve_agent_middleware.return_value = MagicMock(
            skills_enabled=True
        )

        mock_tool = MagicMock()
        mock_tool.name = "identify_dataproducts"
        mock_tool.description = "Identifies data products"
        mock_tool.args_schema = None

        mock_runtime = MagicMock()
        mock_runtime.user = None

        _reset_graph_state()

        with (
            patch(
                "deep_agent.src.agent.config.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                return_value=None,
            ),
            patch(
                "deep_agent.src.agent.config.model.parse_model_config",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.cache.model_cache.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[mock_tool],
            ) as mock_get_mcp,
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.resolve_memory_param",
                return_value=None,
            ),
            patch("deep_agent.aegra.graph._ensure_startup", new_callable=AsyncMock),
            patch(
                "deepagents.create_deep_agent", return_value=mock_compiled
            ) as mock_create,
            patch(
                "deep_agent.src.settings.settings.LIFECYCLE_PERSISTENCE_ENABLED",
                False,
            ),
        ):
            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)

        assert result is mock_compiled
        built_tools = mock_create.call_args.kwargs["tools"]
        # Tools are wrapped by CapabilityToolProxy; compare by name and
        # confirm the enforcement wrapper is present.
        assert [t.name for t in built_tools] == [mock_tool.name]
        assert all(isinstance(t, CapabilityToolProxy) for t in built_tools)
        mock_get_mcp.assert_awaited_once_with(
            sso_token=None, server_names=["dataverse-mcp-prod1"], user_id=None
        )

    @pytest.mark.asyncio
    async def test_explicit_tool_list_excludes_other_mcp_server_tools(self):
        """Explicit 'tools:' allow-list must not be widened with other
        MCP server tools."""
        mock_compiled = MagicMock()
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            "tools": ["identify_dataproducts"],
            "mcps": ["dataverse-mcp-prod1"],
        }
        mock_config.resolve_agent_middleware.return_value = MagicMock(
            skills_enabled=True
        )

        allowed_tool = MagicMock()
        allowed_tool.name = "identify_dataproducts"
        allowed_tool.description = "Identifies data products"
        allowed_tool.args_schema = None

        # Same MCP server also exposes this tool, but it is not in the
        # declared 'tools:' list -- it must NOT end up in the built agent.
        other_tool = MagicMock()
        other_tool.name = "delete_dataproducts"
        other_tool.description = "Deletes data products"
        other_tool.args_schema = None

        mock_config.resolve_tools.return_value = [allowed_tool]

        mock_runtime = MagicMock()
        mock_runtime.user = None

        _reset_graph_state()

        with (
            patch(
                "deep_agent.src.agent.config.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                return_value=None,
            ),
            patch(
                "deep_agent.src.agent.config.model.parse_model_config",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.cache.model_cache.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[allowed_tool, other_tool],
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.resolve_memory_param",
                return_value=None,
            ),
            patch("deep_agent.aegra.graph._ensure_startup", new_callable=AsyncMock),
            patch(
                "deepagents.create_deep_agent", return_value=mock_compiled
            ) as mock_create,
            patch(
                "deep_agent.src.settings.settings.LIFECYCLE_PERSISTENCE_ENABLED",
                False,
            ),
        ):
            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)

        assert result is mock_compiled
        built_tools = mock_create.call_args.kwargs["tools"]
        built_names = [t.name for t in built_tools]
        assert built_names == [allowed_tool.name]
        assert other_tool.name not in built_names
        assert all(isinstance(t, CapabilityToolProxy) for t in built_tools)

    @pytest.mark.asyncio
    async def test_explicit_tool_list_injects_capability_restriction_prompt(self):
        """Explicit 'tools:' list injects a capability restriction instruction
        into the system prompt so the LLM does not loop on MCP resources."""
        mock_compiled = MagicMock()
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            "tools": ["identify_dataproducts"],
            "mcps": ["dataverse-mcp-prod1"],
        }
        mock_config.resolve_agent_middleware.return_value = MagicMock(
            skills_enabled=True
        )

        allowed_tool = MagicMock()
        allowed_tool.name = "identify_dataproducts"
        allowed_tool.description = "Identifies data products"
        allowed_tool.args_schema = None

        mock_config.resolve_tools.return_value = [allowed_tool]

        mock_runtime = MagicMock()
        mock_runtime.user = None

        _reset_graph_state()

        with (
            patch(
                "deep_agent.src.agent.config.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                return_value=None,
            ),
            patch(
                "deep_agent.src.agent.config.model.parse_model_config",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.cache.model_cache.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[allowed_tool],
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.resolve_memory_param",
                return_value=None,
            ),
            patch("deep_agent.aegra.graph._ensure_startup", new_callable=AsyncMock),
            patch(
                "deepagents.create_deep_agent", return_value=mock_compiled
            ) as mock_create,
            patch(
                "deep_agent.src.settings.settings.LIFECYCLE_PERSISTENCE_ENABLED",
                False,
            ),
        ):
            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)

        assert result is mock_compiled
        system_prompt_used = mock_create.call_args.kwargs["system_prompt"]
        assert "Capability Restrictions" in system_prompt_used
        assert "don't have access" in system_prompt_used

    @pytest.mark.asyncio
    async def test_implicit_manifest_does_not_inject_capability_restriction(self):
        """Implicit manifest (tools: omitted) must NOT inject capability restriction."""
        mock_compiled = MagicMock()
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            "mcps": ["dataverse-mcp-prod1"],
        }
        mock_config.resolve_tools.return_value = []
        mock_config.resolve_agent_middleware.return_value = MagicMock(
            skills_enabled=True
        )

        mock_tool = MagicMock()
        mock_tool.name = "identify_dataproducts"
        mock_tool.description = "Identifies data products"
        mock_tool.args_schema = None

        mock_runtime = MagicMock()
        mock_runtime.user = None

        _reset_graph_state()

        with (
            patch(
                "deep_agent.src.agent.config.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                return_value=None,
            ),
            patch(
                "deep_agent.src.agent.config.model.parse_model_config",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.cache.model_cache.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[mock_tool],
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.resolve_memory_param",
                return_value=None,
            ),
            patch("deep_agent.aegra.graph._ensure_startup", new_callable=AsyncMock),
            patch(
                "deepagents.create_deep_agent", return_value=mock_compiled
            ) as mock_create,
            patch(
                "deep_agent.src.settings.settings.LIFECYCLE_PERSISTENCE_ENABLED",
                False,
            ),
        ):
            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)

        assert result is mock_compiled
        system_prompt_used = mock_create.call_args.kwargs["system_prompt"]
        assert "Capability Restrictions" not in system_prompt_used

    @pytest.mark.asyncio
    async def test_hitl_passes_interrupt_on_when_enabled(self):
        """create_deep_agent must receive a non-empty interrupt_on dict when HITL is enabled."""
        from deep_agent.src.agent.config.middleware import HumanApprovalConfig

        mock_compiled = MagicMock()
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            "tools": [],
        }
        mock_config.resolve_tools.return_value = []

        hitl_config = HumanApprovalConfig(enabled=True, mode="all", exclude=[])
        mock_mw = MagicMock(skills_enabled=True)
        mock_mw.human_approval = hitl_config
        mock_config.resolve_agent_middleware.return_value = mock_mw

        mock_runtime = MagicMock()
        mock_runtime.user = None

        # Give the mock a real signature that includes interrupt_on so that the
        # inspect.signature() check inside agent() sees the parameter.
        def _stub(*, interrupt_on=None, **kw): ...

        mock_create = MagicMock(return_value=mock_compiled)
        mock_create.__signature__ = inspect.signature(_stub)

        _reset_graph_state()

        with (
            patch("deep_agent.src.agent.config.agent_config", mock_config),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                return_value=None,
            ),
            patch(
                "deep_agent.src.agent.config.model.parse_model_config",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.cache.model_cache.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.resolve_memory_param",
                return_value=None,
            ),
            patch("deep_agent.aegra.graph._ensure_startup", new_callable=AsyncMock),
            patch("deepagents.create_deep_agent", new=mock_create),
            patch(
                "deep_agent.src.settings.settings.LIFECYCLE_PERSISTENCE_ENABLED",
                False,
            ),
        ):
            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)

        assert result is mock_compiled
        call_kwargs = mock_create.call_args.kwargs
        assert "interrupt_on" in call_kwargs, (
            "interrupt_on was not passed to create_deep_agent"
        )
        assert isinstance(call_kwargs["interrupt_on"], dict)
        assert len(call_kwargs["interrupt_on"]) > 0, (
            "interrupt_on dict must not be empty"
        )
        assert all(
            v is True or (isinstance(v, dict) and "when" in v)
            for v in call_kwargs["interrupt_on"].values()
        )

    @pytest.mark.asyncio
    async def test_hitl_omits_interrupt_on_when_disabled(self):
        """create_deep_agent must NOT receive interrupt_on when HITL is disabled."""
        from deep_agent.src.agent.config.middleware import HumanApprovalConfig

        mock_compiled = MagicMock()
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            "tools": [],
        }
        mock_config.resolve_tools.return_value = []

        hitl_config = HumanApprovalConfig(enabled=False)
        mock_mw = MagicMock(skills_enabled=True)
        mock_mw.human_approval = hitl_config
        mock_config.resolve_agent_middleware.return_value = mock_mw

        mock_runtime = MagicMock()
        mock_runtime.user = None

        def _stub(*, interrupt_on=None, **kw): ...

        mock_create = MagicMock(return_value=mock_compiled)
        mock_create.__signature__ = inspect.signature(_stub)

        _reset_graph_state()

        with (
            patch("deep_agent.src.agent.config.agent_config", mock_config),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                return_value=None,
            ),
            patch(
                "deep_agent.src.agent.config.model.parse_model_config",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.cache.model_cache.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.resolve_memory_param",
                return_value=None,
            ),
            patch("deep_agent.aegra.graph._ensure_startup", new_callable=AsyncMock),
            patch("deepagents.create_deep_agent", new=mock_create),
            patch(
                "deep_agent.src.settings.settings.LIFECYCLE_PERSISTENCE_ENABLED",
                False,
            ),
        ):
            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)

        assert result is mock_compiled
        call_kwargs = mock_create.call_args.kwargs
        assert "interrupt_on" not in call_kwargs, (
            "interrupt_on must not be passed when HITL is disabled"
        )


class TestGraphHelpers:
    """Tests for pure helper functions in deep_agent.aegra.graph."""

    def test_graph_fingerprint_is_deterministic(self):
        """Same inputs always produce the same fingerprint."""
        from deep_agent.aegra.graph import _graph_fingerprint

        result1 = _graph_fingerprint("model", "prompt", ["tool1", "tool2"])
        result2 = _graph_fingerprint("model", "prompt", ["tool1", "tool2"])
        assert result1 == result2

    def test_graph_fingerprint_differs_for_different_inputs(self):
        """Different model names produce different fingerprints."""
        from deep_agent.aegra.graph import _graph_fingerprint

        fp1 = _graph_fingerprint("model-a", "prompt", ["tool1"])
        fp2 = _graph_fingerprint("model-b", "prompt", ["tool1"])
        assert fp1 != fp2

    def test_graph_fingerprint_tool_order_independent(self):
        """Tool order does not affect the fingerprint."""
        from deep_agent.aegra.graph import _graph_fingerprint

        fp1 = _graph_fingerprint("model", "prompt", ["a", "b"])
        fp2 = _graph_fingerprint("model", "prompt", ["b", "a"])
        assert fp1 == fp2

    def test_graph_fingerprint_includes_mcps_and_resources(self):
        """MCP names and resource URIs affect the fingerprint."""
        from deep_agent.aegra.graph import _graph_fingerprint

        base = dict(model_name="model", system_prompt="prompt", tool_names=["t"])
        fp_none = _graph_fingerprint(**base)
        fp_empty = _graph_fingerprint(**base, mcp_names=[], resource_uris=[])
        fp_mcps = _graph_fingerprint(**base, mcp_names=["keep-me"])
        fp_resources = _graph_fingerprint(**base, resource_uris=["template://about"])
        fp_resources_order = _graph_fingerprint(
            **base, resource_uris=["template://echo/{text}", "template://about"]
        )
        fp_resources_order_rev = _graph_fingerprint(
            **base, resource_uris=["template://about", "template://echo/{text}"]
        )

        assert fp_none == fp_empty
        assert fp_mcps != fp_none
        assert fp_resources != fp_none
        assert fp_resources != fp_mcps
        assert fp_resources_order == fp_resources_order_rev

    def test_invalidate_graph_cache_clears_caches(self):
        """invalidate_graph_cache empties both the graph and timestamp caches."""
        import time

        from deep_agent.aegra import graph
        from deep_agent.aegra.graph import invalidate_graph_cache

        graph._graph_cache["test_key"] = object()
        graph._graph_cache_ts["test_key"] = time.time()

        invalidate_graph_cache()

        assert len(graph._graph_cache) == 0
        assert len(graph._graph_cache_ts) == 0

    def test_append_safety_stop_instruction_appends_text(self):
        """Safety stop instruction is appended to the base prompt."""
        from deep_agent.aegra.graph import _append_safety_stop_instruction

        result = _append_safety_stop_instruction("base prompt")
        assert result.startswith("base prompt")
        assert "STOP ALL WORK" in result

    def test_append_capability_restriction_appends_text(self):
        """Capability restriction instruction is appended to the base prompt."""
        from deep_agent.aegra.graph import _append_capability_restriction

        result = _append_capability_restriction("base prompt")
        assert result.startswith("base prompt")
        assert "Capability Restrictions" in result
        assert "don't have access" in result

    def test_append_capability_restriction_preserves_original(self):
        """Original prompt content is preserved after appending restriction."""
        from deep_agent.aegra.graph import _append_capability_restriction

        original = "You are a helpful assistant.\nDo your best."
        result = _append_capability_restriction(original)
        assert result.startswith(original.rstrip())
        assert "Capability Restrictions" in result


class TestGraphCacheHit:
    """Tests for the cache hit path in the agent() factory."""

    @pytest.mark.asyncio
    async def test_returns_cached_graph_on_hit(self):
        """Cache hit returns the previously compiled graph without rebuilding."""
        import time

        from deep_agent.aegra import graph

        _reset_graph_state()

        fixed_key = "deadbeefcafebabe"
        mock_cached_graph = MagicMock(name="cached_graph")
        graph._graph_cache[fixed_key] = mock_cached_graph
        graph._graph_cache_ts[fixed_key] = time.time()

        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            "tools": [],
        }
        mock_config.resolve_tools.return_value = []
        mock_config.resolve_agent_middleware.return_value = MagicMock(
            skills_enabled=True
        )
        mock_config.get_cache_config.return_value.graph.ttl = 3600

        mock_runtime = MagicMock()
        mock_runtime.user = None

        with (
            patch("deep_agent.src.agent.config.agent_config", mock_config),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                return_value=None,
            ),
            patch(
                "deep_agent.src.agent.config.model.parse_model_config",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.cache.model_cache.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.resolve_memory_param",
                return_value=None,
            ),
            patch("deep_agent.aegra.graph._ensure_startup", new_callable=AsyncMock),
            patch(
                "deep_agent.aegra.graph._graph_fingerprint",
                return_value=fixed_key,
            ),
            patch("deepagents.create_deep_agent") as mock_create,
        ):
            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)

        assert result is mock_cached_graph
        mock_create.assert_not_called()

    def _mock_orch_config(self, **orch_overrides):
        """Build a MagicMock agent config with overridable orchestrator settings."""
        mock_config = MagicMock()
        orch = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            "tools": [],
        }
        orch.update(orch_overrides)
        mock_config.get_orchestrator_config.return_value = orch
        mock_config.resolve_tools.return_value = []
        mock_config.resolve_agent_middleware.return_value = MagicMock(
            skills_enabled=True
        )
        mock_config.get_guardrails_config.return_value = MagicMock(enabled=False)
        return mock_config

    async def _build_agent(self, mock_config):
        """Helper that patches everything and calls agent(), returning key mocks."""
        mock_compiled = MagicMock()
        mock_runtime = MagicMock()
        mock_runtime.user = None
        mock_create = MagicMock(return_value=mock_compiled)
        mock_mw = MagicMock(return_value=[])
        mock_subs = MagicMock(return_value=None)
        mcp_servers = mock_config.get_mcp_servers.return_value
        if not isinstance(mcp_servers, dict):
            mcp_servers = {}
        _reset_graph_state()
        with (
            patch("deep_agent.src.agent.config.agent_config", mock_config),
            patch(
                "deep_agent.aegra.mcp_resource_tools._get_server_configs",
                return_value=mcp_servers,
            ),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                return_value=None,
            ),
            patch(
                "deep_agent.src.agent.config.model.parse_model_config",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.cache.model_cache.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                mock_subs,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                mock_mw,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.resolve_memory_param",
                return_value=None,
            ),
            patch("deep_agent.aegra.graph._ensure_startup", new_callable=AsyncMock),
            patch("deepagents.create_deep_agent", mock_create),
            patch(
                "deep_agent.src.settings.settings.LIFECYCLE_PERSISTENCE_ENABLED",
                False,
            ),
        ):
            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)
        return result, mock_compiled, mock_create, mock_mw, mock_subs

    @pytest.mark.asyncio
    async def test_attaches_resource_tools_when_mcp_enabled(self):
        """Resource tools are attached when at least one MCP server is enabled."""
        from deep_agent.aegra.mcp_resource_tools import (
            LIST_TOOL,
            READ_TOOL,
            TEMPLATES_TOOL,
        )

        mock_config = self._mock_orch_config()
        mock_config.get_mcp_servers.return_value = {
            "template-mcp-server": {"enabled": True},
            "template-mcp-server-dcr": {"enabled": False},
        }

        (
            result,
            mock_compiled,
            mock_create,
            mock_mw,
            mock_subs,
        ) = await self._build_agent(mock_config)

        assert result is mock_compiled
        names = [t.name for t in mock_create.call_args.kwargs["tools"]]
        assert names == [LIST_TOOL, TEMPLATES_TOOL, READ_TOOL]
        mw_names = mock_mw.call_args.kwargs["mcp_tool_names"]
        assert {LIST_TOOL, TEMPLATES_TOOL, READ_TOOL} <= set(mw_names)
        mock_subs.assert_called_once()
        assert mock_subs.call_args.kwargs["tools"] == []

    @pytest.mark.asyncio
    async def test_explicit_manifest_excludes_resource_tools(self):
        """When manifest is EXPLICIT, resource tools must NOT be merged in —
        otherwise the LLM uses them to loop around blocked MCP server tools."""
        mock_compiled = MagicMock()
        mock_config = self._mock_orch_config(
            tools=["some_tool"], mcps=["template-mcp-server"]
        )
        mock_config.get_mcp_servers.return_value = {
            "template-mcp-server": {"enabled": True},
        }
        # resolve_tools returns empty (tool name doesn't match) — triggers
        # EXPLICIT-empty path.
        mock_config.resolve_tools.return_value = []

        (
            result,
            mock_compiled,
            mock_create,
            mock_mw,
            mock_subs,
        ) = await self._build_agent(mock_config)

        assert result is mock_compiled
        built_names = [t.name for t in mock_create.call_args.kwargs["tools"]]
        # Resource tools must NOT appear.
        from deep_agent.aegra.mcp_resource_tools import (
            LIST_TOOL,
            READ_TOOL,
            TEMPLATES_TOOL,
        )

        assert LIST_TOOL not in built_names
        assert TEMPLATES_TOOL not in built_names
        assert READ_TOOL not in built_names

    @pytest.mark.asyncio
    async def test_resources_empty_list_allows_all(self):
        """An empty resources list means no URI filtering is applied."""
        mock_config = self._mock_orch_config(resources=[])
        mock_config.get_mcp_servers.return_value = {
            "template-mcp-server": {"enabled": True},
        }

        with patch(
            "deep_agent.aegra.mcp_resource_tools.build_mcp_resource_tools",
            return_value=[],
        ) as mock_build:
            await self._build_agent(mock_config)

        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs["allowed_uris"] is None

    @pytest.mark.asyncio
    async def test_resource_tools_honor_declared_mcps(self):
        """Resource tools are scoped to declared MCP servers only."""
        mock_config = self._mock_orch_config(mcps=["keep-me"])
        mock_config.get_mcp_servers.return_value = {
            "keep-me": {"enabled": True},
            "other": {"enabled": True},
        }

        with patch(
            "deep_agent.aegra.mcp_resource_tools.build_mcp_resource_tools",
            return_value=[],
        ) as mock_build:
            await self._build_agent(mock_config)

        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs["allowed_servers"] == ["keep-me"]
        assert mock_build.call_args.kwargs["allowed_uris"] is None

    @pytest.mark.asyncio
    async def test_resource_tools_honor_declared_resources(self):
        """Declared resource URIs are passed as the allowed_uris filter."""
        mock_config = self._mock_orch_config(
            resources=["template://about", "template://echo/{text}"]
        )
        mock_config.get_mcp_servers.return_value = {
            "template-mcp-server": {"enabled": True},
        }

        with patch(
            "deep_agent.aegra.mcp_resource_tools.build_mcp_resource_tools",
            return_value=[],
        ) as mock_build:
            await self._build_agent(mock_config)

        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs["allowed_uris"] == [
            "template://about",
            "template://echo/{text}",
        ]


class TestGuardianActivationGate:
    """Guardian wrapping requires BOTH guardrail config.enabled AND GUARDIAN_API_BASE."""

    def _build_mock_config(self, guardrails_enabled: bool) -> MagicMock:
        """Build a mock config with the given guardrails enabled flag."""
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            "tools": [],
        }
        mock_config.resolve_tools.return_value = []
        mock_config.resolve_agent_middleware.return_value = MagicMock(
            skills_enabled=True
        )
        guardrail_cfg = MagicMock()
        guardrail_cfg.enabled = guardrails_enabled
        mock_config.get_guardrails_config.return_value = guardrail_cfg
        return mock_config

    def _base_patches(self, mock_config, mock_settings):
        """Return the common list of context-manager patches for guardian tests."""
        return [
            patch("deep_agent.src.agent.config.agent_config", mock_config),
            patch("deep_agent.src.settings.settings", mock_settings),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                return_value=None,
            ),
            patch(
                "deep_agent.src.agent.config.model.parse_model_config",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.cache.model_cache.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_agent.aegra.mcp_resource_tools._get_server_configs",
                return_value={},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.resolve_memory_param",
                return_value=None,
            ),
            patch("deep_agent.aegra.graph._ensure_startup", new_callable=AsyncMock),
            patch("deep_agent.src.pii.get_scrubber", return_value=None),
        ]

    @pytest.mark.asyncio
    async def test_guardian_active_when_config_enabled_and_api_base_set(self):
        """Both guardrail.enabled=True and GUARDIAN_API_BASE set → wrap_tools + SafetyAwareRunnable."""
        import contextlib

        mock_compiled = MagicMock()
        mock_safety = MagicMock()
        mock_config = self._build_mock_config(guardrails_enabled=True)
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = "http://guardian.internal"
        mock_settings.LIFECYCLE_PERSISTENCE_ENABLED = False

        mock_runtime = MagicMock()
        mock_runtime.user = None
        _reset_graph_state()

        with contextlib.ExitStack() as stack:
            for p in self._base_patches(mock_config, mock_settings):
                stack.enter_context(p)
            mock_create = stack.enter_context(
                patch("deepagents.create_deep_agent", return_value=mock_compiled)
            )
            mock_wrap = stack.enter_context(
                patch(
                    "deep_agent.src.guardrails.tool_proxy.wrap_tools", return_value=[]
                )
            )
            mock_safety_cls = stack.enter_context(
                patch(
                    "deep_agent.aegra.safety.SafetyAwareRunnable",
                    return_value=mock_safety,
                )
            )

            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)

        mock_wrap.assert_called_once()
        mock_safety_cls.assert_called_once_with(mock_compiled, outermost=True)
        assert result is mock_safety
        system_prompt_used = mock_create.call_args.kwargs["system_prompt"]
        assert "STOP ALL WORK" in system_prompt_used

    @pytest.mark.asyncio
    async def test_guardian_inactive_when_config_disabled(self):
        """guardrail.enabled=False + GUARDIAN_API_BASE set → no wrapping applied."""
        import contextlib

        mock_compiled = MagicMock()
        mock_config = self._build_mock_config(guardrails_enabled=False)
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = "http://guardian.internal"
        mock_settings.LIFECYCLE_PERSISTENCE_ENABLED = False

        mock_runtime = MagicMock()
        mock_runtime.user = None
        _reset_graph_state()

        with contextlib.ExitStack() as stack:
            for p in self._base_patches(mock_config, mock_settings):
                stack.enter_context(p)
            mock_create = stack.enter_context(
                patch("deepagents.create_deep_agent", return_value=mock_compiled)
            )
            mock_wrap = stack.enter_context(
                patch("deep_agent.src.guardrails.tool_proxy.wrap_tools")
            )
            mock_safety_cls = stack.enter_context(
                patch("deep_agent.aegra.safety.SafetyAwareRunnable")
            )

            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)

        mock_wrap.assert_not_called()
        mock_safety_cls.assert_not_called()
        assert result is mock_compiled
        system_prompt_used = mock_create.call_args.kwargs["system_prompt"]
        assert "STOP ALL WORK" not in system_prompt_used

    @pytest.mark.asyncio
    async def test_guardian_inactive_when_api_base_absent(self):
        """guardrail.enabled=True + no GUARDIAN_API_BASE → no wrapping applied."""
        import contextlib

        mock_compiled = MagicMock()
        mock_config = self._build_mock_config(guardrails_enabled=True)
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = ""
        mock_settings.LIFECYCLE_PERSISTENCE_ENABLED = False

        mock_runtime = MagicMock()
        mock_runtime.user = None
        _reset_graph_state()

        with contextlib.ExitStack() as stack:
            for p in self._base_patches(mock_config, mock_settings):
                stack.enter_context(p)
            mock_create = stack.enter_context(
                patch("deepagents.create_deep_agent", return_value=mock_compiled)
            )
            mock_wrap = stack.enter_context(
                patch("deep_agent.src.guardrails.tool_proxy.wrap_tools")
            )
            mock_safety_cls = stack.enter_context(
                patch("deep_agent.aegra.safety.SafetyAwareRunnable")
            )

            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)

        mock_wrap.assert_not_called()
        mock_safety_cls.assert_not_called()
        assert result is mock_compiled
        system_prompt_used = mock_create.call_args.kwargs["system_prompt"]
        assert "STOP ALL WORK" not in system_prompt_used
