"""Unit tests for aegra graph factory."""

import inspect
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_runtime_mock = MagicMock()
if "langgraph_sdk.runtime" not in sys.modules:
    sys.modules["langgraph_sdk.runtime"] = _runtime_mock


def _reset_graph_state() -> None:
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
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = ""
        with (
            patch("deep_agent.src.settings.settings", mock_settings),
            patch("deep_agent.src.pii.get_scrubber", return_value=None),
        ):
            yield

    @pytest.mark.asyncio
    async def test_builds_agent_without_user(self):
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
                "test_access_token", "test_refresh_token"
            )

    @pytest.mark.asyncio
    async def test_exposes_all_mcp_tools_when_mcps_declared_without_tool_list(self):
        mock_compiled = MagicMock()
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            "tools": [],
            "mcps": ["dataverse-mcp-prod1"],
        }
        mock_config.resolve_tools.return_value = []
        mock_config.resolve_agent_middleware.return_value = MagicMock(
            skills_enabled=True
        )

        mock_tool = MagicMock()
        mock_tool.name = "identify_dataproducts"

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
        assert mock_create.call_args.kwargs["tools"] == [mock_tool]
        mock_get_mcp.assert_awaited_once_with(
            sso_token=None, server_names=["dataverse-mcp-prod1"], user_id=None
        )

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
        assert all(v is True for v in call_kwargs["interrupt_on"].values())

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
        from deep_agent.aegra.graph import _graph_fingerprint

        result1 = _graph_fingerprint("model", "prompt", ["tool1", "tool2"])
        result2 = _graph_fingerprint("model", "prompt", ["tool1", "tool2"])
        assert result1 == result2

    def test_graph_fingerprint_differs_for_different_inputs(self):
        from deep_agent.aegra.graph import _graph_fingerprint

        fp1 = _graph_fingerprint("model-a", "prompt", ["tool1"])
        fp2 = _graph_fingerprint("model-b", "prompt", ["tool1"])
        assert fp1 != fp2

    def test_graph_fingerprint_tool_order_independent(self):
        from deep_agent.aegra.graph import _graph_fingerprint

        fp1 = _graph_fingerprint("model", "prompt", ["a", "b"])
        fp2 = _graph_fingerprint("model", "prompt", ["b", "a"])
        assert fp1 == fp2

    def test_invalidate_graph_cache_clears_caches(self):
        import time

        from deep_agent.aegra import graph
        from deep_agent.aegra.graph import invalidate_graph_cache

        graph._graph_cache["test_key"] = object()
        graph._graph_cache_ts["test_key"] = time.time()

        invalidate_graph_cache()

        assert len(graph._graph_cache) == 0
        assert len(graph._graph_cache_ts) == 0

    def test_append_safety_stop_instruction_appends_text(self):
        from deep_agent.aegra.graph import _append_safety_stop_instruction

        result = _append_safety_stop_instruction("base prompt")
        assert result.startswith("base prompt")
        assert "STOP ALL WORK" in result


class TestGraphCacheHit:
    """Tests for the cache hit path in the agent() factory."""

    @pytest.mark.asyncio
    async def test_returns_cached_graph_on_hit(self):
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
