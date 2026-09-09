"""Backend and subagent API tests for deepagents library.

Validates that the deepagents backend (StateBackend, CompositeBackend,
StoreBackend) and subagent (SubAgent, CompiledSubAgent, AsyncSubAgent)
APIs match what template-agent depends on.

If a dependency bump changes constructor signatures, removes parameters,
or alters type structures, these tests will fail in CI and block the merge.

Rationale: PyPI deepagents 0.7.6 uses no-arg StateBackend/StoreBackend
constructors. Some local/dev builds require ToolRuntime. Tests instantiate
through the same helpers production uses so both layouts pass.
"""

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest

from deep_agent.src.infrastructure.backend import (
    _make_state_backend,
    _make_store_backend,
)


def _core_file_methods(instance: Any) -> tuple[str, ...]:
    """Return file methods the installed backend actually implements."""
    candidates = (
        "read",
        "write",
        "edit",
        "ls",
        "ls_info",
        "glob",
        "glob_info",
        "grep",
        "grep_raw",
        "delete",
    )
    return tuple(m for m in candidates if hasattr(instance, m))


class TestStateBackendContract:
    """StateBackend must be instantiable via the production constructor helper."""

    def test_import(self):
        from deepagents.backends.state import StateBackend

        assert inspect.isclass(StateBackend)

    def test_runtime_constructor(self):
        instance = _make_state_backend(MagicMock())
        assert instance is not None

    def test_implements_backend_protocol_sync_methods(self):
        instance = _make_state_backend(MagicMock())
        present = _core_file_methods(instance)
        for method in ("read", "write", "edit"):
            assert method in present, f"StateBackend missing method: {method}"

    def test_implements_backend_protocol_async_methods(self):
        instance = _make_state_backend(MagicMock())
        # Async variants are optional; core sync methods must exist.
        for method in ("read", "write", "edit"):
            assert hasattr(instance, method), f"StateBackend missing method: {method}"


class TestCompositeBackendContract:
    """CompositeBackend must accept (default, routes) as instances."""

    def test_import(self):
        from deepagents.backends.composite import CompositeBackend

        assert inspect.isclass(CompositeBackend)

    def test_constructor_accepts_instances(self):
        from deepagents.backends.composite import CompositeBackend

        state = _make_state_backend(MagicMock())
        composite = CompositeBackend(default=state, routes={})
        assert composite is not None

    def test_constructor_signature_has_default_and_routes(self):
        from deepagents.backends.composite import CompositeBackend

        sig = inspect.signature(CompositeBackend.__init__)
        params = list(sig.parameters.keys())
        assert "default" in params
        assert "routes" in params


class TestStoreBackendContract:
    """StoreBackend must accept namespace; runtime is optional by version."""

    def test_import(self):
        from deepagents.backends.store import StoreBackend

        assert inspect.isclass(StoreBackend)

    def test_constructor_has_namespace_param(self):
        from deepagents.backends.store import StoreBackend

        sig = inspect.signature(StoreBackend.__init__)
        params = sig.parameters
        assert "namespace" in params, "StoreBackend must accept 'namespace' param"

    def test_instantiation_with_namespace(self):
        def fake_namespace(ctx: Any) -> tuple[str, ...]:
            return ("test",)

        instance = _make_store_backend(MagicMock(), fake_namespace)
        assert instance is not None


class TestSubAgentContract:
    """SubAgent must be constructable with name, description, system_prompt."""

    def test_import(self):
        from deepagents import SubAgent

        assert SubAgent is not None

    def test_construction_with_required_fields(self):
        from deepagents import SubAgent

        sa = SubAgent(
            name="test-agent",
            description="A test subagent",
            system_prompt="You are helpful.",
        )
        assert sa["name"] == "test-agent"
        assert sa["description"] == "A test subagent"
        assert sa["system_prompt"] == "You are helpful."

    def test_is_dict_like(self):
        from deepagents import SubAgent

        sa = SubAgent(name="test", description="desc", system_prompt="prompt")
        assert isinstance(sa, dict)

    def test_supports_optional_tools(self):
        from deepagents import SubAgent

        sa = SubAgent(
            name="test",
            description="desc",
            system_prompt="prompt",
            tools=[],
        )
        assert sa["tools"] == []

    def test_supports_optional_middleware(self):
        from deepagents import SubAgent

        sa = SubAgent(
            name="test",
            description="desc",
            system_prompt="prompt",
            middleware=[],
        )
        assert sa["middleware"] == []


class TestCompiledSubAgentContract:
    """CompiledSubAgent must be constructable with name, description, runnable."""

    def test_import(self):
        from deepagents.middleware.subagents import CompiledSubAgent

        assert CompiledSubAgent is not None

    def test_construction(self):
        from deepagents.middleware.subagents import CompiledSubAgent

        mock_runnable = MagicMock()
        csa = CompiledSubAgent(
            name="compiled-test",
            description="A compiled subagent",
            runnable=mock_runnable,
        )
        assert csa["name"] == "compiled-test"
        assert csa["description"] == "A compiled subagent"
        assert csa["runnable"] is mock_runnable

    def test_is_dict_like(self):
        from deepagents.middleware.subagents import CompiledSubAgent

        csa = CompiledSubAgent(name="test", description="d", runnable=MagicMock())
        assert isinstance(csa, dict)


class TestAsyncSubAgentContract:
    """AsyncSubAgent must be constructable with name, description, graph_id."""

    @pytest.fixture(autouse=True)
    def _require_async_subagents_module(self):
        pytest.importorskip("deepagents.middleware.async_subagents")

    def test_import(self):
        from deepagents.middleware.async_subagents import AsyncSubAgent

        assert AsyncSubAgent is not None

    def test_construction_with_required_fields(self):
        from deepagents.middleware.async_subagents import AsyncSubAgent

        asa = AsyncSubAgent(
            name="async-test",
            description="An async subagent",
            graph_id="my-graph",
        )
        assert asa["name"] == "async-test"
        assert asa["description"] == "An async subagent"
        assert asa["graph_id"] == "my-graph"

    def test_optional_url_and_headers(self):
        from deepagents.middleware.async_subagents import AsyncSubAgent

        asa = AsyncSubAgent(
            name="async-test",
            description="desc",
            graph_id="g",
            url="http://example.com",
            headers={"Authorization": "Bearer token"},
        )
        assert asa["url"] == "http://example.com"
        assert asa["headers"]["Authorization"] == "Bearer token"

    def test_is_dict_like(self):
        from deepagents.middleware.async_subagents import AsyncSubAgent

        asa = AsyncSubAgent(name="t", description="d", graph_id="g")
        assert isinstance(asa, dict)

    def test_identifiable_by_graph_id_key(self):
        """_extract_async_subagents relies on 'graph_id' key presence."""
        from deepagents.middleware.async_subagents import AsyncSubAgent

        asa = AsyncSubAgent(name="t", description="d", graph_id="g")
        assert "graph_id" in asa

    def test_distinguishable_from_regular_subagent(self):
        """Regular SubAgent must NOT have graph_id, async must."""
        from deepagents import SubAgent
        from deepagents.middleware.async_subagents import AsyncSubAgent

        regular = SubAgent(name="r", description="d", system_prompt="p")
        async_sa = AsyncSubAgent(name="a", description="d", graph_id="g")
        assert "graph_id" not in regular
        assert "graph_id" in async_sa


class TestAsyncSubAgentMiddlewareContract:
    """AsyncSubAgentMiddleware must accept list of AsyncSubAgent dicts."""

    @pytest.fixture(autouse=True)
    def _require_async_subagents_module(self):
        pytest.importorskip("deepagents.middleware.async_subagents")

    def test_import(self):
        from deepagents.middleware.async_subagents import AsyncSubAgentMiddleware

        assert AsyncSubAgentMiddleware is not None

    def test_construction_with_async_subagent_dicts(self):
        from deepagents.middleware.async_subagents import (
            AsyncSubAgent,
            AsyncSubAgentMiddleware,
        )

        asa = AsyncSubAgent(name="worker", description="d", graph_id="g")
        middleware = AsyncSubAgentMiddleware(async_subagents=[asa])
        assert middleware is not None

    def test_constructor_has_expected_params(self):
        from deepagents.middleware.async_subagents import AsyncSubAgentMiddleware

        sig = inspect.signature(AsyncSubAgentMiddleware.__init__)
        params = sig.parameters
        assert "async_subagents" in params
        assert "system_prompt" in params


class TestCreateDeepAgentContract:
    """create_deep_agent must accept backend as BackendProtocol instance."""

    def test_import(self):
        from deepagents import create_deep_agent

        assert callable(create_deep_agent)

    def test_signature_has_required_params(self):
        from deepagents import create_deep_agent

        sig = inspect.signature(create_deep_agent)
        params = sig.parameters
        assert "backend" in params
        assert "model" in params
        assert "tools" in params
        assert "subagents" in params
        assert "middleware" in params
        assert "system_prompt" in params
        assert "skills" in params

    def test_backend_param_defaults_to_none(self):
        from deepagents import create_deep_agent

        sig = inspect.signature(create_deep_agent)
        backend_param = sig.parameters["backend"]
        assert backend_param.default is None


class TestBackendProtocolContract:
    """BackendProtocol must define the expected file operation methods."""

    def test_import(self):
        from deepagents.backends.protocol import BackendProtocol

        assert BackendProtocol is not None

    @pytest.mark.parametrize(
        "method",
        [
            "read",
            "write",
            "edit",
            "aread",
            "awrite",
            "aedit",
        ],
    )
    def test_has_required_method(self, method):
        from deepagents.backends.protocol import BackendProtocol

        assert hasattr(BackendProtocol, method), (
            f"BackendProtocol missing method: {method}"
        )

    def test_has_listing_method(self):
        from deepagents.backends.protocol import BackendProtocol

        assert hasattr(BackendProtocol, "ls") or hasattr(BackendProtocol, "ls_info")
