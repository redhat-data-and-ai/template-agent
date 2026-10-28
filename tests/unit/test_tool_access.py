"""Unit tests for global tool exclusion (``middleware.excluded_tools``)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from langchain_core.messages import SystemMessage

from deep_agent.src.agent.config.middleware import (
    MiddlewareFileConfig,
    resolve_middleware,
)
from deep_agent.src.infrastructure.middleware import (
    _tool_name,
    build_restricted_filesystem_middleware,
    build_tool_access_middleware,
)

_AGENT_YAML = (
    Path(__file__).resolve().parents[2] / "config" / "agent" / "runtime" / "agent.yaml"
)


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _request(tools: list[SimpleNamespace], captured: dict) -> SimpleNamespace:
    """Minimal ModelRequest stand-in that records what override() receives."""

    class Req:
        def __init__(self) -> None:
            self.tools = tools
            self.system_message = SystemMessage("SYS")

        def override(self, **kwargs: object) -> "Req":
            captured.update(kwargs)
            return self

    return Req()  # type: ignore[return-value]


def _call(name: str) -> SimpleNamespace:
    return SimpleNamespace(tool_call={"name": name, "id": "call-1", "args": {}})


class TestToolAccessMiddleware:
    def test_returns_none_when_nothing_blocked(self):
        assert build_tool_access_middleware(frozenset()) is None

    def test_blocked_tool_hidden_from_model(self):
        mw = build_tool_access_middleware(frozenset({"delete"}))
        captured: dict = {}
        mw.wrap_model_call(
            _request([_tool("ls"), _tool("delete"), _tool("grep")], captured),
            lambda r: "ok",
        )
        assert [t.name for t in captured["tools"]] == ["ls", "grep"]

    def test_system_prompt_names_blocked_tools(self):
        mw = build_tool_access_middleware(frozenset({"delete"}))
        captured: dict = {}
        mw.wrap_model_call(_request([_tool("delete")], captured), lambda r: "ok")
        content = str(captured["system_message"].content)
        assert "disabled-tools" in content
        assert "delete" in content

    def test_blocked_call_refused_at_tool_node(self):
        """Hiding is not denying: a call that arrives anyway must not execute."""
        mw = build_tool_access_middleware(frozenset({"delete"}))
        result = mw.wrap_tool_call(_call("delete"), lambda r: "EXECUTED")
        assert result.status == "error"
        assert "not permitted" in result.content
        assert result.content != "EXECUTED"

    def test_allowed_call_passes_through(self):
        mw = build_tool_access_middleware(frozenset({"delete"}))
        assert mw.wrap_tool_call(_call("read_file"), lambda r: "EXECUTED") == "EXECUTED"


class TestToolAccessMiddlewareAsync:
    """LangGraph drives the async paths at runtime — they must match the sync ones."""

    async def test_blocked_tool_hidden_from_model(self):
        mw = build_tool_access_middleware(frozenset({"delete"}))
        captured: dict = {}

        async def handler(request):
            return "ok"

        await mw.awrap_model_call(
            _request([_tool("ls"), _tool("delete")], captured), handler
        )
        assert [t.name for t in captured["tools"]] == ["ls"]

    async def test_blocked_call_refused_at_tool_node(self):
        mw = build_tool_access_middleware(frozenset({"delete"}))

        async def handler(request):
            return "EXECUTED"

        result = await mw.awrap_tool_call(_call("delete"), handler)
        assert result.status == "error"
        assert "not permitted" in result.content

    async def test_allowed_call_passes_through(self):
        mw = build_tool_access_middleware(frozenset({"delete"}))

        async def handler(request):
            return "EXECUTED"

        assert await mw.awrap_tool_call(_call("read_file"), handler) == "EXECUTED"


class TestToolNameExtraction:
    def test_reads_name_from_a_tool_object(self):
        assert _tool_name(_tool("read_file")) == "read_file"

    def test_reads_name_from_a_dict_shaped_tool(self):
        """Provider-native tool specs arrive as dicts, not BaseTool."""
        assert _tool_name({"name": "web_search"}) == "web_search"

    def test_returns_none_when_name_is_absent_or_not_a_string(self):
        assert _tool_name({"type": "function"}) is None
        assert _tool_name({"name": 123}) is None
        assert _tool_name(object()) is None


class TestRestrictedFilesystemMiddleware:
    """The strongest layer: a blocked filesystem tool is never registered."""

    def _backend(self):
        from deepagents.backends.state import StateBackend

        return StateBackend()

    def test_blocked_tool_is_not_registered(self):
        mw = build_restricted_filesystem_middleware(
            self._backend(), frozenset({"delete"})
        )
        names = [t.name for t in mw.tools]
        assert "delete" not in names
        assert {"ls", "read_file", "write_file", "edit_file", "glob", "grep"} <= set(
            names
        )

    def test_name_matches_so_it_replaces_the_builtin(self):
        """create_deep_agent merges caller middleware by .name — must match."""
        from deepagents.middleware.filesystem import FilesystemMiddleware

        mw = build_restricted_filesystem_middleware(
            self._backend(), frozenset({"delete"})
        )
        assert mw.name == FilesystemMiddleware(backend=self._backend()).name

    def test_none_when_nothing_filesystem_related_is_blocked(self):
        assert (
            build_restricted_filesystem_middleware(
                self._backend(), frozenset({"some_mcp_tool"})
            )
            is None
        )

    def test_none_when_nothing_blocked(self):
        assert (
            build_restricted_filesystem_middleware(self._backend(), frozenset()) is None
        )

    def test_none_when_no_backend(self):
        assert (
            build_restricted_filesystem_middleware(None, frozenset({"delete"})) is None
        )

    def test_none_when_tools_cannot_be_enumerated(self):
        """A deepagents-side failure must not take the whole agent down."""
        with patch(
            "deepagents.middleware.filesystem.FilesystemMiddleware",
            side_effect=RuntimeError("backend rejected"),
        ):
            assert (
                build_restricted_filesystem_middleware(
                    self._backend(), frozenset({"delete"})
                )
                is None
            )


class TestSubagentCoverage:
    def test_subagents_get_the_global_restriction(self):
        """Delegation must not be a way around excluded_tools."""
        from deep_agent.src.infrastructure.subagents import _subagent_middleware

        built = _subagent_middleware("analyst", [], []) or []
        names = [type(m).__name__ for m in built]
        assert "FilesystemMiddleware" in names
        assert "ToolAccessMiddleware" in names

        fs = next(m for m in built if type(m).__name__ == "FilesystemMiddleware")
        assert "delete" not in [t.name for t in fs.tools]


class TestToolAccessConfig:
    def _resolved(self, overrides: dict | None = None):
        raw = yaml.safe_load(_AGENT_YAML.read_text())
        cfg = MiddlewareFileConfig.model_validate(
            {
                "defaults": raw.get("middleware", {}),
                "profiles": raw.get("harness_profiles", {}),
            }
        )
        return resolve_middleware(cfg, "gemini-3.1-pro-preview", overrides)

    def test_shipped_config_excludes_delete(self):
        assert "delete" in self._resolved().blocked_tools

    def test_agent_override_cannot_re_enable_an_excluded_tool(self):
        """excluded_tools is a global floor — an agent may add, not remove."""
        resolved = self._resolved({"excluded_tools": []})
        assert "delete" in resolved.blocked_tools

    def test_agent_override_can_exclude_additional_tools(self):
        resolved = self._resolved({"excluded_tools": ["execute"]})
        assert {"delete", "execute"} <= resolved.blocked_tools

    def test_agent_override_listing_the_same_tool_does_not_duplicate(self):
        resolved = self._resolved({"excluded_tools": ["delete"]})
        assert resolved.excluded_tools.count("delete") == 1
