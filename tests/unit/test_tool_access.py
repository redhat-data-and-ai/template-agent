"""Unit tests for global tool exclusion (``middleware.excluded_tools``)."""

from pathlib import Path
from types import SimpleNamespace

import yaml
from langchain_core.messages import SystemMessage

from deep_agent.src.agent.config.middleware import (
    MiddlewareFileConfig,
    resolve_middleware,
)
from deep_agent.src.infrastructure.middleware import (
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
