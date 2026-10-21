"""Unit tests for the HITL interrupt_on builder."""

from unittest.mock import MagicMock

import pytest

from deep_agent.src.agent.config.hitl import (
    _DEEPAGENTS_BUILTIN_TOOLS,
    _is_non_memory_path,
    build_interrupt_on,
)
from deep_agent.src.agent.config.middleware import HumanApprovalConfig


def _tool(name: str) -> MagicMock:
    t = MagicMock()
    t.name = name
    return t


class TestBuildInterruptOn:
    def test_disabled_returns_empty(self):
        config = HumanApprovalConfig(enabled=False, mode="all")
        result = build_interrupt_on(
            config, [_tool("send_email"), _tool("delete_record")]
        )
        assert result == {}

    def test_mode_none_returns_empty(self):
        config = HumanApprovalConfig(enabled=True, mode="none")
        result = build_interrupt_on(config, [_tool("send_email")])
        assert result == {}

    def test_mode_all_includes_explicit_tools(self):
        tools = [_tool("send_email"), _tool("search_web"), _tool("delete_record")]
        config = HumanApprovalConfig(enabled=True, mode="all")
        result = build_interrupt_on(config, tools)
        assert isinstance(result["send_email"], dict)
        assert isinstance(result["search_web"], dict)
        assert isinstance(result["delete_record"], dict)

    def test_mode_all_always_includes_builtins(self):
        """Built-in deepagents tools must be interrupted even with no explicit tools."""
        config = HumanApprovalConfig(enabled=True, mode="all")
        result = build_interrupt_on(config, [])
        for builtin in _DEEPAGENTS_BUILTIN_TOOLS:
            assert builtin in result, (
                f"built-in tool '{builtin}' missing from interrupt_on"
            )

    def test_empty_tool_list_still_covers_builtins(self):
        """An agent with no MCP tools still gets HITL for built-in filesystem tools."""
        config = HumanApprovalConfig(enabled=True, mode="all")
        result = build_interrupt_on(config, [])
        assert len(result) == len(_DEEPAGENTS_BUILTIN_TOOLS)
        for name in _DEEPAGENTS_BUILTIN_TOOLS:
            assert name in result
        # Filesystem tools are path-conditional (dict with 'when'):
        # memory paths are exempt, and skill reads are exempt.
        conditional = {"read_file", "write_file", "edit_file", "ls", "glob", "grep"}
        for name in _DEEPAGENTS_BUILTIN_TOOLS:
            if name in conditional:
                assert isinstance(result[name], dict), name
                assert "when" in result[name], name
            else:
                # Every entry now carries a description with risk and
                # rationale, so none are bare booleans.
                assert isinstance(result[name], dict), name
                assert "description" in result[name], name

    def test_exclude_removes_listed_tools(self):
        tools = [_tool("send_email"), _tool("search_web"), _tool("health_check")]
        config = HumanApprovalConfig(
            enabled=True,
            mode="all",
            exclude=["health_check", "search_web", "ls", "read_file"],
        )
        result = build_interrupt_on(config, tools)
        assert isinstance(result.get("send_email"), dict)
        assert "health_check" not in result
        assert "search_web" not in result
        assert "ls" not in result
        assert "read_file" not in result

    def test_exclude_nonexistent_tool_is_harmless(self):
        tools = [_tool("send_email")]
        config = HumanApprovalConfig(enabled=True, mode="all", exclude=["nonexistent"])
        result = build_interrupt_on(config, tools)
        assert isinstance(result["send_email"], dict)
        # builtins still present
        assert "ls" in result

    def test_all_tools_excluded_returns_empty(self):
        all_names = list(_DEEPAGENTS_BUILTIN_TOOLS) + ["send_email"]
        tools = [_tool("send_email")]
        config = HumanApprovalConfig(enabled=True, mode="all", exclude=all_names)
        result = build_interrupt_on(config, tools)
        assert result == {}

    def test_default_config_enabled(self):
        """Default HumanApprovalConfig (enabled=True, mode=all) should interrupt all tools."""
        config = HumanApprovalConfig()
        result = build_interrupt_on(config, [_tool("send_email")])
        assert isinstance(result["send_email"], dict)
        # Built-in tools should also be included
        for builtin in _DEEPAGENTS_BUILTIN_TOOLS:
            assert builtin in result


class TestIsNonMemoryPath:
    def test_interrupts_paths_outside_memories(self):
        req = MagicMock()
        req.tool_call = {"args": {"file_path": "/tmp/notes.md"}}
        assert _is_non_memory_path(req) is True

    def test_allows_memories_path(self):
        req = MagicMock()
        req.tool_call = {"args": {"path": "/memories/user_profile.md"}}
        assert _is_non_memory_path(req) is False


class TestGradedApproval:
    """Approval is per-tool, not all-or-nothing.

    A single global gate forces a choice between interrupting every
    directory listing and interrupting nothing; teams pick nothing, and
    the destructive calls stop being reviewed too.
    """

    @staticmethod
    def _tools(*names):
        return [type("T", (), {"name": n})() for n in names]

    def _config(self, **overrides):
        from deep_agent.src.agent.config.middleware import HumanApprovalConfig

        base = {
            "enabled": True,
            "mode": "all",
            "exclude": [],
            "default_decision": "ask",
            "policies": [
                {"tools": ["ls", "read_file"], "decision": "allow", "risk": "low"},
                {"tools": ["delete_*"], "decision": "ask", "risk": "high"},
                {"tools": ["rm_rf"], "decision": "deny", "risk": "high"},
            ],
        }
        base.update(overrides)
        return HumanApprovalConfig(**base)

    def test_allow_policy_skips_the_prompt(self):
        from deep_agent.src.agent.config.hitl import build_interrupt_on

        result = build_interrupt_on(self._config(), self._tools("ls", "read_file"))
        assert "ls" not in result
        assert "read_file" not in result

    def test_ask_policy_prompts(self):
        from deep_agent.src.agent.config.hitl import build_interrupt_on

        entry = build_interrupt_on(self._config(), self._tools("delete_record"))[
            "delete_record"
        ]
        assert entry["description"]

    def test_deny_policy_never_reaches_a_human(self):
        """A tool that can only be refused should not be offered."""
        from deep_agent.src.agent.config.hitl import build_interrupt_on

        assert "rm_rf" not in build_interrupt_on(self._config(), self._tools("rm_rf"))

    def test_denied_tools_are_reported(self):
        from deep_agent.src.agent.config.hitl import denied_tool_names

        assert "rm_rf" in denied_tool_names(self._config(), self._tools("rm_rf"))

    def test_glob_patterns_cover_tool_families(self):
        from deep_agent.src.agent.config.hitl import resolve_policy

        config = self._config()
        for name in ("delete_user", "delete_everything"):
            assert resolve_policy(name, config).decision == "ask"
            assert resolve_policy(name, config).risk == "high"

    def test_first_matching_policy_wins(self):
        from deep_agent.src.agent.config.hitl import resolve_policy

        config = self._config(
            policies=[
                {"tools": ["delete_tmp"], "decision": "allow"},
                {"tools": ["delete_*"], "decision": "ask", "risk": "high"},
            ]
        )
        assert resolve_policy("delete_tmp", config).decision == "allow"
        assert resolve_policy("delete_prod", config).decision == "ask"

    def test_unmatched_tool_uses_default_decision(self):
        from deep_agent.src.agent.config.hitl import resolve_policy

        assert resolve_policy("mystery_tool", self._config()).decision == "ask"

    def test_legacy_exclude_still_means_allow(self):
        """agent.yaml files predating policies must keep working."""
        from deep_agent.src.agent.config.hitl import build_interrupt_on

        config = self._config(exclude=["write_todos"], policies=[])
        result = build_interrupt_on(config, self._tools("write_todos", "other"))
        assert "write_todos" not in result
        assert isinstance(result["other"], dict)

    def test_no_policies_configured_matches_previous_behaviour(self):
        """With no policies, every tool still asks — unchanged from before."""
        from deep_agent.src.agent.config.hitl import build_interrupt_on

        config = self._config(policies=[])
        result = build_interrupt_on(config, self._tools("a", "b"))
        assert {"a", "b"} <= set(result)

    def test_memory_file_predicate_survives_policies(self):
        """The conditional memory-file approval must not be flattened."""
        from deep_agent.src.agent.config.hitl import (
            _MEMORY_FILE_TOOLS,
            build_interrupt_on,
        )

        tool = sorted(_MEMORY_FILE_TOOLS)[0]
        entry = build_interrupt_on(self._config(policies=[]), self._tools(tool))[tool]
        assert isinstance(entry, dict)
        assert "when" in entry

    def test_describe_policies_reports_risk(self):
        from deep_agent.src.agent.config.hitl import describe_policies

        described = describe_policies({"ls", "delete_record"}, self._config())
        assert described["ls"]["risk"] == "low"
        assert described["delete_record"]["risk"] == "high"


class TestShippedPolicies:
    """The defaults every forked agent inherits."""

    @staticmethod
    def _shipped():
        from pathlib import Path

        import yaml

        from deep_agent.src.agent.config.middleware import HumanApprovalConfig

        path = Path(__file__).parents[2] / "config" / "agent" / "runtime" / "agent.yaml"
        raw = yaml.safe_load(path.read_text())
        return HumanApprovalConfig(**raw["middleware"]["human_approval"])

    def test_read_only_tools_do_not_prompt(self):
        from deep_agent.src.agent.config.hitl import resolve_policy

        config = self._shipped()
        for name in ("ls", "read_file", "glob", "grep"):
            assert resolve_policy(name, config).decision == "allow", name

    def test_destructive_tools_prompt_at_high_risk(self):
        from deep_agent.src.agent.config.hitl import resolve_policy

        config = self._shipped()
        for name in ("delete_user", "drop_table", "execute"):
            policy = resolve_policy(name, config)
            assert policy.decision == "ask", name
            assert policy.risk == "high", name

    def test_unknown_tools_still_prompt(self):
        """An MCP server adding a tool must not get a silent pass."""
        from deep_agent.src.agent.config.hitl import resolve_policy

        assert resolve_policy("some_new_mcp_tool", self._shipped()).decision == "ask"

    def test_nothing_is_denied_by_default(self):
        """Shipping a deny would silently remove a tool from forked agents."""
        config = self._shipped()
        assert all(p.decision != "deny" for p in config.policies)


class TestEffectsAxis:
    """`effects: dry_run` lets the agent read and plan but not act.

    Implemented through the approval policy because that is already the
    enforcement point — a parallel mechanism would be a second place for
    a tool to slip through.
    """

    @staticmethod
    def _tools(*names):
        return [type("T", (), {"name": n})() for n in names]

    def _config(self):
        from deep_agent.src.agent.config.middleware import HumanApprovalConfig

        return HumanApprovalConfig(
            enabled=True,
            mode="all",
            default_decision="ask",
            policies=[
                {"tools": ["ls", "read_file"], "decision": "allow", "risk": "low"},
                {"tools": ["delete_*"], "decision": "ask", "risk": "high"},
            ],
        )

    def test_read_only_tools_still_run(self):
        from deep_agent.src.agent.config.hitl import resolve_policy

        assert (
            resolve_policy("ls", self._config(), effects="dry_run").decision == "allow"
        )

    def test_tools_that_would_ask_are_refused(self):
        from deep_agent.src.agent.config.hitl import resolve_policy

        assert (
            resolve_policy("delete_user", self._config(), effects="dry_run").decision
            == "deny"
        )

    def test_refusal_explains_itself(self):
        """The model needs to know it was dry-run, not that the tool is banned."""
        from deep_agent.src.agent.config.hitl import resolve_policy

        policy = resolve_policy("delete_user", self._config(), effects="dry_run")
        assert "dry run" in policy.reason

    def test_nothing_interrupts_in_dry_run(self):
        """No human is asked for a decision that will not be acted on."""
        from deep_agent.src.agent.config.hitl import build_interrupt_on

        result = build_interrupt_on(
            self._config(), self._tools("delete_user"), effects="dry_run"
        )
        assert result == {}

    def test_dry_run_denies_what_live_would_ask(self):
        from deep_agent.src.agent.config.hitl import denied_tool_names

        tools = self._tools("delete_user")
        assert denied_tool_names(self._config(), tools) == frozenset()
        assert "delete_user" in denied_tool_names(
            self._config(), tools, effects="dry_run"
        )

    def test_live_is_the_default(self):
        """An agent must not silently stop acting because of a typo."""
        from deep_agent.src.agent.config.hitl import resolve_policy

        assert resolve_policy("delete_user", self._config()).decision == "ask"

    def test_shipped_config_is_live(self):
        from pathlib import Path

        import yaml

        from deep_agent.src.agent.config.middleware import ExecutionConfig

        path = Path(__file__).parents[2] / "config" / "agent" / "runtime" / "agent.yaml"
        raw = yaml.safe_load(path.read_text())
        assert ExecutionConfig(**raw["execution"]).effects == "live"

    def test_invalid_effects_value_is_rejected(self):
        import pytest

        from deep_agent.src.agent.config.middleware import ExecutionConfig

        with pytest.raises(Exception):
            ExecutionConfig(effects="pretend")


class TestSkillReadsAreAlwaysApproved:
    """Reading a skill must never prompt.

    The agent reads `SKILL.md` to decide whether a skill applies at all.
    Asking first interrupts before the decision exists, and trains people
    to approve without reading. Writing to a skill is a real change and
    still asks.
    """

    @staticmethod
    def _req(path: str):
        from types import SimpleNamespace

        return SimpleNamespace(tool_call={"args": {"file_path": path}})

    @staticmethod
    def _tight_config():
        """Every filesystem tool set to ask, so the exemption is what we test."""
        from deep_agent.src.agent.config.middleware import HumanApprovalConfig

        return HumanApprovalConfig(
            enabled=True, mode="all", exclude=[], default_decision="ask", policies=[]
        )

    def _interrupt_on(self):
        from deep_agent.src.agent.config.hitl import build_interrupt_on

        return build_interrupt_on(self._tight_config(), [])

    @pytest.mark.parametrize("tool", ["read_file", "ls", "glob", "grep"])
    def test_read_tools_do_not_interrupt_on_a_skill_path(self, tool: str):
        entry = self._interrupt_on()[tool]

        assert isinstance(entry, dict), f"{tool} should be path-conditional"
        assert entry["when"](self._req("/skills/bmi-report/SKILL.md")) is False

    @pytest.mark.parametrize("tool", ["read_file", "ls", "glob", "grep"])
    def test_read_tools_still_interrupt_elsewhere(self, tool: str):
        """The exemption is for skills, not a blanket pass on reads."""
        entry = self._interrupt_on()[tool]

        assert entry["when"](self._req("/reports/quarterly.md")) is True

    @pytest.mark.parametrize("tool", ["write_file", "edit_file"])
    def test_writing_a_skill_still_asks(self, tool: str):
        """Modifying a skill changes what the agent can do later."""
        entry = self._interrupt_on()[tool]

        assert entry["when"](self._req("/skills/bmi-report/SKILL.md")) is True

    def test_memory_exemption_survives(self):
        """read_file is both a memory tool and a read tool; it needs both."""
        entry = self._interrupt_on()["read_file"]

        assert entry["when"](self._req("/memories/user_profile.md")) is False
        assert entry["when"](self._req("/skills/x/SKILL.md")) is False
        assert entry["when"](self._req("/elsewhere.md")) is True

    def test_nested_skill_resources_are_exempt(self):
        """References and assets are read the same way SKILL.md is."""
        entry = self._interrupt_on()["read_file"]

        for path in (
            "/skills/bmi-report/references/bmi_categories.md",
            "/skills/email-formatter/assets/template.html",
        ):
            assert entry["when"](self._req(path)) is False, path

    def test_a_path_merely_mentioning_skills_is_not_exempt(self):
        """Prefix match must not be fooled by a lookalike path."""
        entry = self._interrupt_on()["read_file"]

        assert entry["when"](self._req("/reports/skills-audit.md")) is True

    def test_missing_path_argument_does_not_crash(self):
        from types import SimpleNamespace

        entry = self._interrupt_on()["read_file"]
        req = SimpleNamespace(tool_call={"args": {}})

        assert entry["when"](req) is True

    def test_shipped_config_allows_skill_reads_outright(self):
        """With the shipped policy, read tools are `allow` and never gate."""
        from pathlib import Path

        import yaml

        from deep_agent.src.agent.config.hitl import build_interrupt_on
        from deep_agent.src.agent.config.middleware import HumanApprovalConfig

        path = Path(__file__).parents[2] / "config" / "agent" / "runtime" / "agent.yaml"
        raw = yaml.safe_load(path.read_text())
        config = HumanApprovalConfig(**raw["middleware"]["human_approval"])

        interrupt_on = build_interrupt_on(config, [])

        for tool in ("read_file", "ls", "glob", "grep"):
            assert tool not in interrupt_on, f"{tool} should not prompt at all"
