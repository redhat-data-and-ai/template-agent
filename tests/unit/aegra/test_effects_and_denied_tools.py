"""A denied tool is taken away, not merely gated.

`build_interrupt_on` recorded denials and returned only the approval
map, and the graph called it with no `effects`. So a tool the policy
refuses outright stayed in the model's hands and ran without ever
reaching a prompt, and `effects: dry_run` did nothing — while the
admin endpoint read the setting from the PROMPT.md frontmatter, where
it never lives, and reported `live` for every deployment.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from deep_agent.src.agent.config.hitl import denied_tool_names
from deep_agent.src.agent.config.middleware import HumanApprovalConfig


def _tool(name: str):
    t = MagicMock()
    t.name = name
    return t


class TestDryRunDeniesActingTools:
    """`dry_run` turns "ask a human" into "refuse"."""

    CONFIG = HumanApprovalConfig(enabled=True, mode="all", exclude=[])

    def test_live_asks_rather_than_denies(self):
        denied = denied_tool_names(self.CONFIG, [_tool("write_file")], effects="live")
        assert "write_file" not in denied

    def test_dry_run_denies_the_same_tool(self):
        denied = denied_tool_names(
            self.CONFIG, [_tool("write_file")], effects="dry_run"
        )
        assert "write_file" in denied

    def test_an_excluded_tool_survives_dry_run(self):
        """Investigating is the point of a dry run.

        `dry_run` turns every `ask` into `deny`, so under `mode: all`
        it denies reads too — only a tool the policy already allows
        comes through. That is the knob: exclude what should stay
        usable while effects are off.
        """
        config = HumanApprovalConfig(enabled=True, mode="all", exclude=["read_file"])
        denied = denied_tool_names(config, [_tool("read_file")], effects="dry_run")
        assert "read_file" not in denied


class TestEffectsModeIsReadFromTheTopLevel:
    """`execution:` sits at the top level of agent.yaml."""

    def test_loader_reads_the_top_level_key(self):
        from deep_agent.src.agent.config.loader import AgentConfig
        from deep_agent.src.agent.config.middleware import ExecutionConfig

        # `object.__new__`, not `AgentConfig.__new__`: the class is a
        # singleton, so the latter hands back the process-wide
        # instance and stubbing `_ensure_loaded` on it disables config
        # loading for every test that runs afterwards.
        loader = object.__new__(AgentConfig)
        loader._execution_config = ExecutionConfig(effects="dry_run")
        loader._ensure_loaded = lambda: None  # type: ignore[method-assign]
        assert loader.get_effects_mode() == "dry_run"

    def test_the_admin_endpoint_reports_what_the_loader_says(self):
        from deep_agent.aegra import admin_routes

        with patch(
            "deep_agent.src.agent.config.agent_config.get_effects_mode",
            return_value="dry_run",
        ):
            assert admin_routes._effects_mode() == "dry_run"


class TestGraphCacheIsPerUser:
    """The compiled graph carries one person's integration state.

    `SelfKnowledgeMiddleware` closes over a user id and tells the
    model which integrations *that* user has connected. Cached under a
    key that omitted the user, the second person to ask inherited the
    first person's answer.
    """

    @staticmethod
    def _key(user: str):
        from deep_agent.aegra.graph import _graph_fingerprint

        return _graph_fingerprint(
            "gemini-3.8-flash",
            "prompt",
            ["read_file"],
            user_identity=user,
        )

    def test_two_users_do_not_share_a_graph(self):
        assert self._key("alice") != self._key("bob")

    def test_the_same_user_still_hits_the_cache(self):
        assert self._key("alice") == self._key("alice")

    def test_an_anonymous_run_is_its_own_bucket(self):
        assert self._key("") != self._key("alice")


class TestDenialReachesEverything:
    """Removing a denied tool from the orchestrator is not enough.

    Subagents are built from the unfiltered MCP list, and the
    deepagents built-ins never appear in that list at all. Both stayed
    reachable when the filter only trimmed the orchestrator's own
    tools, so `effects: dry_run` still let a subagent write files.
    """

    @staticmethod
    def _excluded():
        from deep_agent.src.infrastructure.middleware import _profile_excluded_tools

        return _profile_excluded_tools()

    def test_denied_names_join_the_shared_exclusion_list(self):
        from deep_agent.src.infrastructure.middleware import set_policy_denied_tools

        set_policy_denied_tools(["write_file", "dataverse_execute_sql"])
        try:
            excluded = self._excluded()
        finally:
            set_policy_denied_tools([])
        assert "write_file" in excluded
        assert "dataverse_execute_sql" in excluded

    def test_nothing_denied_adds_nothing(self):
        from deep_agent.src.infrastructure.middleware import set_policy_denied_tools

        set_policy_denied_tools([])
        before = self._excluded()
        assert "write_file" not in before

    def test_subagents_read_the_same_list(self):
        """One source, so the two cannot disagree."""
        import inspect

        from deep_agent.src.infrastructure import subagents

        source = inspect.getsource(subagents)
        assert "_build_tool_exclusions(_profile_excluded_tools(" in source
