"""What the agent is told about itself.

An agent that does not know its own limits spends turns discovering
them in front of a user — calling a denied tool, offering to edit a
read-only path, promising an action it is in dry run and cannot
take. Each reads as unreliability rather than as a constraint.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from deep_agent.src.self_knowledge.describe import build_self_description


def _policy(tools, decision, risk="medium"):
    return SimpleNamespace(tools=tools, decision=decision, risk=risk, reason="")


def _rule(operations, paths, mode):
    return SimpleNamespace(operations=operations, paths=paths, mode=mode)


def _config(policies=(), rules=(), skills=(), subagents=None, profiles=None):
    cfg = MagicMock()
    cfg.get_middleware_config.return_value = SimpleNamespace(
        defaults=SimpleNamespace(
            human_approval=SimpleNamespace(policies=list(policies), exclude=[])
        )
    )
    cfg.get_filesystem_config.return_value = SimpleNamespace(permissions=list(rules))
    cfg.get_available_skills.return_value = {s: None for s in skills}
    cfg.get_all_subagent_configs.return_value = subagents or {}
    cfg.get_providers_config.return_value = SimpleNamespace(
        harness_profiles=profiles or {}
    )
    return cfg


def _describe(cfg, model="", connected=None, unconnected=None, dry_run=False):
    with (
        patch("deep_agent.src.agent.config.agent_config", cfg),
        patch(
            "deep_agent.aegra.admin_routes._effects_mode",
            return_value="dry_run" if dry_run else "live",
        ),
    ):
        return build_self_description(model, connected, unconnected)


class TestWhatItCanDo:
    def test_lists_skills_by_name(self):
        out = _describe(_config(skills=["incident-triage", "data-discovery"]))
        assert "data-discovery, incident-triage" in out

    def test_names_subagents_and_what_they_reach(self):
        # What they reach, not what they are for: PROMPT.md says
        # what they are for, and this is the part config changes.
        cfg = _config(subagents={"investigator": {"mcps": ["datadog", "gitlab"]}})
        out = _describe(cfg)
        assert "`investigator`" in out
        assert "reaches datadog, gitlab" in out

    def test_lists_integrations_that_work_now(self):
        out = _describe(_config(), connected=["gitlab", "atlan"])
        assert "Integrations you can use now: atlan, gitlab" in out


class TestWhatItCannotDo:
    def test_names_unconnected_servers_and_what_to_tell_the_user(self):
        out = _describe(_config(), unconnected=["slack"])
        assert "slack" in out
        # The instruction matters as much as the list: without it
        # the model calls the tool and reports the error.
        assert "Settings" in out

    def test_dry_run_is_stated_as_a_fact_about_this_run(self):
        out = _describe(_config(), dry_run=True)
        assert "dry run" in out.lower()
        assert "do not promise" in out.lower()

    def test_denied_tools_are_not_offered(self):
        cfg = _config(policies=[_policy(["drop_*"], "deny")])
        out = _describe(cfg)
        assert "drop_*" in out
        assert "never run" in out

    def test_tools_that_pause_are_called_out_separately(self):
        cfg = _config(policies=[_policy(["write_file"], "ask")])
        out = _describe(cfg)
        assert "pause for a person" in out
        assert "write_file" in out

    def test_write_denied_paths(self):
        cfg = _config(rules=[_rule(["write", "edit"], ["documents/**"], "deny")])
        out = _describe(cfg)
        assert "read but not write: documents/**" in out

    def test_ignores_a_deny_rule_that_is_not_about_writing(self):
        cfg = _config(rules=[_rule(["read"], ["secrets/**"], "deny")])
        assert "secrets/**" not in _describe(cfg)

    def test_names_tools_the_model_does_not_have(self):
        profiles = {"gemini-3.8-flash": SimpleNamespace(excluded_tools=["shell"])}
        out = _describe(_config(profiles=profiles), model="gemini-3.8-flash")
        assert "gemini-3.8-flash" in out
        assert "shell" in out


class TestEmptiness:
    def test_nothing_to_say_is_an_empty_string(self):
        # A heading followed by nothing trains the model to skip the
        # section.
        assert _describe(_config()) == ""

    def test_unreadable_config_costs_the_section_not_the_turn(self):
        cfg = MagicMock()
        for name in (
            "get_middleware_config",
            "get_filesystem_config",
            "get_available_skills",
            "get_all_subagent_configs",
            "get_providers_config",
        ):
            getattr(cfg, name).side_effect = RuntimeError("bad yaml")
        assert _describe(cfg) == ""


class TestShape:
    def test_both_halves_are_headed_so_the_model_can_tell_them_apart(self):
        cfg = _config(skills=["a"], policies=[_policy(["drop_*"], "deny")])
        out = _describe(cfg)
        assert "## What you can do" in out
        assert "## What you cannot do" in out
        assert out.index("can do") < out.index("cannot do")


class TestModelIdentity:
    """A model does not reliably know its own version.

    Asked "which model do you use?" the agent answered "Gemini 3.7
    Flash" from training data, while every generation in the trace
    recorded gemini-3.8-flash. The deployment knows; this is the
    file whose job is telling it.
    """

    def test_states_the_model_it_is_running_on(self):
        from deep_agent.src.self_knowledge.describe import build_self_description

        out = build_self_description(model_name="gemini-3.8-flash")

        assert "gemini-3.8-flash" in out

    def test_says_nothing_when_the_model_is_unknown(self):
        from deep_agent.src.self_knowledge.describe import build_self_description

        out = build_self_description(model_name="")

        assert "running on the model" not in out
