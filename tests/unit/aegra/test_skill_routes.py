"""The skills report."""

from __future__ import annotations

import pytest


class TestSkillsReport:
    @pytest.mark.asyncio
    async def test_attached_skills_come_first(self):
        # Eight skills ship dormant, waiting on an MCP server that is
        # commented out. Interleaved alphabetically with the live
        # ones, a reader has to check `attached` on every row to learn
        # what the agent can currently do.
        from deep_agent.aegra.skill_routes import list_skills

        skills = (await list_skills())["skills"]
        attached = [s["attached"] for s in skills]
        assert attached == sorted(attached, reverse=True)

    @pytest.mark.asyncio
    async def test_every_skill_reports_a_description(self):
        from deep_agent.aegra.skill_routes import list_skills

        for skill in (await list_skills())["skills"]:
            assert skill["description"], f"{skill['name']} has no description"

    @pytest.mark.asyncio
    async def test_a_dormant_skill_names_no_agent(self):
        from deep_agent.aegra.skill_routes import list_skills

        for skill in (await list_skills())["skills"]:
            assert bool(skill["agents"]) == skill["attached"]

    @pytest.mark.asyncio
    async def test_source_splits_frontmatter_from_body(self):
        from deep_agent.aegra.skill_routes import read_skill

        files = (await read_skill("incident-triage"))["files"]
        skill_md = next(f for f in files if f["path"] == "SKILL.md")
        assert skill_md["frontmatter"]["name"] == "incident-triage"
        # The body must not still carry the frontmatter delimiters,
        # or the rendered view shows them as a horizontal rule.
        assert not skill_md["body"].lstrip().startswith("---")

    @pytest.mark.asyncio
    async def test_an_unknown_skill_is_404_not_a_path(self):
        from fastapi import HTTPException

        from deep_agent.aegra.skill_routes import read_skill

        with pytest.raises(HTTPException) as exc:
            await read_skill("../../etc/passwd")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_subagents_report_their_reach(self):
        # The useful thing about a subagent is its blast radius:
        # "can reach Datadog and GitLab" says more than its prose.
        from deep_agent.aegra.skill_routes import list_subagents

        agents = (await list_subagents())["subagents"]
        assert agents
        for agent in agents:
            assert "mcps" in agent and "skills" in agent

    @pytest.mark.asyncio
    async def test_a_subagent_splits_frontmatter_from_body(self):
        from deep_agent.aegra.skill_routes import read_subagent

        doc = (await read_subagent("curator"))["files"][0]
        assert doc["frontmatter"]["name"] == "curator"
        assert not doc["body"].lstrip().startswith("---")

    @pytest.mark.asyncio
    async def test_an_unknown_subagent_is_404_not_a_path(self):
        from fastapi import HTTPException

        from deep_agent.aegra.skill_routes import read_subagent

        with pytest.raises(HTTPException) as exc:
            await read_subagent("../../etc/passwd")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_the_prompt_splits_frontmatter_from_body(self):
        from deep_agent.aegra.skill_routes import read_prompt

        doc = (await read_prompt())["files"][0]
        assert doc["path"] == "PROMPT.md"
        # The routing table lives in the body, not the frontmatter.
        assert len(doc["body"]) > 500
        assert "mcps" in doc["frontmatter"]

    @pytest.mark.asyncio
    async def test_whoami_reports_the_role_rather_than_gating_on_it(self):
        """It exists so another service can decide; refusing
        non-developers would make it useless for establishing that
        someone is not one."""
        from unittest.mock import MagicMock

        from deep_agent.aegra.skill_routes import whoami

        request = MagicMock()
        request.headers = {}
        out = await whoami(request)
        assert "user_id" in out
        assert "is_developer" in out

    @pytest.mark.asyncio
    async def test_an_unresolvable_role_is_not_privileged(self):
        # `None` means the lookup failed. A failure must not grant
        # access — the BFF treats anything other than True as a no.
        from unittest.mock import MagicMock, patch

        from deep_agent.aegra import skill_routes

        async def boom(_user):
            raise RuntimeError("ldap down")

        request = MagicMock()
        request.headers = {}
        # Only reachable with auth on; with it off there is no role
        # to resolve and the dev user is the operator.
        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch("deep_agent.src.ldap.resolve_user_role", boom),
            patch(
                "deep_agent.aegra.auth_helpers.authenticated_user_id",
                return_value="someone",
            ),
        ):
            out = await skill_routes.whoami(request)
        assert out["is_developer"] is None

    @pytest.mark.asyncio
    async def test_with_auth_off_the_dev_user_is_the_operator(self):
        """The UI already assigns `owners` when auth is off. The
        agent resolving `users` instead is how the two came to
        disagree about the single local user."""
        from unittest.mock import MagicMock

        from deep_agent.aegra.skill_routes import whoami

        request = MagicMock()
        request.headers = {}
        out = await whoami(request)
        assert out["is_developer"] is True
