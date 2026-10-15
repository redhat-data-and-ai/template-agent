"""Capability counting across MCP servers.

The distinction worth keeping: `None` means the server does not
implement that capability, `0` means it implements it and has none.
A column of zeroes hides which is which, and one unsupported
capability must not cost the other two — servers routinely reject
`prompts/list` with method-not-found rather than an empty list.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import deep_agent.aegra.mcp_host as mh


def _session(**methods):
    return SimpleNamespace(**methods)


def _patch_session(session):
    @asynccontextmanager
    async def _fake(*args, **kwargs):
        yield session

    return patch.object(mh, "mcp_session", _fake)


class TestCountCapabilities:
    @pytest.mark.asyncio
    async def test_counts_each_capability(self):
        session = _session(
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[1, 2, 3])),
            list_resources=AsyncMock(return_value=SimpleNamespace(resources=[1])),
            list_prompts=AsyncMock(return_value=SimpleNamespace(prompts=[])),
        )
        with _patch_session(session):
            counts = await mh.count_capabilities("jira", user_id="u1", sso_token=None)

        assert counts == {"tools": 3, "resources": 1, "prompts": 0}

    @pytest.mark.asyncio
    async def test_unimplemented_is_none_not_zero(self):
        # "no prompts" and "does not do prompts" are different facts.
        session = _session(
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[1])),
        )
        with _patch_session(session):
            counts = await mh.count_capabilities("jira", user_id="u1", sso_token=None)

        assert counts["tools"] == 1
        assert counts["prompts"] is None
        assert counts["resources"] is None

    @pytest.mark.asyncio
    async def test_a_rejected_capability_does_not_cost_the_others(self):
        session = _session(
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[1, 2])),
            list_prompts=AsyncMock(side_effect=RuntimeError("method not found")),
            list_resources=AsyncMock(return_value=SimpleNamespace(resources=[1])),
        )
        with _patch_session(session):
            counts = await mh.count_capabilities("jira", user_id="u1", sso_token=None)

        assert counts["tools"] == 2
        assert counts["resources"] == 1
        assert counts["prompts"] is None

    @pytest.mark.asyncio
    async def test_a_result_without_the_expected_attribute_stays_none(self):
        session = _session(list_tools=AsyncMock(return_value=SimpleNamespace()))
        with _patch_session(session):
            counts = await mh.count_capabilities("jira", user_id="u1", sso_token=None)

        assert counts["tools"] is None
