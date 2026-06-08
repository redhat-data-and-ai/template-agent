"""Unit tests for MCP tool cache in deep_agent/aegra/mcp.py.

Covers Bug 2: the module-level ``_cached_tools`` list has no cache key,
so ``server_names`` is completely ignored on a cache hit.  The first
caller's tool list is returned to every subsequent caller for the full
TTL window, regardless of which servers they requested.

Repro scenario (from production config):
    1. graph.py calls get_mcp_tools(server_names=["main-mcp"])
       → cache MISS  → fetches [validate_email] → stored in _cached_tools
    2. subagents.py / _build_compiled_subagent calls
       get_mcp_tools(server_names=["analytics-mcp"])
       → cache HIT   → returns [validate_email]   ← BUG
       (should return [calculate_bmi, search_web])
"""

import sys
import time
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch the broken langchain_mcp_adapters import before importing the module
# under test.  The installed langchain_mcp_adapters version references
# mcp.client.session.ElicitationFnT which does not exist in the pinned mcp
# package, making the real import fail at collection time.  We only need the
# MultiServerMCPClient symbol to be importable; its behaviour is fully mocked
# inside each test.
# ---------------------------------------------------------------------------

_fake_mcp_adapters_client = types.ModuleType("langchain_mcp_adapters.client")
_fake_mcp_adapters_client.MultiServerMCPClient = MagicMock()

_fake_mcp_adapters = types.ModuleType("langchain_mcp_adapters")
_fake_mcp_adapters.client = _fake_mcp_adapters_client

sys.modules.setdefault("langchain_mcp_adapters", _fake_mcp_adapters)
sys.modules.setdefault("langchain_mcp_adapters.client", _fake_mcp_adapters_client)

# Now safe to import the module under test
import deep_agent.aegra.mcp as mcp_module  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(name: str) -> MagicMock:
    """Create a minimal fake LangChain tool with the given name."""
    t = MagicMock()
    t.name = name
    return t


_FAKE_SERVERS = {
    "main-mcp": {
        "url": "http://main-mcp.internal",
        "enabled": True,
        "transport": "streamable_http",
        "timeout": 10,
    },
    "analytics-mcp": {
        "url": "http://analytics-mcp.internal",
        "enabled": True,
        "transport": "streamable_http",
        "timeout": 10,
    },
}


async def _fake_connect(name: str, config: dict, timeout: int, *, required: bool = False):
    """Simulate two MCP servers with completely different tool sets."""
    if name == "main-mcp":
        return [_tool("validate_email")]
    if name == "analytics-mcp":
        return [_tool("calculate_bmi"), _tool("search_web")]
    return []


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestMcpToolCache:
    """Tests for get_mcp_tools caching behaviour."""

    def setup_method(self):
        """Reset module-level cache state before each test."""
        mcp_module._tool_cache = {}
        mcp_module._tool_cache_ts = {}

    # ── Baseline: first call works correctly ────────────────────────────────

    async def test_cache_miss_fetches_correct_tools_for_main_mcp(self):
        """First call hits the network and returns the correct tools."""
        with (
            patch.object(mcp_module, "_get_server_configs", return_value=_FAKE_SERVERS),
            patch(
                "deep_agent.aegra.mcp._connect_single_server",
                side_effect=_fake_connect,
            ),
        ):
            tools = await mcp_module.get_mcp_tools(
                sso_token=None,
                server_names=["main-mcp"],
            )

        assert [t.name for t in tools] == ["validate_email"]

    async def test_cache_miss_fetches_correct_tools_for_analytics_mcp(self):
        """First call for analytics-mcp returns its own tools."""
        with (
            patch.object(mcp_module, "_get_server_configs", return_value=_FAKE_SERVERS),
            patch(
                "deep_agent.aegra.mcp._connect_single_server",
                side_effect=_fake_connect,
            ),
        ):
            tools = await mcp_module.get_mcp_tools(
                sso_token=None,
                server_names=["analytics-mcp"],
            )

        assert [t.name for t in tools] == ["calculate_bmi", "search_web"]

    # ── Bug 2: second caller with different server_names gets wrong tools ───

    async def test_bug2_analyst_gets_own_tools_after_orchestrator_cached(self):
        """
        Each caller must get tools for the server_names THEY requested.

        Orchestrator calls get_mcp_tools(server_names=["main-mcp"])
            → cache MISS → connects → gets [validate_email] → stored
        Analyst calls   get_mcp_tools(server_names=["analytics-mcp"])
            → must fetch analytics-mcp separately
            → must return [calculate_bmi, search_web]

        FAILS now (Bug 2): cache hit returns orchestrator's [validate_email].
        PASSES after fix:  cache is keyed by server_names.
        """
        connect_mock = AsyncMock(side_effect=_fake_connect)

        with (
            patch.object(mcp_module, "_get_server_configs", return_value=_FAKE_SERVERS),
            patch("deep_agent.aegra.mcp._connect_single_server", connect_mock),
        ):
            # Orchestrator goes first — populates the cache
            orchestrator_tools = await mcp_module.get_mcp_tools(
                sso_token=None,
                server_names=["main-mcp"],
            )
            assert [t.name for t in orchestrator_tools] == ["validate_email"]

            # Analyst requests a completely different server
            analyst_tools = await mcp_module.get_mcp_tools(
                sso_token=None,
                server_names=["analytics-mcp"],
            )

        # analytics-mcp must have been contacted for the analyst
        assert connect_mock.call_count == 2, (
            f"Expected 2 connect calls (one per server), got {connect_mock.call_count}. "
            "analytics-mcp was never contacted — cache served the wrong result."
        )

        analyst_tool_names = [t.name for t in analyst_tools]
        assert analyst_tool_names == ["calculate_bmi", "search_web"], (
            f"Analyst got {analyst_tool_names!r} instead of ['calculate_bmi', 'search_web']. "
            "Cache returned orchestrator's tools to the analyst."
        )

    async def test_bug2_orchestrator_gets_own_tools_after_analyst_cached(self):
        """
        Same scenario in reverse: orchestrator must not get analyst's cached tools.

        Analyst calls       get_mcp_tools(server_names=["analytics-mcp"])
            → cache MISS → gets [calculate_bmi, search_web] → stored
        Orchestrator calls  get_mcp_tools(server_names=["main-mcp"])
            → must fetch main-mcp separately
            → must return [validate_email]

        FAILS now (Bug 2): cache hit returns analyst's [calculate_bmi, search_web].
        PASSES after fix:  cache is keyed by server_names.
        """
        connect_mock = AsyncMock(side_effect=_fake_connect)

        with (
            patch.object(mcp_module, "_get_server_configs", return_value=_FAKE_SERVERS),
            patch("deep_agent.aegra.mcp._connect_single_server", connect_mock),
        ):
            # Analyst goes first
            await mcp_module.get_mcp_tools(
                sso_token=None,
                server_names=["analytics-mcp"],
            )

            # Orchestrator requests its own server
            orchestrator_tools = await mcp_module.get_mcp_tools(
                sso_token=None,
                server_names=["main-mcp"],
            )

        assert connect_mock.call_count == 2, (
            f"Expected 2 connect calls (one per server), got {connect_mock.call_count}. "
            "main-mcp was never contacted — cache served the analyst's tools to the orchestrator."
        )

        orchestrator_tool_names = [t.name for t in orchestrator_tools]
        assert orchestrator_tool_names == ["validate_email"], (
            f"Orchestrator got {orchestrator_tool_names!r} instead of ['validate_email']. "
            "Cache returned analyst's tools to the orchestrator."
        )

    # ── Control: same server_names = cache hit is correct behaviour ─────────

    async def test_same_server_names_cache_hit_is_correct(self):
        """Cache hit with the same server_names is correct and expected."""
        connect_mock = AsyncMock(side_effect=_fake_connect)

        with (
            patch.object(mcp_module, "_get_server_configs", return_value=_FAKE_SERVERS),
            patch("deep_agent.aegra.mcp._connect_single_server", connect_mock),
        ):
            first = await mcp_module.get_mcp_tools(
                sso_token=None, server_names=["main-mcp"]
            )
            second = await mcp_module.get_mcp_tools(
                sso_token=None, server_names=["main-mcp"]
            )

        # Only one real connection attempt — second is a correct cache hit
        assert connect_mock.call_count == 1
        assert [t.name for t in first] == [t.name for t in second] == ["validate_email"]

    # ── TTL expiry forces a fresh fetch ─────────────────────────────────────

    async def test_expired_cache_refetches_from_network(self):
        """After TTL expires the cache is bypassed and tools are re-fetched."""
        connect_mock = AsyncMock(side_effect=_fake_connect)

        with (
            patch.object(mcp_module, "_get_server_configs", return_value=_FAKE_SERVERS),
            patch("deep_agent.aegra.mcp._connect_single_server", connect_mock),
        ):
            # Pre-populate with a stale cache entry for the same key
            stale_key = frozenset(["main-mcp"])
            mcp_module._tool_cache[stale_key] = [_tool("stale_tool")]
            mcp_module._tool_cache_ts[stale_key] = time.time() - 99999

            tools = await mcp_module.get_mcp_tools(
                sso_token=None,
                server_names=["main-mcp"],
            )

        # Network was hit despite pre-populated cache (TTL expired)
        assert connect_mock.call_count == 1
        assert [t.name for t in tools] == ["validate_email"]
