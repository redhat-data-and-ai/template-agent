"""The notice telling the model which integrations are usable.

The design constraint worth testing: a *wrong* list is worse than no
list. Telling the model an integration is unavailable when it works
stops it using a tool the user is waiting on.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.src.mcp_status import middleware as mw


class TestBuildNotice:
    def test_names_the_connected_servers(self):
        notice = mw.build_notice(["jira", "gitlab"], [])
        assert "Connected and usable: gitlab, jira." == notice

    def test_tells_the_model_not_to_call_unconnected_tools(self):
        notice = mw.build_notice([], ["slack"])
        assert "slack" in notice
        # The instruction matters as much as the list: without it the
        # model calls the tool and reports the error to the user.
        assert "Settings" in notice

    def test_both_halves_when_some_are_connected(self):
        notice = mw.build_notice(["jira"], ["slack"])
        assert "Connected and usable" in notice
        assert "Not connected" in notice

    def test_empty_when_there_is_nothing_to_say(self):
        assert mw.build_notice([], []) == ""

    def test_sorts_so_the_prompt_is_stable(self):
        # An unstable notice would change the prompt between turns and
        # invalidate the provider's cache for no reason.
        assert mw.build_notice(["b", "a"], []) == mw.build_notice(["a", "b"], [])


class TestConnectionState:
    @pytest.mark.asyncio
    async def test_splits_connected_from_unconnected(self):
        connections = {
            "connections": [
                {"mcp_name": "jira", "connected": True},
                {"mcp_name": "slack", "connected": False},
            ]
        }
        with patch(
            "deep_agent.aegra.mcp_oauth_handlers.handle_mcp_connections",
            new=AsyncMock(return_value=connections),
        ):
            assert await mw.connection_state("u1") == (["jira"], ["slack"])

    @pytest.mark.asyncio
    async def test_a_failure_suppresses_the_notice_entirely(self):
        with patch(
            "deep_agent.aegra.mcp_oauth_handlers.handle_mcp_connections",
            new=AsyncMock(side_effect=RuntimeError("redis down")),
        ):
            assert await mw.connection_state("u1") == ([], [])

    @pytest.mark.asyncio
    async def test_an_empty_result_is_not_an_error(self):
        with patch(
            "deep_agent.aegra.mcp_oauth_handlers.handle_mcp_connections",
            new=AsyncMock(return_value=None),
        ):
            assert await mw.connection_state("u1") == ([], [])


def _request(state=None, **attrs):
    request = MagicMock()
    request.state = state if state is not None else {}
    request.system_message = attrs.get("system_message")
    request.override = MagicMock(return_value="overridden")
    return request


class TestBuildMcpStatusMiddleware:
    def test_builds_a_middleware_instance(self):
        assert mw.build_mcp_status_middleware("u1") is not None

    @pytest.mark.asyncio
    async def test_attaches_the_notice_to_the_system_message(self):
        middleware = mw.build_mcp_status_middleware("u1")
        handler = AsyncMock(return_value="answered")
        request = _request()

        with patch.object(
            mw, "connection_state", new=AsyncMock(return_value=(["jira"], ["slack"]))
        ):
            result = await middleware.awrap_model_call(request, handler)

        assert result == "answered"
        request.override.assert_called_once()
        attached = request.override.call_args.kwargs["system_message"]
        assert "jira" in attached.content and "slack" in attached.content

    @pytest.mark.asyncio
    async def test_appends_to_an_existing_system_message(self):
        from langchain_core.messages import SystemMessage

        middleware = mw.build_mcp_status_middleware("u1")
        request = _request(system_message=SystemMessage(content="You are an agent."))

        with patch.object(
            mw, "connection_state", new=AsyncMock(return_value=(["jira"], []))
        ):
            await middleware.awrap_model_call(request, AsyncMock())

        attached = request.override.call_args.kwargs["system_message"]
        assert attached.content.startswith("You are an agent.")
        assert "jira" in attached.content

    @pytest.mark.asyncio
    async def test_appends_a_block_to_structured_content(self):
        from langchain_core.messages import SystemMessage

        middleware = mw.build_mcp_status_middleware("u1")
        request = _request(
            system_message=SystemMessage(content=[{"type": "text", "text": "base"}])
        )
        with patch.object(
            mw, "connection_state", new=AsyncMock(return_value=(["jira"], []))
        ):
            await middleware.awrap_model_call(request, AsyncMock())

        blocks = request.override.call_args.kwargs["system_message"].content
        assert blocks[-1]["text"].startswith("Connected")

    @pytest.mark.asyncio
    async def test_no_notice_leaves_the_request_untouched(self):
        middleware = mw.build_mcp_status_middleware("u1")
        request = _request()
        with patch.object(mw, "connection_state", new=AsyncMock(return_value=([], []))):
            await middleware.awrap_model_call(request, AsyncMock())
        request.override.assert_not_called()

    @pytest.mark.asyncio
    async def test_caches_so_every_turn_does_not_pay_a_round_trip(self):
        middleware = mw.build_mcp_status_middleware("u1")
        state = AsyncMock(return_value=(["jira"], []))
        with patch.object(mw, "connection_state", new=state):
            await middleware.awrap_model_call(_request(), AsyncMock())
            await middleware.awrap_model_call(_request(), AsyncMock())
        assert state.await_count == 1

    def test_sync_path_uses_whatever_the_async_path_cached(self):
        # The sync path cannot await the lookup, and blocking the event
        # loop to find out would cost more than the notice is worth.
        middleware = mw.build_mcp_status_middleware("u1")
        request = _request()
        handler = MagicMock(return_value="answered")
        assert middleware.wrap_model_call(request, handler) == "answered"
        request.override.assert_not_called()
