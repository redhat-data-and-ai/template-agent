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


class TestOfferConnect:
    """A declined turn must still offer the one click that fixes it.

    The Connect button was only ever built from a tool *failure*.
    Told up front that the warehouse server was not connected, the model
    correctly declined without calling anything, so nothing was
    recorded and the reply sent the reader to a settings page.
    """

    @staticmethod
    def _response(text: str, extra: dict | None = None):
        message = MagicMock()
        message.text = text
        message.additional_kwargs = extra if extra is not None else {}
        response = MagicMock()
        response.result = [message]
        return response, message

    @staticmethod
    def _offer(response, unconnected):
        # Built by a factory, so reach the method through an instance.
        instance = mw.build_mcp_status_middleware(user_id="u1")
        return instance._offer_connect(response, unconnected)

    def test_stamps_the_server_the_answer_names(self):
        response, message = self._response(
            "I cannot look up your manager because the warehouse server is not connected."
        )
        self._offer(response, ("warehouse", "miro"))

        assert message.additional_kwargs["auth_required"] == [
            {"mcp_name": "warehouse", "reason": "not_connected"}
        ]

    def test_stamps_nothing_when_the_answer_names_nothing(self):
        response, message = self._response("Your manager is Jane Roe.")
        self._offer(response, ("warehouse", "miro"))

        assert "auth_required" not in message.additional_kwargs

    def test_narrows_a_discovery_stamp_to_the_relevant_server(self):
        # Discovery records every unconnected server; five buttons
        # under an answer about one of them is the problem again.
        existing = {
            "auth_required": [
                {"mcp_name": "atlan", "reason": "not_connected"},
                {"mcp_name": "warehouse", "reason": "expired"},
                {"mcp_name": "miro", "reason": "not_connected"},
            ]
        }
        response, message = self._response(
            "the warehouse server is not connected, so I could not look that up.",
            existing,
        )
        self._offer(response, ("atlan", "warehouse", "miro"))

        assert message.additional_kwargs["auth_required"] == [
            {"mcp_name": "warehouse", "reason": "expired"}
        ]

    def test_keeps_the_full_list_when_the_answer_attributes_nothing(self):
        existing = {"auth_required": [{"mcp_name": "atlan", "reason": "not_connected"}]}
        response, message = self._response("Something went wrong.", existing)
        self._offer(response, ("atlan",))

        assert message.additional_kwargs["auth_required"] == [
            {"mcp_name": "atlan", "reason": "not_connected"}
        ]

    def test_a_tool_failure_reason_wins_over_the_derived_one(self):
        existing = {"auth_required": [{"mcp_name": "warehouse", "reason": "expired"}]}
        response, message = self._response(
            "the warehouse server needs reconnecting.", existing
        )
        self._offer(response, ("warehouse",))

        assert message.additional_kwargs["auth_required"] == [
            {"mcp_name": "warehouse", "reason": "expired"}
        ]


class TestConnectionStateCarriesTheAccount:
    """The notice must name who each integration acts as."""

    @staticmethod
    def _entries(entries):
        from unittest.mock import AsyncMock as _AsyncMock

        return patch(
            "deep_agent.aegra.mcp_oauth_handlers.handle_mcp_connections",
            _AsyncMock(return_value={"connections": entries}),
        )

    async def test_annotates_a_connected_integration_with_its_account(self):
        with self._entries(
            [{"mcp_name": "warehouse", "connected": True, "account": "alice@corp"}]
        ):
            connected, unconnected = await mw.connection_state("u1")

        assert connected == ["warehouse (as alice@corp)"]
        assert unconnected == []

    async def test_falls_back_to_the_bare_name_without_an_account(self):
        with self._entries(
            [{"mcp_name": "gitlab", "connected": True, "account": None}]
        ):
            connected, _ = await mw.connection_state("u1")

        assert connected == ["gitlab"]

    async def test_unconnected_names_stay_bare_so_matching_still_works(self):
        # narrow_auth_requests matches these against the reply text.
        with self._entries([{"mcp_name": "miro", "connected": False, "account": None}]):
            connected, unconnected = await mw.connection_state("u1")

        assert connected == []
        assert unconnected == ["miro"]
