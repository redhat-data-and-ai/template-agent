"""The middleware that puts the self-description in front of the model."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.src.self_knowledge import middleware as mw


def _request(system_message=None):
    request = MagicMock()
    request.system_message = system_message
    request.override = MagicMock(return_value="overridden")
    return request


def _build(notice="## What you cannot do\n\n- nothing", **kwargs):
    with patch(
        "deep_agent.src.self_knowledge.describe.build_self_description",
        return_value=notice,
    ):
        return mw.build_self_knowledge_middleware(**kwargs)


class TestBuild:
    def test_none_when_there_is_nothing_to_say(self):
        assert _build(notice="") is None

    def test_built_when_there_is(self):
        assert _build() is not None


class TestApply:
    @pytest.mark.asyncio
    async def test_attaches_to_an_absent_system_message(self):
        built = _build()
        request = _request()
        with patch.object(mw, "_apply", wraps=mw._apply):
            await built.awrap_model_call(request, AsyncMock(return_value="answered"))
        request.override.assert_called_once()

    @pytest.mark.asyncio
    async def test_appends_to_an_existing_system_message(self):
        from langchain_core.messages import SystemMessage

        built = _build()
        request = _request(SystemMessage(content="You are an agent."))
        await built.awrap_model_call(request, AsyncMock())
        attached = request.override.call_args.kwargs["system_message"]
        assert attached.content.startswith("You are an agent.")
        assert "cannot do" in attached.content

    @pytest.mark.asyncio
    async def test_appends_a_block_to_structured_content(self):
        from langchain_core.messages import SystemMessage

        built = _build()
        request = _request(SystemMessage(content=[{"type": "text", "text": "base"}]))
        await built.awrap_model_call(request, AsyncMock())
        blocks = request.override.call_args.kwargs["system_message"].content
        assert blocks[-1]["text"].startswith("## What")

    @pytest.mark.asyncio
    async def test_an_unattachable_request_still_answers(self):
        # A missing notice costs the model some context; failing
        # here would cost the answer.
        built = _build()
        request = MagicMock()
        request.system_message = SimpleNamespace(content=object())
        request.override.side_effect = RuntimeError("cannot override")
        result = await built.awrap_model_call(request, AsyncMock(return_value="ok"))
        assert result == "ok"


class TestConnectionState:
    @pytest.mark.asyncio
    async def test_asks_for_connection_state_once_and_caches_it(self):
        state = AsyncMock(return_value=(["gitlab"], ["slack"]))
        with (
            patch("deep_agent.src.mcp_status.middleware.connection_state", new=state),
            patch(
                "deep_agent.src.self_knowledge.describe.build_self_description",
                return_value="notice",
            ),
        ):
            built = mw.build_self_knowledge_middleware("u1")
            await built.awrap_model_call(_request(), AsyncMock())
            await built.awrap_model_call(_request(), AsyncMock())
        assert state.await_count == 1

    @pytest.mark.asyncio
    async def test_no_identity_means_no_lookup(self):
        # Without a user the state is not knowable, and guessing
        # produces the exact wrong answer: "nothing is connected".
        state = AsyncMock()
        with (
            patch("deep_agent.src.mcp_status.middleware.connection_state", new=state),
            patch(
                "deep_agent.src.self_knowledge.describe.build_self_description",
                return_value="notice",
            ),
        ):
            built = mw.build_self_knowledge_middleware(None)
            await built.awrap_model_call(_request(), AsyncMock())
        state.assert_not_awaited()

    def test_sync_path_uses_what_the_async_path_cached(self):
        built = _build()
        request = _request()
        assert built.wrap_model_call(request, MagicMock(return_value="ok")) == "ok"


class TestOfferConnect:
    """A turn that declines must still offer the Connect button.

    This middleware is the one actually wired into the graph, and it
    tells the model up front which servers are unreachable — so the
    model declines without calling anything, nothing raises, and the
    failure-driven button never appeared.
    """

    @staticmethod
    def _response(text: str, extra: dict | None = None):
        message = MagicMock()
        message.text = text
        message.additional_kwargs = extra if extra is not None else {}
        response = MagicMock()
        response.result = [message]
        return response, message

    def test_stamps_the_server_the_answer_names(self):
        from deep_agent.src.self_knowledge.middleware import _offer_connect

        response, message = self._response("the warehouse server is not connected.")
        _offer_connect(response, ("warehouse", "miro"), "use warehouse for this")

        assert message.additional_kwargs["auth_required"] == [
            {"mcp_name": "warehouse", "reason": "not_connected"}
        ]

    def test_stamps_nothing_on_a_normal_answer(self):
        from deep_agent.src.self_knowledge.middleware import _offer_connect

        # With an `asked_for` that names the server. Left at its
        # default `""` nothing could ever match, so this passed
        # whatever the answer said.
        response, message = self._response("Your manager is Jane Roe.")
        _offer_connect(response, ("warehouse",), "use warehouse to find my manager")

        assert "auth_required" not in message.additional_kwargs

    def test_narrows_a_discovery_stamp_to_the_named_server(self):
        from deep_agent.src.self_knowledge.middleware import _offer_connect

        existing = {
            "auth_required": [
                {"mcp_name": "atlan", "reason": "not_connected"},
                {"mcp_name": "warehouse", "reason": "expired"},
            ]
        }
        response, message = self._response(
            "the warehouse server needs reconnecting.", existing
        )
        _offer_connect(response, ("atlan", "warehouse"), "use warehouse please")

        assert message.additional_kwargs["auth_required"] == [
            {"mcp_name": "warehouse", "reason": "expired"}
        ]

    def test_nothing_unconnected_is_a_no_op(self):
        from deep_agent.src.self_knowledge.middleware import _offer_connect

        response, message = self._response("the warehouse server is fine.")
        _offer_connect(response, (), "use warehouse")

        assert "auth_required" not in message.additional_kwargs


class TestOfferConnectNeedsAnAsk:
    """A Connect button answers "I could not do what you asked".

    Matching only the answer turned "which integrations can you use
    right now?" into four Connect buttons: the reply listed them, so
    every one looked like a blocked request.
    """

    @staticmethod
    def _response(text: str):
        message = MagicMock()
        message.text = text
        message.additional_kwargs = {}
        response = MagicMock()
        response.result = [message]
        return response, message

    def test_no_button_when_the_question_only_asked_about_integrations(self):
        from deep_agent.src.self_knowledge.middleware import _offer_connect

        response, message = self._response(
            "Connected: gworkspace. Not connected: atlassian, datadog, gitlab, miro."
        )
        _offer_connect(
            response,
            ("atlassian", "datadog", "gitlab", "miro"),
            "which integrations can you use right now?",
        )

        assert "auth_required" not in message.additional_kwargs

    def test_button_when_the_question_asked_to_use_one(self):
        from deep_agent.src.self_knowledge.middleware import _offer_connect

        response, message = self._response("Miro is not connected.")
        _offer_connect(
            response, ("miro", "datadog"), "use the miro mcp to list my boards"
        )

        assert message.additional_kwargs["auth_required"] == [
            {"mcp_name": "miro", "reason": "not_connected"}
        ]
