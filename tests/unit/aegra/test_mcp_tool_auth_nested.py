"""A subagent reports an authorization need; it does not ask for one."""

from __future__ import annotations

from unittest.mock import patch

from deep_agent.aegra import mcp_tool_auth as mta
from deep_agent.aegra.mcp_auth import NeedsAuthorization


class TestInNestedGraph:
    """LangGraph namespaces nested graphs with a non-empty checkpoint_ns."""

    def test_top_level_has_no_namespace(self):
        with patch("langgraph.config.get_config", return_value={"configurable": {}}):
            assert mta._in_nested_graph() is False

    def test_subagent_has_a_namespace(self):
        with patch(
            "langgraph.config.get_config",
            return_value={"configurable": {"checkpoint_ns": "task:abc123"}},
        ):
            assert mta._in_nested_graph() is True

    def test_unknown_counts_as_top_level(self):
        """Interrupting when we could have reported is recoverable.

        Reporting when we should have paused throws away a resumable
        run, so the uncertain case takes the recoverable side.
        """
        with patch("langgraph.config.get_config", side_effect=RuntimeError("no ctx")):
            assert mta._in_nested_graph() is False

    def test_no_config_at_all(self):
        with patch("langgraph.config.get_config", return_value=None):
            assert mta._in_nested_graph() is False


class TestAuthMessage:
    """What the orchestrator reads when a subagent cannot reach a server.

    `interrupt()` inside a subagent pauses a graph nobody is watching:
    it bubbles out of the `task` tool as an error, the parent run ends
    `interrupted`, and /runs/wait answers 200 with an empty body. That
    is what "it hangs on subagent calls" actually was.
    """

    def test_names_the_server_but_never_the_url(self):
        """The address is this deployment's own and the model copies it.

        Asked for a manager lookup with the warehouse server disconnected, the
        agent wrote `http://localhost:8081/mcp/warehouse/connect`
        into its answer — correct only on the machine that rendered
        it. The UI builds the button from the name alone.
        """
        exc = NeedsAuthorization("atlan", "https://agent.example/mcp/atlan/connect")
        msg = mta._mcp_auth_message(exc)

        assert "atlan" in msg
        assert "https://agent.example/mcp/atlan/connect" not in msg
        assert "http" not in msg

    def test_tells_the_model_not_to_retry(self):
        """Without this the subagent loops on a tool that cannot succeed."""
        exc = NeedsAuthorization("atlan", "https://agent.example/connect")
        assert "not retry" in mta._mcp_auth_message(exc).lower()

    def test_is_marked_so_the_orchestrator_can_recognise_it(self):
        exc = NeedsAuthorization("atlan", "https://agent.example/connect")
        # The marker carries the server and reason — enough for the
        # UI to render an action instead of prose.
        assert mta._mcp_auth_message(exc).startswith("[AUTH_REQUIRED mcp=atlan")


class TestAuthReason:
    """Expired and never-connected are different instructions.

    Both used to read "not connected", which sends someone to set up
    an integration they already set up.
    """

    def test_never_connected_says_so(self):
        exc = NeedsAuthorization("atlan", "https://x/connect")
        msg = mta._mcp_auth_message(exc)
        assert "reason=not_connected" in msg
        assert "not connected" in msg

    def test_expired_session_says_so(self):
        exc = NeedsAuthorization("gitlab", "https://x/connect", reason="expired")
        msg = mta._mcp_auth_message(exc)
        assert "reason=expired" in msg
        assert "expired" in msg

    def test_marker_is_machine_readable(self):
        """The UI parses this to offer a connect button in place of prose."""
        exc = NeedsAuthorization("miro", "https://x/connect")
        msg = mta._mcp_auth_message(exc)
        assert "[AUTH_REQUIRED mcp=miro" in msg
        assert "reason=not_connected]" in msg
        assert "url=" not in msg

    def test_interrupt_payload_carries_the_reason_too(self):
        import json

        exc = NeedsAuthorization("atlan", "https://x/connect", reason="expired")
        assert json.loads(mta._mcp_auth_interrupt_payload(exc))["reason"] == "expired"


class TestNarrowAuthRequests:
    """Five Connect buttons under an answer about one integration.

    Discovery records every unconnected server, so the reply to
    "who is my manager, use the warehouse server" carried buttons for Atlan,
    Datadog, the warehouse server, Miro and unstructured-data. Picking yours out
    of that is the work the button was supposed to remove.
    """

    REQUESTS = [
        {"mcp_name": "atlan", "reason": "not_connected"},
        {"mcp_name": "warehouse", "reason": "expired"},
        {"mcp_name": "miro", "reason": "not_connected"},
    ]

    def test_keeps_only_what_the_answer_names(self):
        out = mta.narrow_auth_requests(
            self.REQUESTS,
            "the warehouse server is not connected, so I could not look that up.",
        )
        assert out == [{"mcp_name": "warehouse", "reason": "expired"}]

    def test_matches_regardless_of_case(self):
        out = mta.narrow_auth_requests(self.REQUESTS, "connect WAREHOUSE to continue")
        assert [r["mcp_name"] for r in out] == ["warehouse"]

    def test_offers_nothing_when_the_answer_attributes_nothing(self):
        """Seven buttons, none of them the one the sentence is about.

        Connect the warehouse server and ask again: the warehouse server drops out of the
        recorded list, nothing matches, and keeping the remainder
        offered every other integration instead. No button is the
        better error.
        """
        out = mta.narrow_auth_requests(self.REQUESTS, "Something went wrong.")
        assert out == []

    def test_keeps_several_when_several_are_named(self):
        out = mta.narrow_auth_requests(
            self.REQUESTS, "Both the warehouse server and Miro need connecting."
        )
        assert [r["mcp_name"] for r in out] == ["warehouse", "miro"]

    def test_empty_input_is_unchanged(self):
        assert mta.narrow_auth_requests([], "the warehouse server") == []


class TestAuthPromptCooldown:
    """Ask once, then inform — do not nag.

    The agent now calls an unconnected integration deliberately so
    the run pauses and the user can authorize it. For someone who
    does not want that integration, that meant a blocking prompt in
    every new chat, for ever.
    """

    @staticmethod
    def _patch_redis(store: dict):
        from unittest.mock import AsyncMock

        from deep_agent.aegra import redis as rd

        async def fake_run_redis(fn, *args, **kwargs):
            if fn is rd.cache_get:
                return store.get(args[0])
            if fn is rd.cache_set:
                store[args[0]] = args[1]
                return True
            if fn is rd.cache_delete:
                store.pop(args[0], None)
                return True
            return None

        del AsyncMock
        return patch("deep_agent.aegra.redis.run_redis", fake_run_redis)

    async def test_the_first_ask_is_remembered(self):
        store: dict = {}
        with (
            self._patch_redis(store),
            patch("deep_agent.aegra.mcp._current_user_id") as uid,
            patch.object(mta.settings, "MCP_AUTH_PROMPT_COOLDOWN_SECONDS", 3600),
        ):
            uid.get.return_value = "user-1"
            assert await mta._already_asked("miro") is False
            await mta._remember_asked("miro")
            assert await mta._already_asked("miro") is True

    async def test_each_integration_is_tracked_separately(self):
        store: dict = {}
        with (
            self._patch_redis(store),
            patch("deep_agent.aegra.mcp._current_user_id") as uid,
            patch.object(mta.settings, "MCP_AUTH_PROMPT_COOLDOWN_SECONDS", 3600),
        ):
            uid.get.return_value = "user-1"
            await mta._remember_asked("miro")
            assert await mta._already_asked("atlan") is False

    async def test_connecting_clears_it(self):
        store: dict = {}
        with (
            self._patch_redis(store),
            patch("deep_agent.aegra.mcp._current_user_id") as uid,
            patch.object(mta.settings, "MCP_AUTH_PROMPT_COOLDOWN_SECONDS", 3600),
        ):
            uid.get.return_value = "user-1"
            await mta._remember_asked("miro")
            await mta.forget_auth_prompt("user-1", "miro")
            assert await mta._already_asked("miro") is False

    async def test_a_zero_cooldown_always_asks(self):
        store: dict = {}
        with (
            self._patch_redis(store),
            patch("deep_agent.aegra.mcp._current_user_id") as uid,
            patch.object(mta.settings, "MCP_AUTH_PROMPT_COOLDOWN_SECONDS", 0),
        ):
            uid.get.return_value = "user-1"
            await mta._remember_asked("miro")
            assert await mta._already_asked("miro") is False

    async def test_an_unreachable_cache_still_asks(self):
        async def boom(fn, *args, **kwargs):
            raise RuntimeError("redis down")

        with (
            patch("deep_agent.aegra.redis.run_redis", boom),
            patch("deep_agent.aegra.mcp._current_user_id") as uid,
            patch.object(mta.settings, "MCP_AUTH_PROMPT_COOLDOWN_SECONDS", 3600),
        ):
            uid.get.return_value = "user-1"
            # A prompt the user can act on beats silence.
            assert await mta._already_asked("miro") is False


class TestCooldownFollowsTheAsk:
    """Record "do not ask again" on a decline, never on the first ask.

    The cooldown used to be written just before `interrupt()`. By the
    time the pause reached the reader the flag was already set, so the
    next turn took the "already asked" branch and answered in prose.
    The interrupt fired exactly once, ever, and every later attempt
    looked as though the pause had never been built.
    """

    @staticmethod
    def _tool(side_effects):
        """A StructuredTool whose coroutine yields `side_effects` in turn."""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        class _In(BaseModel):
            pass

        calls = iter(side_effects)

        async def _coro(**kwargs):
            outcome = next(calls)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return StructuredTool(
            name="mcp__warehouse", coroutine=_coro, args_schema=_In, func=lambda: ""
        )

    @staticmethod
    def _patched(remembered: list[str], asked: set[str]):
        async def _already(mcp_name: str) -> bool:
            return mcp_name in asked

        async def _remember(mcp_name: str) -> None:
            remembered.append(mcp_name)

        return (
            patch.object(mta, "_already_asked", _already),
            patch.object(mta, "_remember_asked", _remember),
            patch.object(mta, "_in_nested_graph", lambda: False),
        )

    async def test_the_first_ask_pauses_and_records_nothing(self):
        from langgraph.errors import GraphInterrupt

        exc = NeedsAuthorization("warehouse", "https://x/connect")
        tool = self._tool([exc])
        remembered: list[str] = []
        a, b, c = self._patched(remembered, set())

        with a, b, c, patch.object(mta, "interrupt", side_effect=GraphInterrupt()):
            wrapped = mta._wrap_single_tool(tool)
            with __import__("pytest").raises(GraphInterrupt):
                await wrapped.coroutine()

        assert remembered == []

    async def test_connecting_during_the_pause_just_works(self):
        """`interrupt()` returns on replay; the retry then succeeds."""
        exc = NeedsAuthorization("warehouse", "https://x/connect")
        tool = self._tool([exc, "the answer"])
        remembered: list[str] = []
        a, b, c = self._patched(remembered, set())

        with a, b, c, patch.object(mta, "interrupt", return_value=None):
            wrapped = mta._wrap_single_tool(tool)
            assert await wrapped.coroutine() == "the answer"

        assert remembered == []

    async def test_declining_records_the_cooldown_once(self):
        """Came back from the pause still unconnected: stop asking."""
        exc = NeedsAuthorization("warehouse", "https://x/connect")
        tool = self._tool([exc, exc])
        remembered: list[str] = []
        a, b, c = self._patched(remembered, set())

        with a, b, c, patch.object(mta, "interrupt", return_value=None):
            wrapped = mta._wrap_single_tool(tool)
            out = await wrapped.coroutine()

        assert remembered == ["warehouse"]
        assert out.startswith("[AUTH_REQUIRED mcp=warehouse")

    async def test_a_standing_cooldown_reports_without_pausing(self):
        exc = NeedsAuthorization("warehouse", "https://x/connect")
        tool = self._tool([exc])
        remembered: list[str] = []
        a, b, c = self._patched(remembered, {"warehouse"})

        with a, b, c, patch.object(mta, "interrupt") as mock_interrupt:
            wrapped = mta._wrap_single_tool(tool)
            out = await wrapped.coroutine()

        mock_interrupt.assert_not_called()
        assert remembered == []
        assert out.startswith("[AUTH_REQUIRED mcp=warehouse")
