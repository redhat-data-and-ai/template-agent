"""Unit tests for the clarify question tool."""

from deep_agent.src.agent.clarify import attach_clarify, clarify

_EMPTY_QUESTION = "No question was asked. Call clarify again with a non-empty question."
_NO_ANSWER = "The user did not provide an answer."
_NON_INTERACTIVE = "No preference. Continue with the most reasonable choice."


def _interactive_config():
    return {"configurable": {"interactive": True}}


class TestClarify:
    def test_real_answer_pauses_with_cleaned_options(self, monkeypatch):
        seen: dict = {}

        def fake_interrupt(payload):
            seen["payload"] = payload
            return "  kg  "

        monkeypatch.setattr("deep_agent.src.agent.clarify.interrupt", fake_interrupt)
        monkeypatch.setattr(
            "deep_agent.src.agent.clarify._is_interactive", lambda: True
        )

        result = clarify.invoke(
            {"question": " Units? ", "options": ["kg", "  ", "lbs"]}
        )

        assert result == "kg"
        assert seen["payload"] == {
            "type": "user_question",
            "question": "Units?",
            "options": ["kg", "lbs"],
        }

    def test_blank_question_does_not_pause(self, monkeypatch):
        def fake_interrupt(payload):
            raise AssertionError(f"blank question must not pause: {payload}")

        monkeypatch.setattr("deep_agent.src.agent.clarify.interrupt", fake_interrupt)
        monkeypatch.setattr(
            "deep_agent.src.agent.clarify._is_interactive", lambda: True
        )

        assert clarify.invoke({"question": "   "}) == _EMPTY_QUESTION

    def test_blank_options_are_dropped(self, monkeypatch):
        seen: dict = {}

        def fake_interrupt(payload):
            seen["payload"] = payload
            return "stone"

        monkeypatch.setattr("deep_agent.src.agent.clarify.interrupt", fake_interrupt)
        monkeypatch.setattr(
            "deep_agent.src.agent.clarify._is_interactive", lambda: True
        )

        result = clarify.invoke({"question": "Units?", "options": ["  ", ""]})

        assert result == "stone"
        assert seen["payload"]["options"] == []

    def test_non_text_answer_is_reported(self, monkeypatch):
        monkeypatch.setattr(
            "deep_agent.src.agent.clarify.interrupt",
            lambda payload: {"answer": "kg"},
        )
        monkeypatch.setattr(
            "deep_agent.src.agent.clarify._is_interactive", lambda: True
        )

        assert clarify.invoke({"question": "Units?"}) == _NO_ANSWER

    def test_non_interactive_run_does_not_pause(self, monkeypatch):
        def fake_interrupt(payload):
            raise AssertionError(f"headless must not pause: {payload}")

        monkeypatch.setattr("deep_agent.src.agent.clarify.interrupt", fake_interrupt)
        monkeypatch.setattr(
            "deep_agent.src.agent.clarify._is_interactive", lambda: False
        )

        assert clarify.invoke({"question": "Color?"}) == _NON_INTERACTIVE

    def test_is_interactive_reads_configurable_flag(self, monkeypatch):
        import langgraph.config as lg_config

        from deep_agent.src.agent.clarify import _is_interactive

        monkeypatch.setattr(lg_config, "get_config", lambda: _interactive_config())
        assert _is_interactive() is True

        monkeypatch.setattr(lg_config, "get_config", lambda: {"configurable": {}})
        assert _is_interactive() is False

        monkeypatch.setattr(lg_config, "get_config", lambda: None)
        assert _is_interactive() is False

    def test_description_says_when_to_ask_and_when_to_omit_options(self):
        description = clarify.description
        assert "clarify" in description
        assert "omit options" in description
        assert "complete answer" in description
        assert "any number" in description

    def test_attach_adds_the_tool_once(self):
        once = attach_clarify([])
        twice = attach_clarify(once)

        assert [tool.name for tool in once] == ["clarify"]
        assert [tool.name for tool in twice] == ["clarify"]
        assert twice[0] is clarify
