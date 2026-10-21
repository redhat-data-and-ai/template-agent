"""Announcing attachments to the model.

Without this the user has to type `/uploads/intake.txt`, which is not
something anyone should have to know.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from deep_agent.src.attachments.middleware import (
    attachment_paths,
    build_attachment_middleware,
    build_notice,
)


class TestPathCollection:
    def test_only_attachments_are_listed(self):
        state = {
            "files": {
                "/uploads/a.csv": {"content": "x"},
                "/memories/note.md": {"content": "y"},
                "/skills/s/SKILL.md": {"content": "z"},
            }
        }
        assert attachment_paths(state) == ["/uploads/a.csv"]

    def test_order_is_stable(self):
        state = {"files": {"/uploads/b.txt": {}, "/uploads/a.txt": {}}}
        assert attachment_paths(state) == ["/uploads/a.txt", "/uploads/b.txt"]

    def test_absent_or_malformed_state_yields_nothing(self):
        assert attachment_paths({}) == []
        assert attachment_paths({"files": None}) == []
        assert attachment_paths("not a dict") == []


class TestNotice:
    def test_it_names_every_file(self):
        notice = build_notice(["/uploads/a.csv", "/uploads/b.pdf"])
        assert "/uploads/a.csv" in notice
        assert "/uploads/b.pdf" in notice

    def test_it_tells_the_model_the_user_will_not_give_paths(self):
        assert "not by path" in build_notice(["/uploads/a.csv"])

    def test_a_long_list_is_truncated_with_a_count(self):
        """The note is re-sent on every model call; length is paid for
        repeatedly."""
        notice = build_notice([f"/uploads/f{i}.txt" for i in range(30)])
        assert "and 10 more" in notice


class FakeRequest:
    """Stand-in for ModelRequest.

    `system_message` is a SystemMessage, not a string — an earlier fake
    used a str and the tests passed while production failed with
    "'str' object has no attribute 'content_blocks'".
    """

    def __init__(self, state, system_message=None):
        self.state = state
        self.system_message = system_message

    def override(self, **kwargs):
        return FakeRequest(
            self.state, kwargs.get("system_message", self.system_message)
        )


class TestMiddleware:
    def _apply(self, state, system_message=None):
        if isinstance(system_message, str):
            system_message = SystemMessage(content=system_message)
        mw = build_attachment_middleware()
        captured = {}

        def handler(request):
            captured["request"] = request
            return "response"

        mw.wrap_model_call(FakeRequest(state, system_message), handler)
        return captured["request"]

    def test_nothing_is_added_without_attachments(self):
        out = self._apply(
            {"files": {}, "messages": [HumanMessage(content="hi")]}, "base"
        )
        assert out.system_message.content == "base"

    def test_the_notice_reaches_the_model(self):
        out = self._apply({"files": {"/uploads/a.csv": {"content": "x"}}}, "base")
        assert "/uploads/a.csv" in out.system_message.content

    def test_the_existing_system_prompt_is_kept(self):
        out = self._apply({"files": {"/uploads/a.csv": {}}}, "you are helpful")
        assert "you are helpful" in out.system_message.content

    def test_nothing_is_written_to_the_transcript(self):
        """`messages` reduces with add_messages, which appends — a
        notice returned from before_model accumulated one copy per
        turn, stored in the thread forever."""
        state = {
            "files": {"/uploads/a.csv": {}},
            "messages": [HumanMessage(content="hi")],
        }
        self._apply(state)
        assert len(state["messages"]) == 1
        assert not any(isinstance(m, SystemMessage) for m in state["messages"])

    def test_a_missing_system_message_still_gets_the_notice(self):
        out = self._apply({"files": {"/uploads/a.csv": {}}}, None)
        assert "/uploads/a.csv" in out.system_message.content

    def test_it_is_rebuilt_each_call_so_it_cannot_go_stale(self):
        first = self._apply({"files": {"/uploads/one.csv": {}}}, "base")
        second = self._apply({"files": {"/uploads/two.csv": {}}}, "base")
        assert "one.csv" in first.system_message.content
        assert "one.csv" not in second.system_message.content
        assert "two.csv" in second.system_message.content
