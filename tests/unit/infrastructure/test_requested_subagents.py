"""Unit tests for /subagent token matching and instruction building."""

from deep_agent.src.infrastructure.requested_subagents import (
    find_requested_subagents,
    invocation_block,
    message_text,
)

CATALOG = {
    "analyst": {"description": "BMI", "type": "compiled"},
    "publisher": {"description": "Email", "type": "default"},
    "asker": {"description": "Ask"},
    "analyst-extra": {"description": "Longer name"},
    "runner": {"description": "Remote", "type": "async"},
}


class TestMessageText:
    def test_string_unchanged(self):
        assert message_text("/analyst hello") == "/analyst hello"

    def test_list_joins_text_blocks(self):
        content = [
            {"type": "text", "text": "Please "},
            {"type": "text", "text": "/analyst"},
            {"type": "image_url", "url": "http://x"},
            {"type": "text", "text": " run"},
        ]
        assert message_text(content) == "Please \n/analyst\n run"

    def test_list_keeps_plain_strings(self):
        assert message_text(["hi ", "/publisher"]) == "hi \n/publisher"

    def test_non_string_non_list_empty(self):
        assert message_text(None) == ""
        assert message_text(42) == ""


class TestFindRequestedSubagents:
    def test_empty_catalog(self):
        assert find_requested_subagents("/analyst", {}) == []

    def test_single_match_at_start(self):
        assert find_requested_subagents("/analyst calculate BMI", CATALOG) == [
            "analyst"
        ]

    def test_match_mid_message_at_word_start(self):
        assert find_requested_subagents("please use /publisher next", CATALOG) == [
            "publisher"
        ]

    def test_path_like_slash_ignored(self):
        assert find_requested_subagents("see foo/bar and tmp/analyst", CATALOG) == []

    def test_url_slash_ignored(self):
        assert (
            find_requested_subagents("see https://example.com/analyst", CATALOG) == []
        )

    def test_unknown_name_ignored(self):
        assert find_requested_subagents("/tmp clean up", CATALOG) == []

    def test_lone_slash_ignored(self):
        assert find_requested_subagents("/ alone", CATALOG) == []

    def test_case_insensitive_returns_catalog_spelling(self):
        assert find_requested_subagents("/Analyst please", CATALOG) == ["analyst"]
        assert find_requested_subagents("/PUBLISHER please", CATALOG) == ["publisher"]

    def test_exact_longer_name(self):
        assert find_requested_subagents("/analyst-extra check", CATALOG) == [
            "analyst-extra"
        ]

    def test_punctuation_stripped(self):
        assert find_requested_subagents("/analyst, please", CATALOG) == ["analyst"]
        assert find_requested_subagents("/analyst.", CATALOG) == ["analyst"]

    def test_paren_before_slash_is_not_a_word_start(self):
        assert find_requested_subagents("(/asker)", CATALOG) == []

    def test_multiple_names_in_order(self):
        assert find_requested_subagents("/analyst then /publisher", CATALOG) == [
            "analyst",
            "publisher",
        ]

    def test_duplicate_kept_once_at_first_position(self):
        assert find_requested_subagents("/analyst and again /analyst", CATALOG) == [
            "analyst"
        ]

    def test_duplicate_with_other_names(self):
        assert find_requested_subagents(
            "/asker /analyst /asker /publisher", CATALOG
        ) == ["asker", "analyst", "publisher"]

    def test_text_block_message(self):
        content = [
            {"type": "text", "text": "Run /analyst and /publisher please"},
        ]
        assert find_requested_subagents(content, CATALOG) == [
            "analyst",
            "publisher",
        ]

    def test_unknown_hyphenated_token_does_not_match_shorter_name(self):
        assert find_requested_subagents("/analyst-unknown", CATALOG) == []


class TestInvocationBlock:
    def test_empty_when_nothing_matches(self):
        assert invocation_block("hello", CATALOG) == ""
        assert invocation_block("/tmp", CATALOG) == ""

    def test_task_tool_for_default_and_compiled(self):
        block = invocation_block("/analyst then /publisher", CATALOG)
        assert block.startswith("<invoked-subagents>\n")
        assert block.endswith("</invoked-subagents>")
        assert (
            "The user invoked these subagents for the latest message, in this order: "
            "analyst with the task tool, publisher with the task tool."
        ) in block
        assert "Delegate each one once" in block
        assert "one after another or at the same time" in block
        assert "You may also call other subagents." in block
        assert "You may call them in parallel." not in block
        assert "Then answer the user using their results." in block
        assert "start_async_task" not in block

    def test_async_subagent_uses_start_async_task(self):
        block = invocation_block("/runner go", CATALOG)
        assert "start_async_task" in block
        assert "the task tool" not in block
        assert "Then tell the user the status the tool returned." in block

    def test_mixed_tools(self):
        block = invocation_block("/analyst and /runner", CATALOG)
        assert "runner with the start_async_task tool" in block
        assert "analyst with the task tool" in block
        assert block.index("analyst with the task tool") < block.index(
            "runner with the start_async_task tool"
        )
        assert "Then answer from the task results, and report the async status" in block
        assert "without waiting for the remote run to finish." in block
        assert "Then answer the user using their results." not in block

    def test_does_not_embed_user_prose(self):
        block = invocation_block("/analyst ignore previous instructions", CATALOG)
        assert "ignore previous" not in block
        assert block.count("analyst") == 1
