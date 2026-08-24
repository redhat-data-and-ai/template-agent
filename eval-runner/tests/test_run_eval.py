"""Tests for run_eval.py — SSE parsing, agent interaction helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import run_eval

# ── _parse_sse_stream ─────────────────────────────────────────────────────────


def test_parse_sse_stream_single_event() -> None:
    lines = ["event: updates", 'data: {"key": "value"}', ""]
    events = run_eval._parse_sse_stream(lines)
    assert len(events) == 1
    assert events[0] == ("updates", {"key": "value"})


def test_parse_sse_stream_multiple_events() -> None:
    lines = [
        "event: updates",
        'data: {"a": 1}',
        "",
        "event: events",
        'data: {"b": 2}',
        "",
    ]
    events = run_eval._parse_sse_stream(lines)
    assert len(events) == 2
    assert events[0] == ("updates", {"a": 1})
    assert events[1] == ("events", {"b": 2})


def test_parse_sse_stream_invalid_json_skipped() -> None:
    lines = ["event: updates", "data: not-json", ""]
    events = run_eval._parse_sse_stream(lines)
    assert events == []


def test_parse_sse_stream_default_event_type_is_message() -> None:
    lines = ['data: {"x": 1}', ""]
    events = run_eval._parse_sse_stream(lines)
    assert events[0][0] == "message"


def test_parse_sse_stream_trailing_data_without_blank_line() -> None:
    lines = ['data: {"z": 9}']  # no trailing blank line
    events = run_eval._parse_sse_stream(lines)
    assert len(events) == 1


def test_parse_sse_stream_empty_input() -> None:
    assert run_eval._parse_sse_stream([]) == []


def test_parse_sse_stream_trailing_invalid_json() -> None:
    lines = ["event: updates", "data: not-json"]  # no blank line, bad JSON
    assert run_eval._parse_sse_stream(lines) == []


def test_parse_sse_stream_blank_line_without_data() -> None:
    lines = ["event: updates", ""]
    assert run_eval._parse_sse_stream(lines) == []


# ── _extract_text ─────────────────────────────────────────────────────────────


def test_extract_text_plain_string() -> None:
    assert run_eval._extract_text("hello world") == "hello world"


def test_extract_text_list_of_text_blocks() -> None:
    content = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert run_eval._extract_text(content) == "first\nsecond"


def test_extract_text_list_skips_non_text_blocks() -> None:
    content = [
        {"type": "image", "url": "http://x.com/img.png"},
        {"type": "text", "text": "only this"},
    ]
    result = run_eval._extract_text(content)
    assert "only this" in result


def test_extract_text_empty_list() -> None:
    assert run_eval._extract_text([]) == ""


def test_extract_text_unrecognised_type() -> None:
    assert run_eval._extract_text(42) == ""


# ── _resolve_tool_name ────────────────────────────────────────────────────────


def test_resolve_tool_name_direct() -> None:
    tc = {"name": "calculate_bmi", "args": {}}
    assert run_eval._resolve_tool_name(tc) == "calculate_bmi"


def test_resolve_tool_name_task_subagent() -> None:
    tc = {"name": "task", "args": {"subagent_type": "analyst"}}
    assert run_eval._resolve_tool_name(tc) == "analyst"


def test_resolve_tool_name_task_without_subagent_type() -> None:
    tc = {"name": "task", "args": {}}
    assert run_eval._resolve_tool_name(tc) == "task"


# ── _unwrap_update_data ───────────────────────────────────────────────────────


def test_unwrap_subgraph_list_format() -> None:
    data = [["ns1", "ns2"], {"agent": {"messages": []}}]
    assert run_eval._unwrap_update_data(data) == {"agent": {"messages": []}}


def test_unwrap_dict_passthrough() -> None:
    data: dict[str, Any] = {"agent": {"messages": []}}
    assert run_eval._unwrap_update_data(data) == data


def test_unwrap_other_returns_empty() -> None:
    assert run_eval._unwrap_update_data("bad_input") == {}
    assert run_eval._unwrap_update_data(None) == {}


# ── _extract_node_updates ─────────────────────────────────────────────────────


def test_extract_node_updates_top_level_dict() -> None:
    data = {"agent": {"messages": [{"type": "ai", "content": "hi"}]}}
    updates = list(run_eval._extract_node_updates(data))  # iterable now
    assert len(updates) == 1
    assert updates[0]["messages"][0]["content"] == "hi"


def test_extract_node_updates_subgraph_list_format() -> None:
    data = [["ns1", "ns2"], {"agent": {"messages": []}}]
    updates = list(run_eval._extract_node_updates(data))  # iterable now
    assert len(updates) == 1
    assert updates[0] == {"messages": []}


def test_extract_node_updates_invalid_returns_empty() -> None:
    assert list(run_eval._extract_node_updates("bad")) == []
    assert list(run_eval._extract_node_updates([])) == []


# ── _has_interrupt ────────────────────────────────────────────────────────────


def test_has_interrupt_true() -> None:
    events: list[tuple[str, Any]] = [
        ("updates", {"__interrupt__": [{"value": {"action_requests": []}}]})
    ]
    assert run_eval._has_interrupt(events) is True


def test_has_interrupt_false() -> None:
    events: list[tuple[str, Any]] = [("updates", {"agent": {"messages": []}})]
    assert run_eval._has_interrupt(events) is False


def test_has_interrupt_empty_events() -> None:
    assert run_eval._has_interrupt([]) is False


def test_has_interrupt_subgraph_list_format() -> None:
    events: list[tuple[str, Any]] = [
        ("updates", [["ns"], {"__interrupt__": [{"value": {}}]}])
    ]
    assert run_eval._has_interrupt(events) is True


def test_has_interrupt_skips_non_updates_events() -> None:
    events: list[tuple[str, Any]] = [
        ("events", {"event": "on_tool_start"}),
        ("updates", {"agent": {}}),
    ]
    assert run_eval._has_interrupt(events) is False


# ── _count_interrupted_tool_calls ─────────────────────────────────────────────


def test_count_interrupted_tool_calls_two() -> None:
    events = [
        ("updates", {"__interrupt__": [{"value": {"action_requests": ["a", "b"]}}]})
    ]
    assert run_eval._count_interrupted_tool_calls(events) == 2


def test_count_interrupted_tool_calls_defaults_to_one() -> None:
    events: list[tuple[str, Any]] = [("updates", {"__interrupt__": [{"value": {}}]})]
    assert run_eval._count_interrupted_tool_calls(events) == 1


def test_count_interrupted_tool_calls_no_interrupt() -> None:
    events: list[tuple[str, Any]] = [("updates", {"agent": {}})]
    assert run_eval._count_interrupted_tool_calls(events) == 1


def test_count_interrupted_tool_calls_skips_non_updates() -> None:
    events = [
        ("events", {"event": "on_tool_start"}),
        ("updates", {"__interrupt__": [{"value": {"action_requests": ["x"]}}]}),
    ]
    assert run_eval._count_interrupted_tool_calls(events) == 1


# ── _last_nonempty ────────────────────────────────────────────────────────────


def test_last_nonempty_returns_last_non_blank() -> None:
    assert run_eval._last_nonempty(["", "first", "last", ""]) == "last"


def test_last_nonempty_all_empty() -> None:
    assert run_eval._last_nonempty(["", "   ", ""]) == ""


def test_last_nonempty_empty_list() -> None:
    assert run_eval._last_nonempty([]) == ""


def test_last_nonempty_single_entry() -> None:
    assert run_eval._last_nonempty(["only"]) == "only"


# ── _headers ──────────────────────────────────────────────────────────────────


def test_headers_with_token() -> None:
    h = run_eval._headers("tok123")
    assert h["Authorization"] == "Bearer tok123"
    assert h["Content-Type"] == "application/json"


def test_headers_without_token() -> None:
    h = run_eval._headers(None)
    assert "Authorization" not in h
    assert h["Content-Type"] == "application/json"


def test_headers_with_empty_string() -> None:
    h = run_eval._headers("")
    assert "Authorization" not in h


# ── _collect_from_events ──────────────────────────────────────────────────────


def test_collect_from_events_ai_message() -> None:
    events = [
        (
            "updates",
            {
                "agent": {
                    "messages": [
                        {"type": "ai", "content": "BMI is 22.9", "tool_calls": []}
                    ]
                }
            },
        )
    ]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert "BMI is 22.9" in texts


def test_collect_from_events_tool_start_event() -> None:
    events = [
        (
            "events",
            {
                "event": "on_tool_start",
                "name": "calculate_bmi",
                "run_id": "r1",
                "data": {"input": {"weight": 70}},
            },
        )
    ]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert any(tc["tool_name"] == "calculate_bmi" for tc in tcs)


def test_collect_from_events_internal_tools_excluded() -> None:
    events = [
        (
            "events",
            {
                "event": "on_tool_start",
                "name": "write_todos",
                "run_id": "r2",
                "data": {"input": {}},
            },
        )
    ]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert tcs == []


def test_collect_from_events_tool_result_captured_as_context() -> None:
    events = [
        (
            "updates",
            {"agent": {"messages": [{"type": "tool", "content": "BMI result: 22.9"}]}},
        )
    ]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert "BMI result: 22.9" in ctxs


def test_collect_from_events_tool_message_with_empty_content() -> None:
    events = [("updates", {"agent": {"messages": [{"type": "tool", "content": ""}]}})]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert ctxs == []


def test_collect_from_events_deduplicates_tool_calls_by_run_id() -> None:
    events = [
        (
            "events",
            {
                "event": "on_tool_start",
                "name": "calc",
                "run_id": "dup",
                "data": {"input": {}},
            },
        ),
        (
            "events",
            {
                "event": "on_tool_start",
                "name": "calc",
                "run_id": "dup",
                "data": {"input": {}},
            },
        ),
    ]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert len(tcs) == 1


def test_collect_from_events_empty() -> None:
    texts, tcs, ctxs, _abc = run_eval._collect_from_events([])
    assert texts == []
    assert tcs == []
    assert ctxs == []


def test_collect_from_events_ai_tool_calls_from_updates() -> None:
    events = [
        (
            "updates",
            {
                "agent": {
                    "messages": [
                        {
                            "type": "ai",
                            "content": "",
                            "tool_calls": [
                                {"name": "calculate_bmi", "args": {"weight": 70}}
                            ],
                        }
                    ]
                }
            },
        )
    ]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert any(tc["tool_name"] == "calculate_bmi" for tc in tcs)


def test_collect_from_events_stream_mode_list_wrapper() -> None:
    events = [
        (
            "updates",
            [
                "updates",
                {
                    "agent": {
                        "messages": [
                            {"type": "ai", "content": "wrapped", "tool_calls": []}
                        ]
                    }
                },
            ],
        )
    ]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert "wrapped" in texts


def test_collect_from_events_skips_invalid_update_and_message() -> None:
    events = [
        ("updates", {"agent": "not-a-dict", "other": [1, 2]}),
        (
            "updates",
            {"node": {"messages": ["not-a-dict", {"type": "tool", "content": ""}]}},
        ),
    ]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert texts == []
    assert tcs == []
    assert ctxs == []


def test_collect_from_events_excludes_internal_ai_tool_calls() -> None:
    events = [
        (
            "updates",
            {
                "agent": {
                    "messages": [
                        {
                            "type": "ai",
                            "content": "",
                            "tool_calls": [{"name": "write_todos", "args": {}}],
                        }
                    ]
                }
            },
        )
    ]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert tcs == []


def test_collect_from_events_skips_duplicate_run_id() -> None:
    events = [
        (
            "events",
            {
                "event": "on_tool_start",
                "name": "calc",
                "run_id": "seen",
                "data": {"input": {}},
            },
        ),
        (
            "events",
            {
                "event": "on_tool_start",
                "name": "other",
                "run_id": "seen",
                "data": {"input": {}},
            },
        ),
    ]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert len(tcs) == 1


def test_collect_from_events_non_tool_start_event_ignored() -> None:
    events = [("events", {"event": "on_chain_start", "name": "agent"})]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert tcs == []


def test_collect_from_events_tool_start_without_run_id() -> None:
    events = [
        (
            "events",
            {
                "event": "on_tool_start",
                "name": "calc",
                "run_id": "",
                "data": {"input": {"x": 1}},
            },
        )
    ]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert len(tcs) == 1


def test_collect_from_events_tool_start_non_dict_args() -> None:
    events = [
        (
            "events",
            {
                "event": "on_tool_start",
                "name": "calc",
                "run_id": "r3",
                "data": {"input": "not-a-dict"},
            },
        )
    ]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert tcs == []


def test_collect_from_events_ignores_unknown_event_type() -> None:
    events = [("message", {"data": "ignored"})]
    texts, tcs, ctxs, _abc = run_eval._collect_from_events(events)
    assert texts == []
    assert tcs == []
    assert ctxs == []


# ── _collect_from_events: on_tool_end and ai_before_ctx ─────────────────────


def test_collect_from_events_on_tool_end_plain_string() -> None:
    """on_tool_end with plain string output → added to contexts."""
    events = [
        (
            "events",
            {
                "event": "on_tool_end",
                "name": "calculate_bmi",
                "run_id": "r1",
                "data": {"output": '{"bmi": 22.9}'},
            },
        )
    ]
    texts, tcs, ctxs, ai_before_ctx = run_eval._collect_from_events(events)
    assert '{"bmi": 22.9}' in ctxs
    assert ai_before_ctx is False  # no AI text seen


def test_collect_from_events_on_tool_end_dict_with_content_list() -> None:
    """on_tool_end with LangChain ToolMessage dict → extracts text content."""
    output = {"content": [{"type": "text", "text": "BMI is 22.9"}], "type": "tool"}
    events = [
        (
            "events",
            {
                "event": "on_tool_end",
                "name": "calculate_bmi",
                "run_id": "r1",
                "data": {"output": output},
            },
        )
    ]
    texts, tcs, ctxs, ai_before_ctx = run_eval._collect_from_events(events)
    assert "BMI is 22.9" in ctxs


def test_collect_from_events_on_tool_end_list_output() -> None:
    """on_tool_end with list of content blocks → extracts text items."""
    output = [{"type": "text", "text": "result text"}, {"type": "image"}]
    events = [
        (
            "events",
            {
                "event": "on_tool_end",
                "name": "search_web",
                "run_id": "r1",
                "data": {"output": output},
            },
        )
    ]
    texts, tcs, ctxs, ai_before_ctx = run_eval._collect_from_events(events)
    assert "result text" in ctxs


def test_collect_from_events_on_tool_end_internal_tool_excluded() -> None:
    """on_tool_end for internal tools → not added to contexts."""
    events = [
        (
            "events",
            {
                "event": "on_tool_end",
                "name": "write_todos",
                "run_id": "r1",
                "data": {"output": "Updated todo list"},
            },
        )
    ]
    texts, tcs, ctxs, _ = run_eval._collect_from_events(events)
    assert ctxs == []


def test_collect_from_events_ai_before_ctx_true() -> None:
    """AI text before context → ai_before_ctx True (delegation pattern)."""
    events = [
        ("updates", {"agent": {"messages": [{"type": "ai", "content": "Welcome!"}]}}),
        (
            "events",
            {
                "event": "on_tool_end",
                "name": "calculate_bmi",
                "run_id": "r1",
                "data": {"output": '{"bmi": 22.9}'},
            },
        ),
    ]
    texts, tcs, ctxs, ai_before_ctx = run_eval._collect_from_events(events)
    assert ai_before_ctx is True


def test_collect_from_events_ai_after_ctx_false() -> None:
    """Context before AI text → ai_before_ctx False (real response)."""
    events = [
        (
            "events",
            {
                "event": "on_tool_end",
                "name": "calculate_bmi",
                "run_id": "r1",
                "data": {"output": '{"bmi": 22.9}'},
            },
        ),
        (
            "updates",
            {"agent": {"messages": [{"type": "ai", "content": "Your BMI is 22.9"}]}},
        ),
    ]
    texts, tcs, ctxs, ai_before_ctx = run_eval._collect_from_events(events)
    assert ai_before_ctx is False


def test_collect_from_events_no_ai_no_context_false() -> None:
    """No AI text and no context → ai_before_ctx False."""
    events: list[tuple[str, Any]] = [("updates", {"agent": {"messages": []}})]
    texts, tcs, ctxs, ai_before_ctx = run_eval._collect_from_events(events)
    assert ai_before_ctx is False


# ── _dedup_tool_calls ─────────────────────────────────────────────────────────


def test_dedup_tool_calls_removes_duplicates() -> None:
    calls = [
        {"tool_name": "calculate_bmi", "arguments": {"height_cm": 175}},
        {"tool_name": "calculate_bmi", "arguments": {"height_cm": 175}},
        {"tool_name": "search_web", "arguments": {"query": "tips"}},
    ]
    result = run_eval._dedup_tool_calls(calls)
    assert len(result) == 2
    assert result[0]["tool_name"] == "calculate_bmi"


def test_dedup_tool_calls_different_args_kept() -> None:
    calls = [
        {"tool_name": "calculate_bmi", "arguments": {"height_cm": 175}},
        {"tool_name": "calculate_bmi", "arguments": {"height_cm": 180}},
    ]
    result = run_eval._dedup_tool_calls(calls)
    assert len(result) == 2


# ── _strip_args_for_no_arg_expected ──────────────────────────────────────────


def test_strip_args_removes_args_for_no_arg_tools() -> None:
    actual = [
        {"tool_name": "calculate_bmi", "arguments": {"height_cm": 175, "weight_kg": 70}}
    ]
    expected = [[{"tool_name": "calculate_bmi"}]]  # no arguments key
    result = run_eval._strip_args_for_no_arg_expected(actual, expected)
    assert result[0] == {"tool_name": "calculate_bmi"}
    assert "arguments" not in result[0]


def test_strip_args_keeps_args_when_expected_has_args() -> None:
    actual = [{"tool_name": "calculate_bmi", "arguments": {"height_cm": 175}}]
    expected = [[{"tool_name": "calculate_bmi", "arguments": {"height_cm": ".*"}}]]
    result = run_eval._strip_args_for_no_arg_expected(actual, expected)
    assert result[0]["arguments"] == {"height_cm": 175}  # args preserved


def test_strip_args_returns_original_when_no_stripping_needed() -> None:
    actual = [{"tool_name": "search_web", "arguments": {"query": "tips"}}]
    expected = [[{"tool_name": "calculate_bmi"}]]  # different tool
    result = run_eval._strip_args_for_no_arg_expected(actual, expected)
    assert result is actual  # same object — no copy


# ── _subprocess_env ───────────────────────────────────────────────────────────


def test_subprocess_env_no_gcp_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    extra_env, tmp_files = run_eval._subprocess_env()
    assert tmp_files == []


def test_subprocess_env_with_valid_gcp_json(monkeypatch: pytest.MonkeyPatch) -> None:
    sa = json.dumps(
        {
            "project_id": "my-proj",
            "type": "service_account",
            "client_email": "sa@my-proj.iam.gserviceaccount.com",
        }
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT", sa)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    extra_env, tmp_files = run_eval._subprocess_env()
    try:
        assert extra_env.get("GOOGLE_CLOUD_PROJECT") == "my-proj"
        assert extra_env.get("VERTEXAI_PROJECT") == "my-proj"
        assert extra_env.get("VERTEXAI_LOCATION") == "us-central1"
        assert "GOOGLE_APPLICATION_CREDENTIALS" in extra_env
        assert len(tmp_files) == 1
        assert tmp_files[0].exists()
    finally:
        for p in tmp_files:
            p.unlink(missing_ok=True)


def test_subprocess_env_skips_write_when_creds_already_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT", '{"project_id":"x"}')
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/already/set.json")
    extra_env, tmp_files = run_eval._subprocess_env()
    assert tmp_files == []


def test_subprocess_env_handles_invalid_json_gracefully(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT", "not-json")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    extra_env, tmp_files = run_eval._subprocess_env()
    try:
        assert "could not parse GCP" in caplog.text
    finally:
        for p in tmp_files:
            p.unlink(missing_ok=True)


def test_subprocess_env_valid_json_without_project_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS_CONTENT", '{"type":"service_account"}'
    )
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    extra_env, tmp_files = run_eval._subprocess_env()
    try:
        assert "GOOGLE_APPLICATION_CREDENTIALS" in extra_env
        assert "GOOGLE_CLOUD_PROJECT" not in extra_env
    finally:
        for p in tmp_files:
            p.unlink(missing_ok=True)


def test_subprocess_env_write_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT", '{"project_id":"x"}')
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    with patch(
        "run_eval.tempfile.NamedTemporaryFile", side_effect=OSError("disk full")
    ):
        extra_env, tmp_files = run_eval._subprocess_env()
    assert extra_env == {}
    assert tmp_files == []
    assert "could not write GCP credentials" in caplog.text


# ── _extract_interrupt_response ───────────────────────────────────────────────


def test_extract_interrupt_response_with_description() -> None:
    events = [
        (
            "updates",
            {
                "__interrupt__": [
                    {
                        "value": {
                            "action_requests": [
                                {
                                    "description": "Approve sending email?",
                                    "name": "send_email",
                                    "args": {},
                                }
                            ]
                        }
                    }
                ]
            },
        )
    ]
    response = run_eval._extract_interrupt_response(events)
    assert "Approve sending email?" in response


def test_extract_interrupt_response_falls_back_to_name_args() -> None:
    events = [
        (
            "updates",
            {
                "__interrupt__": [
                    {
                        "value": {
                            "action_requests": [
                                {
                                    "name": "send_email",
                                    "args": {"to": "user@example.com"},
                                }
                            ]
                        }
                    }
                ]
            },
        )
    ]
    response = run_eval._extract_interrupt_response(events)
    assert "send_email" in response


def test_extract_interrupt_response_empty_when_no_interrupt() -> None:
    events: list[tuple[str, Any]] = [("updates", {"agent": {}})]
    assert run_eval._extract_interrupt_response(events) == ""


def test_extract_interrupt_response_skips_non_updates() -> None:
    events = [
        ("events", {"event": "on_tool_start"}),
        ("updates", {"agent": {}}),
    ]
    assert run_eval._extract_interrupt_response(events) == ""


def test_extract_interrupt_response_empty_action_requests() -> None:
    events: list[tuple[str, Any]] = [
        ("updates", {"__interrupt__": [{"value": {"action_requests": []}}]})
    ]
    assert run_eval._extract_interrupt_response(events) == ""


def test_extract_interrupt_response_multiple_requests() -> None:
    events = [
        (
            "updates",
            {
                "__interrupt__": [
                    {
                        "value": {
                            "action_requests": [
                                {"description": "First approval"},
                                {"description": "Second approval"},
                            ]
                        }
                    }
                ]
            },
        )
    ]
    response = run_eval._extract_interrupt_response(events)
    assert "First approval" in response
    assert "Second approval" in response


# ── _find_lightspeed_cmd ────────────────────────────────────────────────────────


def test_find_lightspeed_cmd_from_venv_bin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "lightspeed-eval").write_text("#!/bin/sh\n")
    monkeypatch.setattr(run_eval.sys, "executable", str(fake_bin / "python"))
    result = run_eval._find_lightspeed_cmd()
    assert result == [str(fake_bin / "lightspeed-eval")]


def test_find_lightspeed_cmd_via_shutil_which(monkeypatch: pytest.MonkeyPatch) -> None:
    # Now uses shutil.which (local import inside the function)
    monkeypatch.setattr(run_eval.sys, "executable", "/usr/bin/python3")
    with patch.object(Path, "exists", return_value=False):
        with patch("shutil.which", return_value="/usr/local/bin/lightspeed-eval"):
            result = run_eval._find_lightspeed_cmd()
    assert result == ["/usr/local/bin/lightspeed-eval"]


def test_find_lightspeed_cmd_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval.sys, "executable", "/usr/bin/python3")
    with patch.object(Path, "exists", return_value=False):
        with patch("shutil.which", return_value=None):
            result = run_eval._find_lightspeed_cmd()
    assert result is None


# ── _parse_args ─────────────────────────────────────────────────────────────────


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_HOST", raising=False)
    monkeypatch.delenv("AGENT_AUTH_TOKEN", raising=False)
    with patch.object(run_eval.sys, "argv", ["run_eval.py"]):
        args = run_eval._parse_args()
    assert args.agent_url == run_eval.DEFAULT_AGENT_URL
    assert args.eval_data == [run_eval.DEFAULT_EVAL_DATA]
    assert args.system == run_eval.DEFAULT_SYSTEM
    assert args.output_dir == run_eval.DEFAULT_OUTPUT_DIR
    assert args.auth_token is None
    assert args.timeout == 300


def test_parse_args_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_HOST", "http://agent.test")
    monkeypatch.setenv("AGENT_AUTH_TOKEN", "secret")
    with patch.object(run_eval.sys, "argv", ["run_eval.py"]):
        args = run_eval._parse_args()
    assert args.agent_url == "http://agent.test"
    assert args.auth_token == "secret"


# ── _run_lightspeed ───────────────────────────────────────────────────────────


def test_run_lightspeed_invokes_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = tmp_path / "system.yaml"
    populated = tmp_path / "populated.yaml"
    output_dir = tmp_path / "output"
    system.write_text("system: true")
    populated.write_text("data: true")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT", raising=False)

    mock_run = MagicMock(return_value=MagicMock(returncode=0))
    with patch("run_eval.subprocess.run", mock_run):
        code = run_eval._run_lightspeed(
            system, populated, output_dir, ["/usr/bin/lightspeed-eval"]
        )

    assert code == 0
    assert output_dir.exists()
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd == [
        "/usr/bin/lightspeed-eval",
        "--system-config",
        str(system),
        "--eval-data",
        str(populated),
        "--output-dir",
        str(output_dir),
    ]


def test_run_lightspeed_returns_nonzero_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = tmp_path / "system.yaml"
    populated = tmp_path / "populated.yaml"
    output_dir = tmp_path / "output"
    system.write_text("system: true")
    populated.write_text("data: true")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT", raising=False)

    mock_run = MagicMock(return_value=MagicMock(returncode=2))
    with patch("run_eval.subprocess.run", mock_run):
        code = run_eval._run_lightspeed(
            system, populated, output_dir, ["/usr/bin/lightspeed-eval"]
        )

    assert code == 2


def test_run_lightspeed_cleans_up_temp_credential_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = tmp_path / "system.yaml"
    populated = tmp_path / "populated.yaml"
    output_dir = tmp_path / "output"
    system.write_text("system: true")
    populated.write_text("data: true")
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS_CONTENT",
        json.dumps({"project_id": "test-proj", "type": "service_account"}),
    )
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    created_paths: list[Path] = []
    original_subprocess_env = run_eval._subprocess_env

    def capture_subprocess_env() -> tuple[dict[str, str], list[Path]]:
        env, files = original_subprocess_env()
        created_paths.extend(files)
        return env, files

    mock_run = MagicMock(return_value=MagicMock(returncode=0))
    with patch("run_eval._subprocess_env", side_effect=capture_subprocess_env):
        with patch("run_eval.subprocess.run", mock_run):
            run_eval._run_lightspeed(
                system, populated, output_dir, ["/usr/bin/lightspeed-eval"]
            )

    assert created_paths
    for path in created_paths:
        assert not path.exists()


# ── main ──────────────────────────────────────────────────────────────────────


def _main_args(tmp_path: Path) -> MagicMock:
    eval_data = tmp_path / "eval_data.yaml"
    system = tmp_path / "system.yaml"
    eval_data.write_text("- conversation_group_id: g1\n  turns: []\n")
    system.write_text("metrics: []\n")
    return MagicMock(
        agent_url="http://localhost:5002",
        eval_data=[eval_data],
        system=system,
        output_dir=tmp_path / "output",
        auth_token=None,
        timeout=300,
    )


def test_main_exits_when_eval_data_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    args = _main_args(tmp_path)
    args.eval_data = [missing]
    with patch.object(run_eval, "_parse_args", return_value=args):
        with pytest.raises(SystemExit) as exc:
            run_eval.main()
    assert exc.value.code == 1


def test_main_exits_when_system_missing(tmp_path: Path) -> None:
    args = _main_args(tmp_path)
    args.system = tmp_path / "missing_system.yaml"
    with patch.object(run_eval, "_parse_args", return_value=args):
        with pytest.raises(SystemExit) as exc:
            run_eval.main()
    assert exc.value.code == 1


def test_main_exits_when_lightspeed_not_found(tmp_path: Path) -> None:
    args = _main_args(tmp_path)
    with patch.object(run_eval, "_parse_args", return_value=args):
        with patch.object(run_eval, "_find_lightspeed_cmd", return_value=None):
            with pytest.raises(SystemExit) as exc:
                run_eval.main()
    assert exc.value.code == 1


def test_main_success_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _main_args(tmp_path)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    with patch.object(run_eval, "_parse_args", return_value=args):
        with patch.object(
            run_eval, "_find_lightspeed_cmd", return_value=["/bin/lightspeed-eval"]
        ):
            with patch.object(
                run_eval, "_populate_dataset", return_value=[{"turns": []}]
            ):
                with patch.object(run_eval, "_run_lightspeed", return_value=0):
                    with pytest.raises(SystemExit) as exc:
                        run_eval.main()
    assert exc.value.code == 0


def test_main_exits_with_lightspeed_failure_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _main_args(tmp_path)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    with patch.object(run_eval, "_parse_args", return_value=args):
        with patch.object(
            run_eval, "_find_lightspeed_cmd", return_value=["/bin/lightspeed-eval"]
        ):
            with patch.object(
                run_eval, "_populate_dataset", return_value=[{"turns": []}]
            ):
                with patch.object(run_eval, "_run_lightspeed", return_value=3):
                    with pytest.raises(SystemExit) as exc:
                        run_eval.main()
    assert exc.value.code == 3


def test_main_warns_without_google_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    args = _main_args(tmp_path)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_CONTENT", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    with caplog.at_level(logging.WARNING, logger="run_eval"):
        with patch.object(run_eval, "_parse_args", return_value=args):
            with patch.object(
                run_eval, "_find_lightspeed_cmd", return_value=["/bin/lightspeed-eval"]
            ):
                with patch.object(
                    run_eval, "_populate_dataset", return_value=[{"turns": []}]
                ):
                    with patch.object(run_eval, "_run_lightspeed", return_value=0):
                        with pytest.raises(SystemExit):
                            run_eval.main()
    assert "no Google credentials found" in caplog.text
