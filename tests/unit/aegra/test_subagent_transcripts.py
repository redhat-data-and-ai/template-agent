"""Saved subagent transcript shaping."""

from deep_agent.aegra.subagent_transcripts import (
    select_transcripts,
    transcript_from_messages,
)


def test_transcript_keeps_thought_answer_and_tool_result() -> None:
    built = transcript_from_messages(
        [
            {"type": "human", "content": "Get height and weight"},
            {
                "type": "ai",
                "id": "sub-1",
                "content": [
                    {"type": "text", "text": "Let me check", "thought": True},
                    {"type": "text", "text": "Top accounts"},
                ],
                "tool_calls": [
                    {"id": "inner-1", "name": "get_sql", "args": {"query": "orders"}}
                ],
            },
            {"type": "tool", "tool_call_id": "inner-1", "content": "rows"},
        ]
    )

    assert built is not None
    assert built["delegation"] == "Get height and weight"
    assert built["entries"] == [
        {
            "id": "sub-1",
            "thinking": "Let me check",
            "content": "Top accounts",
            "tool_calls": [
                {
                    "id": "inner-1",
                    "name": "get_sql",
                    "args": {"query": "orders"},
                    "content": "rows",
                }
            ],
        }
    ]


def test_select_transcripts_uses_the_delegation_list_for_each_launch() -> None:
    chosen = select_transcripts(
        {
            "tools:task-b|tools:inner": [
                {
                    "type": "ai",
                    "id": "only-tool",
                    "content": "",
                    "tool_calls": [{"id": "x", "name": "get_sql", "args": {}}],
                },
            ],
            "tools:task-b": [
                {"type": "human", "content": "Second question"},
                {"type": "ai", "id": "b", "content": "Second answer"},
            ],
            "tools:task-a": [
                {"type": "human", "content": "First question"},
                {"type": "ai", "id": "a", "content": "First answer"},
            ],
        }
    )

    assert [item["delegation"] for item in chosen] == [
        "First question",
        "Second question",
    ]
    assert chosen[0]["entries"][0]["content"] == "First answer"
