"""Unit tests for aegra serialization module."""

import json
from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deep_agent.aegra.serialization import (
    deserialize_message,
    deserialize_state,
    serialize_message,
    serialize_state,
    state_from_json,
    state_to_json,
)


class TestSerializeMessage:
    def test_human_message(self):
        msg = HumanMessage(content="hello", id="h1")
        result = serialize_message(msg)
        assert result["type"] == "human"
        assert result["content"] == "hello"
        assert result["id"] == "h1"

    def test_ai_message_without_tool_calls(self):
        msg = AIMessage(content="response", id="a1")
        result = serialize_message(msg)
        assert result["type"] == "ai"
        assert "tool_calls" not in result

    def test_ai_message_with_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "search", "args": {"q": "test"}}],
            id="a2",
        )
        result = serialize_message(msg)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search"

    def test_tool_message(self):
        msg = ToolMessage(
            content='{"result": true}',
            tool_call_id="tc1",
            name="search",
            id="t1",
        )
        result = serialize_message(msg)
        assert result["type"] == "tool"
        assert result["tool_call_id"] == "tc1"
        assert result["name"] == "search"

    def test_tool_message_preserves_artifact_and_mcp_app(self):
        msg = ToolMessage(
            content="ok",
            tool_call_id="tc1",
            name="show_chart",
            id="t1",
            artifact={
                "structured_content": {"n": 1},
                "mcp_app": {
                    "server": "chart-mcp-server",
                    "resourceUri": "ui://charts/app.html",
                    "result": {
                        "content": [{"type": "text", "text": "ok"}],
                        "structuredContent": {"n": 1},
                        "isError": False,
                    },
                },
            },
        )
        result = serialize_message(msg)
        assert result["artifact"]["structured_content"]["n"] == 1
        assert result["mcpApp"]["resourceUri"] == "ui://charts/app.html"
        assert result["mcpApp"]["server"] == "chart-mcp-server"
        assert result["mcpApp"]["result"]["isError"] is False
        assert result["mcpApp"]["result"]["structuredContent"] == {"n": 1}
        assert result["mcpApp"]["result"]["content"] == [{"type": "text", "text": "ok"}]

    def test_system_message(self):
        msg = SystemMessage(content="you are helpful")
        result = serialize_message(msg)
        assert result["type"] == "system"

    def test_response_metadata_included(self):
        msg = AIMessage(
            content="hi",
            response_metadata={"model": "gemini-2.5"},
        )
        result = serialize_message(msg)
        assert result["response_metadata"]["model"] == "gemini-2.5"


class TestDeserializeMessage:
    def test_human_message(self):
        data = {"type": "human", "content": "hello", "id": "h1"}
        msg = deserialize_message(data)
        assert isinstance(msg, HumanMessage)
        assert msg.content == "hello"

    def test_ai_message(self):
        data = {"type": "ai", "content": "response", "id": "a1"}
        msg = deserialize_message(data)
        assert isinstance(msg, AIMessage)

    def test_ai_message_with_tool_calls(self):
        data = {
            "type": "ai",
            "content": "",
            "tool_calls": [{"id": "tc1", "name": "search", "args": {"q": "t"}}],
        }
        msg = deserialize_message(data)
        assert isinstance(msg, AIMessage)
        assert msg.tool_calls[0]["name"] == "search"

    def test_system_message(self):
        data = {"type": "system", "content": "sys prompt"}
        msg = deserialize_message(data)
        assert isinstance(msg, SystemMessage)

    def test_tool_message(self):
        data = {
            "type": "tool",
            "content": "result",
            "tool_call_id": "tc1",
            "name": "search",
        }
        msg = deserialize_message(data)
        assert isinstance(msg, ToolMessage)
        assert msg.tool_call_id == "tc1"

    def test_tool_message_with_artifact(self):
        data = {
            "type": "tool",
            "content": "ok",
            "tool_call_id": "tc1",
            "name": "show",
            "artifact": {"structured_content": {"n": 1}},
        }
        msg = deserialize_message(data)
        assert isinstance(msg, ToolMessage)
        assert msg.artifact["structured_content"]["n"] == 1

    def test_tool_message_mcp_app_without_artifact(self):
        data = {
            "type": "tool",
            "content": "ok",
            "tool_call_id": "tc1",
            "name": "show",
            "mcpApp": {
                "server": "srv",
                "resourceUri": "ui://x",
                "result": {"content": [], "isError": False},
            },
        }
        msg = deserialize_message(data)
        assert isinstance(msg, ToolMessage)
        assert msg.additional_kwargs["mcpApp"]["resourceUri"] == "ui://x"
        assert msg.additional_kwargs["mcpApp"]["result"]["isError"] is False

    def test_tool_message_merges_mcp_app_into_existing_additional_kwargs(self):
        data = {
            "type": "tool",
            "content": "ok",
            "tool_call_id": "tc1",
            "name": "show",
            "additional_kwargs": {"trace": "t1"},
            "mcpApp": {
                "server": "srv",
                "resourceUri": "ui://merged",
                "result": {"content": [], "isError": False},
            },
        }
        msg = deserialize_message(data)
        assert msg.additional_kwargs["trace"] == "t1"
        assert msg.additional_kwargs["mcpApp"]["resourceUri"] == "ui://merged"

    def test_human_message_roundtrips_additional_kwargs(self):
        original = HumanMessage(
            content="retry",
            id="h1",
            additional_kwargs={"opa_retry": True},
        )
        restored = deserialize_message(serialize_message(original))
        assert isinstance(restored, HumanMessage)
        assert restored.additional_kwargs["opa_retry"] is True

    def test_unknown_type_defaults_to_human(self):
        data = {"type": "unknown_type", "content": "fallback"}
        msg = deserialize_message(data)
        assert isinstance(msg, HumanMessage)

    def test_missing_type_defaults_to_human(self):
        data = {"content": "no type"}
        msg = deserialize_message(data)
        assert isinstance(msg, HumanMessage)


class TestSerializeState:
    def test_roundtrip(self):
        state = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="hello"),
            ],
            "extra": "value",
        }
        serialized = serialize_state(state)
        assert "_serialized_at" in serialized
        assert len(serialized["messages"]) == 2

        restored = deserialize_state(serialized)
        assert len(restored["messages"]) == 2
        assert isinstance(restored["messages"][0], HumanMessage)
        assert isinstance(restored["messages"][1], AIMessage)
        assert "_serialized_at" not in restored

    def test_non_message_values_preserved(self):
        state = {"count": 42, "flag": True, "messages": []}
        serialized = serialize_state(state)
        assert serialized["count"] == 42
        assert serialized["flag"] is True


class TestStateJsonConversion:
    def test_state_to_json_and_back(self):
        state = {
            "messages": [HumanMessage(content="test")],
            "meta": {"run": "abc"},
        }
        json_str = state_to_json(state)
        restored = state_from_json(json_str)
        assert len(restored["messages"]) == 1
        assert isinstance(restored["messages"][0], HumanMessage)

    def test_state_to_json_with_indent(self):
        state = {"messages": [HumanMessage(content="x")]}
        json_str = state_to_json(state, indent=2)
        assert "\n" in json_str

    def test_handles_nested_objects(self):
        state = {
            "messages": [],
            "nested": {"key": [1, 2, {"inner": "val"}]},
        }
        json_str = state_to_json(state)
        restored = state_from_json(json_str)
        assert restored["nested"]["key"][2]["inner"] == "val"

    def test_handles_datetime(self):
        state = {
            "messages": [],
            "timestamp": datetime.now(UTC),
        }
        json_str = state_to_json(state)
        assert "timestamp" in json_str

    def test_handles_bytes(self):
        state = {"messages": [], "data": b"hello bytes"}
        json_str = state_to_json(state)
        restored = state_from_json(json_str)
        assert restored["data"] == "hello bytes"
