"""Tests for aegra.converters module."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deep_agent.aegra.converters import (
    extract_final_response,
    langgraph_messages_to_dicts,
    stream_request_to_langgraph_input,
)


class TestStreamRequestToLanggraphInput:
    """Tests for converting raw messages to LangGraph input format."""

    def test_basic_message(self):
        result = stream_request_to_langgraph_input("Hello, agent!")
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)
        assert result["messages"][0].content == "Hello, agent!"

    def test_multiline_message(self):
        result = stream_request_to_langgraph_input("Line 1\nLine 2")
        assert result["messages"][0].content == "Line 1\nLine 2"

    def test_empty_message(self):
        result = stream_request_to_langgraph_input("")
        assert result["messages"][0].content == ""


class TestLanggraphMessagesToDicts:
    """Tests for LangChain message serialization."""

    def test_human_message(self):
        msgs = [HumanMessage(content="hi")]
        result = langgraph_messages_to_dicts(msgs)
        assert result == [{"content": "hi", "role": "human"}]

    def test_ai_message(self):
        msgs = [AIMessage(content="hello")]
        result = langgraph_messages_to_dicts(msgs)
        assert result == [{"content": "hello", "role": "ai"}]

    def test_system_message(self):
        msgs = [SystemMessage(content="you are helpful")]
        result = langgraph_messages_to_dicts(msgs)
        assert result == [{"content": "you are helpful", "role": "system"}]

    def test_mixed_conversation(self):
        msgs = [
            SystemMessage(content="system"),
            HumanMessage(content="question"),
            AIMessage(content="answer"),
        ]
        result = langgraph_messages_to_dicts(msgs)
        assert len(result) == 3
        assert [r["role"] for r in result] == ["system", "human", "ai"]

    def test_ai_message_with_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate_bmi",
                    "args": {"height": 180, "weight": 80},
                    "id": "tc1",
                }
            ],
        )
        result = langgraph_messages_to_dicts([msg])
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["name"] == "calculate_bmi"

    def test_empty_list(self):
        assert langgraph_messages_to_dicts([]) == []


class TestExtractFinalResponse:
    """Tests for extracting the last AI response from state."""

    def test_extracts_last_ai_message(self):
        state = {
            "messages": [
                HumanMessage(content="What's my BMI?"),
                AIMessage(content="Your BMI is 24.7"),
            ]
        }
        assert extract_final_response(state) == "Your BMI is 24.7"

    def test_skips_empty_ai_messages(self):
        state = {
            "messages": [
                AIMessage(content="first response"),
                AIMessage(content=""),
            ]
        }
        assert extract_final_response(state) == "first response"

    def test_no_ai_messages(self):
        state = {"messages": [HumanMessage(content="hello")]}
        assert extract_final_response(state) is None

    def test_empty_messages(self):
        assert extract_final_response({"messages": []}) is None
        assert extract_final_response({}) is None
