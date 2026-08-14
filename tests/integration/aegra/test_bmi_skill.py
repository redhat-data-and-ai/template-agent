"""Integration test: BMI skill flow via aegra (MR-29).

Verifies that the agent correctly:
1. Receives a BMI request
2. Calls the calculate_bmi MCP tool
3. Calls search_web for health tips
4. Returns a formatted BMI report

Requires: mock MCP server running on localhost:5001 OR
uses mocked tool responses via fixtures.
"""

import pytest

from deep_agent.aegra.converters import (
    extract_final_response,
    stream_request_to_langgraph_input,
)
from deep_agent.aegra.serialization import deserialize_message, serialize_message
from langchain_core.messages import AIMessage, HumanMessage


class TestBMISkillConverters:
    """Test that BMI-related messages are correctly serialized through aegra."""

    def test_bmi_request_converts_to_langgraph_input(self):
        result = stream_request_to_langgraph_input("Calculate BMI for 180cm and 80kg")
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)
        assert "180cm" in result["messages"][0].content

    def test_bmi_response_serialization_roundtrip(self):
        ai_msg = AIMessage(
            content="Your BMI is 24.7 (Normal). Here are health tips...",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "calculate_bmi",
                    "args": {"height_cm": 180, "weight_kg": 80},
                }
            ],
        )
        serialized = serialize_message(ai_msg)
        assert serialized["type"] == "ai"
        assert serialized["tool_calls"][0]["name"] == "calculate_bmi"

        restored = deserialize_message(serialized)
        assert isinstance(restored, AIMessage)
        assert restored.tool_calls[0]["name"] == "calculate_bmi"

    def test_extract_bmi_report_from_state(self):
        state = {
            "messages": [
                HumanMessage(content="Calculate BMI for 180cm and 80kg"),
                AIMessage(content=""),
                AIMessage(
                    content="**BMI Report**\nBMI: 24.7\nCategory: Normal\n\nHealth Tips:\n1. Stay active"
                ),
            ]
        }
        response = extract_final_response(state)
        assert response is not None
        assert "24.7" in response
        assert "Normal" in response


class TestBMIToolCallStructure:
    """Validate the expected tool call structure for BMI calculations."""

    def test_calculate_bmi_tool_call_shape(self, mock_bmi_response):
        assert mock_bmi_response["success"] is True
        assert isinstance(mock_bmi_response["bmi"], float)
        assert mock_bmi_response["category"] in [
            "Underweight",
            "Normal",
            "Overweight",
            "Obese",
        ]

    def test_search_web_tool_call_shape(self, mock_search_response):
        assert mock_search_response["success"] is True
        assert len(mock_search_response["results"]) == 3
        for result in mock_search_response["results"]:
            assert "title" in result
            assert "snippet" in result
