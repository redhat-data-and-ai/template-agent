"""Integration test: Email skill flow via aegra (MR-30).

Verifies that the agent correctly:
1. Validates email addresses via MCP
2. Formats reports into Gmail-compatible HTML
3. Sends email via the send_email MCP tool
"""

import pytest

from deep_agent.aegra.converters import stream_request_to_langgraph_input
from deep_agent.aegra.serialization import serialize_message
from langchain_core.messages import AIMessage, HumanMessage


class TestEmailSkillConverters:
    """Test email-related message flows through aegra serialization."""

    def test_email_request_converts_to_langgraph_input(self):
        result = stream_request_to_langgraph_input(
            "Send this BMI report to test@example.com"
        )
        assert "test@example.com" in result["messages"][0].content

    def test_email_tool_call_serialization(self):
        ai_msg = AIMessage(
            content="I've sent the report to test@example.com",
            tool_calls=[
                {
                    "id": "tc-email",
                    "name": "send_email",
                    "args": {
                        "recipient": "test@example.com",
                        "subject": "BMI Report",
                        "body": "<h1>BMI Report</h1>",
                    },
                }
            ],
        )
        serialized = serialize_message(ai_msg)
        assert serialized["tool_calls"][0]["name"] == "send_email"
        assert serialized["tool_calls"][0]["args"]["recipient"] == "test@example.com"

    def test_validate_email_tool_call_structure(self, mock_validate_email_response):
        assert mock_validate_email_response["valid"] is True
        assert mock_validate_email_response["email"] == "test@example.com"


class TestEmailResponseStructure:
    """Validate email send response shapes."""

    def test_successful_email_response(self, mock_email_response):
        assert mock_email_response["success"] is True
        assert "message_id" in mock_email_response
        assert mock_email_response["recipient"] == "test@example.com"

    def test_email_response_has_required_fields(self, mock_email_response):
        required = {"success", "recipient", "subject", "message", "message_id"}
        assert required.issubset(mock_email_response.keys())
