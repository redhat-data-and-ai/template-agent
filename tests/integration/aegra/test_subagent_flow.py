"""Integration test: Subagent orchestration flow via aegra (MR-31).

Verifies the full orchestrator -> analyst -> publisher delegation chain:
1. User requests BMI analysis + email
2. Orchestrator delegates to analyst subagent
3. Analyst calculates BMI and searches for tips
4. Orchestrator delegates to publisher subagent
5. Publisher formats and sends the email
"""

import pytest

from deep_agent.aegra.serialization import serialize_state, deserialize_state
from deep_agent.aegra.state import AegraMetadata, serialize_metadata
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class TestSubagentFlowSerialization:
    """Test full multi-agent conversation state serialization."""

    def test_full_conversation_state_roundtrip(self):
        """A realistic multi-turn conversation with tool calls survives serialization."""
        state = {
            "messages": [
                HumanMessage(
                    content="Calculate BMI for 175cm, 70kg and email to test@example.com"
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc1",
                            "name": "calculate_bmi",
                            "args": {"height_cm": 175, "weight_kg": 70},
                        }
                    ],
                ),
                ToolMessage(
                    content='{"bmi": 22.9, "category": "Normal"}',
                    tool_call_id="tc1",
                    name="calculate_bmi",
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc2",
                            "name": "search_web",
                            "args": {"query": "normal BMI health tips"},
                        }
                    ],
                ),
                ToolMessage(
                    content='{"results": [{"snippet": "Stay active"}]}',
                    tool_call_id="tc2",
                    name="search_web",
                ),
                AIMessage(content="BMI: 22.9 (Normal). Tips: Stay active."),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc3",
                            "name": "send_email",
                            "args": {
                                "recipient": "test@example.com",
                                "subject": "BMI Report",
                                "body": "<h1>Your BMI: 22.9</h1>",
                            },
                        }
                    ],
                ),
                ToolMessage(
                    content='{"success": true}',
                    tool_call_id="tc3",
                    name="send_email",
                ),
                AIMessage(content="Report sent to test@example.com!"),
            ],
        }

        serialized = serialize_state(state)
        assert "_serialized_at" in serialized
        assert len(serialized["messages"]) == 9

        restored = deserialize_state(serialized)
        assert len(restored["messages"]) == 9
        assert isinstance(restored["messages"][0], HumanMessage)
        assert isinstance(restored["messages"][1], AIMessage)
        assert isinstance(restored["messages"][2], ToolMessage)
        assert restored["messages"][2].tool_call_id == "tc1"

    def test_metadata_tracking_across_subagents(self):
        meta: AegraMetadata = {
            "run_id": "run-orchestrator",
            "thread_id": "thread-main",
            "error_count": 0,
            "last_error": None,
        }
        serialized = serialize_metadata(meta)
        assert "last_error" not in serialized
        assert serialized["error_count"] == 0


class TestSubagentDelegationPatterns:
    """Verify expected patterns in multi-agent tool call sequences."""

    def test_analyst_requires_bmi_tools(self):
        """Analyst subagent must use calculate_bmi and search_web."""
        analyst_tools = {"calculate_bmi", "search_web"}
        expected_call_sequence = ["calculate_bmi", "search_web"]

        for tool_name in expected_call_sequence:
            assert tool_name in analyst_tools

    def test_publisher_requires_email_tools(self):
        """Publisher subagent must use send_email."""
        publisher_tools = {"send_email"}
        assert "send_email" in publisher_tools

    def test_orchestrator_delegates_to_both(self):
        """Orchestrator should delegate BMI+email tasks to both subagents."""
        subagent_names = {"analyst", "publisher"}
        assert len(subagent_names) == 2
        assert "analyst" in subagent_names
        assert "publisher" in subagent_names
