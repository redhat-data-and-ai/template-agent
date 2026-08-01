"""Shared test fixtures for aegra integration tests (MR-34).

Provides:
- Mock MCP server (in-process via httpx)
- LangGraph API client fixture
- Thread/run management helpers
- State snapshot assertions
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


MOCK_MCP_URL = "http://mock-mcp:5001"
LANGGRAPH_API_URL = os.environ.get("LANGGRAPH_API_URL", "http://127.0.0.1:2024")


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def mock_bmi_response() -> dict[str, Any]:
    """Standard BMI calculation response for a normal-weight person."""
    return {
        "success": True,
        "bmi": 24.7,
        "category": "Normal",
        "height_cm": 180,
        "weight_kg": 80,
    }


@pytest.fixture()
def mock_email_response() -> dict[str, Any]:
    """Standard email send response."""
    return {
        "success": True,
        "recipient": "test@example.com",
        "subject": "BMI Report",
        "message": "Email sent successfully to test@example.com",
        "message_id": "mock-12345",
    }


@pytest.fixture()
def mock_search_response() -> dict[str, Any]:
    """Standard web search response with health tips."""
    return {
        "success": True,
        "query": "normal BMI health tips",
        "category": "Normal",
        "results": [
            {"title": "Tip 1", "snippet": "Maintain a balanced diet"},
            {"title": "Tip 2", "snippet": "Exercise 150 min/week"},
            {"title": "Tip 3", "snippet": "Stay hydrated"},
        ],
    }


@pytest.fixture()
def mock_validate_email_response() -> dict[str, Any]:
    """Standard email validation response."""
    return {
        "success": True,
        "valid": True,
        "email": "test@example.com",
        "message": "Valid email format",
    }


@pytest.fixture()
def sample_thread_id() -> str:
    return "test-thread-001"


@pytest.fixture()
def sample_user_id() -> str:
    return "test-user-001"


@pytest.fixture()
def sample_bmi_input() -> dict[str, Any]:
    """Standard BMI request payload for the agent."""
    return {
        "messages": [
            {
                "role": "human",
                "content": (
                    "Calculate BMI for someone who is 180cm tall and weighs 80kg. "
                    "Send the report to test@example.com"
                ),
            }
        ]
    }


@pytest.fixture()
def sample_email_input() -> dict[str, Any]:
    """Email-only request payload for testing the publisher subagent."""
    return {
        "messages": [
            {
                "role": "human",
                "content": "Send this report to test@example.com: BMI is 24.7, Normal weight.",
            }
        ]
    }


@pytest.fixture()
def langgraph_api_url() -> str:
    """Base URL for the LangGraph API server."""
    return LANGGRAPH_API_URL


@pytest.fixture()
def api_headers() -> dict[str, str]:
    """Default headers for LangGraph API requests."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
