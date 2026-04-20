"""Streaming response components for the template agent system.

This package contains the modular components that handle streaming responses,
message deduplication, tool call tracking, and event formatting.
"""

from template_agent.src.streaming.context import StreamContext
from template_agent.src.streaming.converter import remove_tool_calls
from template_agent.src.streaming.deduplicator import MessageDeduplicator
from template_agent.src.streaming.handlers import (
    TokenEventHandler,
    UpdateEventHandler,
)
from template_agent.src.streaming.tracker import ToolCallTracker

__all__ = [
    "StreamContext",
    "MessageDeduplicator",
    "ToolCallTracker",
    "UpdateEventHandler",
    "TokenEventHandler",
    "remove_tool_calls",
]
