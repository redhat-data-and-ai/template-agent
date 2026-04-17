"""Streaming response components for the template agent system.

This package contains the modular components that handle streaming responses,
message deduplication, tool call tracking, and event formatting.
"""

from template_agent.src.core.streaming.context import StreamContext
from template_agent.src.core.streaming.converter import remove_tool_calls
from template_agent.src.core.streaming.deduplicator import MessageDeduplicator
from template_agent.src.core.streaming.handlers import (
    TokenEventHandler,
    UpdateEventHandler,
)
from template_agent.src.core.streaming.tracker import ToolCallTracker

__all__ = [
    "StreamContext",
    "MessageDeduplicator",
    "ToolCallTracker",
    "UpdateEventHandler",
    "TokenEventHandler",
    "remove_tool_calls",
]
