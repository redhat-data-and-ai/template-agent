"""Stream context for carrying metadata through event processing.

This module provides the StreamContext dataclass that carries essential
metadata (run_id, trace_id, thread_id, session_id, user_id) through the
entire streaming pipeline, ensuring all events have consistent context.
"""

from dataclasses import dataclass


@dataclass
class StreamContext:
    """Context object for streaming metadata.

    Carries run, trace, thread, session, and user identifiers plus configuration
    through the event processing pipeline.

    All fields are required to ensure complete context is available
    throughout the streaming pipeline.
    """

    run_id: str
    trace_id: str
    thread_id: str
    session_id: str
    user_id: str
    stream_tokens: bool
