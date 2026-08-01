"""Schema definitions for the template agent.

This module contains Pydantic models and TypedDict definitions for
request/response validation, data serialization, and API documentation
for the template agent service.
"""

from typing import Any, Literal, NotRequired

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class UserInput(BaseModel):
    """Basic user input model for agent interactions.

    This model represents the core user input structure used across
    different endpoints in the template agent API.
    """

    message: str = Field(
        description="User input message to be processed by the agent.",
        examples=["What is 2 multiplied by 3?"],
    )
    thread_id: str | None = Field(
        description="Thread ID to persist and continue a multi-turn conversation. Auto-generated if not provided.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    session_id: str | None = Field(
        description="Session ID to persist and continue a conversation across multiple threads.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(
        description="User ID to persist and continue a conversation across multiple threads.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class StreamRequest(UserInput):
    """User input model for streaming agent responses.

    Extends UserInput with streaming-specific configuration options
    for real-time response generation.
    """

    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client in real-time.",
        default=True,
    )


class ToolCall(TypedDict):
    """Represents a request to call a tool.

    This TypedDict defines the structure for tool call requests,
    including the tool name, arguments, and optional identifier.
    """

    name: str
    """The name of the tool to be called."""
    args: dict[str, Any]
    """The arguments to pass to the tool call."""
    id: str | None
    """An optional identifier associated with the tool call."""
    type: NotRequired[Literal["tool_call"]]
    """The type identifier for tool calls."""


class ChatMessage(BaseModel):
    """Message model for chat conversations.

    This model represents individual messages in a chat conversation,
    supporting different message types and metadata.
    """

    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="The role or type of the message in the conversation.",
        examples=["human", "ai", "tool", "custom"],
    )
    content: str = Field(
        description="The text content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: list[ToolCall] = Field(
        description="Tool calls included in this message.",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="ID of the tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="Run ID associated with this message for tracking (hex format).",
        default=None,
        examples=["847c62858fc94560a83f4e6285809254"],
    )
    trace_id: str | None = Field(
        description="Trace ID associated with this message for tracing (hex format).",
        default=None,
        examples=["847c62858fc94560a83f4e6285809254"],
    )
    thread_id: str | None = Field(
        description="Thread ID associated with this message for conversation tracking (hex format).",
        default=None,
        examples=["847c62858fc94560a83f4e6285809254"],
    )
    session_id: str | None = Field(
        description="Session ID associated with this message for session tracking (hex format).",
        default=None,
        examples=["847c62858fc94560a83f4e6285809254"],
    )
    response_metadata: dict[str, Any] = Field(
        description="Additional metadata for the response, such as headers, logprobs, or token counts.",
        default={},
    )
    custom_data: dict[str, Any] = Field(
        description="Custom data associated with this message.",
        default={},
    )


class FeedbackRequest(BaseModel):
    """Feedback model for recording user feedback to LangFuse.

    This model represents feedback data that can be recorded to
    LangFuse for analytics and monitoring purposes.
    """

    trace_id: str = Field(
        description="Trace ID to record feedback for (hex format, no hyphens).",
        examples=["847c62858fc94560a83f4e6285809254"],
    )
    name: str = Field(
        description="Score name/identifier.",
        examples=["user-rating", "thumbs-up"],
    )
    value: float = Field(
        description="Score value.",
        examples=[0.8, 1.0],
    )
    kwargs: dict[str, Any] = Field(
        description="Additional feedback parameters passed to LangFuse.",
        default={},
        examples=[{"comment": "In-line human feedback"}],
    )
    thread_id: str | None = Field(
        default=None,
        description="Thread ID for persistence",
    )
    message_id: str | None = Field(
        default=None,
        description="Message ID for persistence",
    )
    user_id: str | None = Field(
        default=None,
        description="User ID for persistence",
    )


class FeedbackResponse(BaseModel):
    """Response model for feedback operations.

    Simple response model indicating successful feedback recording.
    """

    status: Literal["success"] = "success"


class ChatHistoryResponse(BaseModel):
    """Response model for chat history requests.

    Contains a list of chat messages representing the conversation history.
    """

    messages: list[ChatMessage]
