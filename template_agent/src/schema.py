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
        description="Run ID associated with this message for tracking.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    thread_id: str | None = Field(
        description="Thread ID associated with this message for conversation tracking.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    session_id: str | None = Field(
        description="Session ID associated with this message for session tracking.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    ai_call_id: str | None = Field(
        description="Unique identifier for the AI call that generated this message.",
        default=None,
        examples=["ai_call_847c6285-8fc9-4560-a83f-4e6285809254"],
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

    run_id: str = Field(
        description="Run ID to record feedback for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="Feedback key identifier.",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="Feedback score value.",
        examples=[0.8],
    )
    kwargs: dict[str, Any] = Field(
        description="Additional feedback parameters passed to LangFuse.",
        default={},
        examples=[{"comment": "In-line human feedback"}],
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


class OpenAIMessage(BaseModel):
    """OpenAI-compatible message model."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="The role of the message author.",
    )
    content: str | None = Field(
        description="The contents of the message.",
        default=None,
    )
    name: str | None = Field(
        description="The name of the author of this message.",
        default=None,
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        description="The tool calls generated by the model.",
        default=None,
    )
    tool_call_id: str | None = Field(
        description="Tool call that this message is responding to.",
        default=None,
    )


class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(
        description="ID of the model to use.",
        default="gpt-4",
    )
    messages: list[OpenAIMessage] = Field(
        description="A list of messages comprising the conversation so far.",
    )
    max_tokens: int | None = Field(
        description="The maximum number of tokens to generate.",
        default=None,
    )
    temperature: float | None = Field(
        description="Sampling temperature between 0 and 2.",
        default=1.0,
        ge=0,
        le=2,
    )
    top_p: float | None = Field(
        description="Nucleus sampling parameter.",
        default=1.0,
        ge=0,
        le=1,
    )
    n: int | None = Field(
        description="Number of chat completion choices to generate.",
        default=1,
    )
    stream: bool = Field(
        description="Whether to incrementally stream back partial message deltas.",
        default=False,
    )
    stop: str | list[str] | None = Field(
        description="Up to 4 sequences where the API will stop generating tokens.",
        default=None,
    )
    presence_penalty: float | None = Field(
        description="Penalty for new tokens based on their presence in the text so far.",
        default=0.0,
        ge=-2.0,
        le=2.0,
    )
    frequency_penalty: float | None = Field(
        description="Penalty for new tokens based on their frequency in the text so far.",
        default=0.0,
        ge=-2.0,
        le=2.0,
    )
    user: str | None = Field(
        description="A unique identifier representing your end-user.",
        default=None,
    )


class OpenAIUsage(BaseModel):
    """OpenAI-compatible usage statistics."""

    prompt_tokens: int = Field(description="Number of tokens in the prompt.")
    completion_tokens: int = Field(description="Number of tokens in the completion.")
    total_tokens: int = Field(description="Total number of tokens used.")


class OpenAIChoice(BaseModel):
    """OpenAI-compatible choice object."""

    index: int = Field(description="The index of the choice in the list of choices.")
    message: OpenAIMessage = Field(
        description="A chat completion message generated by the model."
    )
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = (
        Field(
            description="The reason the model stopped generating tokens.",
            default=None,
        )
    )


class OpenAIChatResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(description="A unique identifier for the chat completion.")
    object: Literal["chat.completion"] = Field(
        description="The object type, which is always 'chat.completion'.",
        default="chat.completion",
    )
    created: int = Field(
        description="The Unix timestamp of when the chat completion was created."
    )
    model: str = Field(description="The model used for the chat completion.")
    choices: list[OpenAIChoice] = Field(
        description="A list of chat completion choices."
    )
    usage: OpenAIUsage | None = Field(
        description="Usage statistics for the completion request.",
        default=None,
    )


class OpenAIChatStreamDelta(BaseModel):
    """OpenAI-compatible streaming delta object."""

    role: Literal["assistant"] | None = Field(
        description="The role of the author of this message.",
        default=None,
    )
    content: str | None = Field(
        description="The contents of the chunk message.",
        default=None,
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        description="The tool calls generated by the model.",
        default=None,
    )


class OpenAIChatStreamChoice(BaseModel):
    """OpenAI-compatible streaming choice object."""

    index: int = Field(description="The index of the choice in the list of choices.")
    delta: OpenAIChatStreamDelta = Field(
        description="A chat completion delta generated by the model."
    )
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = (
        Field(
            description="The reason the model stopped generating tokens.",
            default=None,
        )
    )


class OpenAIChatStreamResponse(BaseModel):
    """OpenAI-compatible streaming chat completion response."""

    id: str = Field(description="A unique identifier for the chat completion.")
    object: Literal["chat.completion.chunk"] = Field(
        description="The object type, which is always 'chat.completion.chunk'.",
        default="chat.completion.chunk",
    )
    created: int = Field(
        description="The Unix timestamp of when the chat completion was created."
    )
    model: str = Field(description="The model used for the chat completion.")
    choices: list[OpenAIChatStreamChoice] = Field(
        description="A list of chat completion choices."
    )
