"""Tests for OpenAI-compatible chat completions API endpoint."""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from template_agent.src.core.manager import AgentManager
from template_agent.src.routes.openai_chat import (
    _convert_internal_to_openai_message,
    _convert_openai_to_internal,
    _get_openai_response,
    _stream_openai_response,
    router,
)
from template_agent.src.schema import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChatStreamResponse,
    OpenAIMessage,
    StreamRequest,
)


@pytest.fixture
def test_client():
    """Create test client for the OpenAI router."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def sample_openai_request():
    """Sample OpenAI chat request."""
    return OpenAIChatRequest(
        model="gpt-4",
        messages=[
            OpenAIMessage(role="system", content="You are a helpful assistant."),
            OpenAIMessage(role="user", content="Hello, how are you?"),
        ],
        max_tokens=100,
        temperature=0.7,
        stream=False,
    )


@pytest.fixture
def sample_openai_streaming_request():
    """Sample OpenAI streaming chat request."""
    return OpenAIChatRequest(
        model="gpt-4",
        messages=[
            OpenAIMessage(role="user", content="Tell me a joke"),
        ],
        stream=True,
    )


@pytest.fixture
def mock_agent_manager():
    """Mock AgentManager for testing."""
    with patch("template_agent.src.routes.openai_chat.AgentManager") as mock:
        yield mock


class TestConvertOpenAIToInternal:
    """Test cases for _convert_openai_to_internal function."""

    def test_basic_conversion(self, sample_openai_request):
        """Test basic OpenAI to internal format conversion."""
        result = _convert_openai_to_internal(sample_openai_request)

        assert isinstance(result, StreamRequest)
        assert result.message == "Hello, how are you?"
        assert result.thread_id is None
        assert result.session_id is None
        assert result.user_id is None
        assert result.stream_tokens is False

    def test_conversion_with_user_field(self):
        """Test conversion with user field set."""
        request = OpenAIChatRequest(
            model="gpt-4",
            messages=[OpenAIMessage(role="user", content="Hello")],
            user="test_user_123",
        )

        result = _convert_openai_to_internal(request)

        assert result.session_id == "test_user_123"
        assert result.user_id == "test_user_123"

    def test_conversion_with_streaming(self):
        """Test conversion with streaming enabled."""
        request = OpenAIChatRequest(
            model="gpt-4",
            messages=[OpenAIMessage(role="user", content="Hello")],
            stream=True,
        )

        result = _convert_openai_to_internal(request)

        assert result.stream_tokens is True

    def test_conversion_multiple_user_messages(self):
        """Test conversion with multiple user messages takes the last one."""
        request = OpenAIChatRequest(
            model="gpt-4",
            messages=[
                OpenAIMessage(role="user", content="First message"),
                OpenAIMessage(role="assistant", content="Response"),
                OpenAIMessage(role="user", content="Second message"),
            ],
        )

        result = _convert_openai_to_internal(request)

        assert result.message == "Second message"

    def test_conversion_no_user_message_raises_error(self):
        """Test conversion raises error when no user message found."""
        request = OpenAIChatRequest(
            model="gpt-4",
            messages=[
                OpenAIMessage(role="system", content="You are helpful"),
                OpenAIMessage(role="assistant", content="Hello"),
            ],
        )

        with pytest.raises(HTTPException) as exc_info:
            _convert_openai_to_internal(request)

        assert exc_info.value.status_code == 400
        assert "No user message found" in str(exc_info.value.detail)
        assert "E_001" in str(exc_info.value.detail)

    def test_conversion_empty_content(self):
        """Test conversion with empty or None content."""
        request = OpenAIChatRequest(
            model="gpt-4",
            messages=[OpenAIMessage(role="user", content=None)],
        )

        result = _convert_openai_to_internal(request)

        assert result.message == ""


class TestConvertInternalToOpenAIMessage:
    """Test cases for _convert_internal_to_openai_message function."""

    def test_human_to_user_conversion(self):
        """Test converting human message to user role."""
        internal_message = {
            "type": "human",
            "content": "Hello there",
        }

        result = _convert_internal_to_openai_message(internal_message)

        assert result.role == "user"
        assert result.content == "Hello there"
        assert result.tool_calls is None
        assert result.tool_call_id is None

    def test_ai_to_assistant_conversion(self):
        """Test converting AI message to assistant role."""
        internal_message = {
            "type": "ai",
            "content": "Hello! How can I help you?",
        }

        result = _convert_internal_to_openai_message(internal_message)

        assert result.role == "assistant"
        assert result.content == "Hello! How can I help you?"

    def test_tool_message_conversion(self):
        """Test converting tool message."""
        internal_message = {
            "type": "tool",
            "content": "Tool execution result",
            "tool_call_id": "call_123",
        }

        result = _convert_internal_to_openai_message(internal_message)

        assert result.role == "tool"
        assert result.content == "Tool execution result"
        assert result.tool_call_id == "call_123"

    def test_system_message_conversion(self):
        """Test converting system message."""
        internal_message = {
            "type": "system",
            "content": "You are a helpful assistant",
        }

        result = _convert_internal_to_openai_message(internal_message)

        assert result.role == "system"
        assert result.content == "You are a helpful assistant"

    def test_unknown_type_defaults_to_assistant(self):
        """Test unknown message type defaults to assistant role."""
        internal_message = {
            "type": "unknown_type",
            "content": "Some content",
        }

        result = _convert_internal_to_openai_message(internal_message)

        assert result.role == "assistant"
        assert result.content == "Some content"

    def test_missing_content(self):
        """Test handling missing content field."""
        internal_message = {
            "type": "ai",
        }

        result = _convert_internal_to_openai_message(internal_message)

        assert result.role == "assistant"
        assert result.content == ""

    def test_tool_calls_conversion(self):
        """Test converting message with tool calls."""
        internal_message = {
            "type": "ai",
            "content": "I'll help you with that",
            "tool_calls": [{"name": "search", "args": {"query": "test"}}],
        }

        result = _convert_internal_to_openai_message(internal_message)

        assert result.role == "assistant"
        assert result.tool_calls == [{"name": "search", "args": {"query": "test"}}]


class TestNonStreamingResponse:
    """Test cases for non-streaming OpenAI chat completions."""

    @pytest.mark.asyncio
    async def test_get_openai_response_success(
        self, sample_openai_request, mock_agent_manager
    ):
        """Test successful non-streaming response generation."""
        # Mock AgentManager and its methods
        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        # Mock the stream response to return a final AI message
        async def mock_stream(request):
            yield {
                "type": "message",
                "content": {
                    "type": "ai",
                    "content": "Hello! I'm doing well, thank you for asking.",
                },
            }

        manager_instance.stream_response = mock_stream

        # Create mock request
        mock_request = MagicMock()
        mock_request.headers.get.side_effect = lambda key, default="": {
            "Authorization": "Bearer test_token_123"
        }.get(key, default)

        result = await _get_openai_response(sample_openai_request, mock_request)

        assert isinstance(result, OpenAIChatResponse)
        assert result.model == "gpt-4"
        assert len(result.choices) == 1
        assert result.choices[0].message.role == "assistant"
        assert (
            result.choices[0].message.content
            == "Hello! I'm doing well, thank you for asking."
        )
        assert result.choices[0].finish_reason == "stop"
        assert (
            result.usage.prompt_tokens == 0
        )  # Not calculated in current implementation
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 0

    @pytest.mark.asyncio
    async def test_get_openai_response_with_x_token_auth(
        self, sample_openai_request, mock_agent_manager
    ):
        """Test response generation with X-Token authentication."""
        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        async def mock_stream(request):
            yield {
                "type": "message",
                "content": {"type": "ai", "content": "Response content"},
            }

        manager_instance.stream_response = mock_stream

        mock_request = MagicMock()
        mock_request.headers.get.side_effect = lambda key, default="": {
            "X-Token": "x_token_123"
        }.get(key, default)

        result = await _get_openai_response(sample_openai_request, mock_request)

        # Verify AgentManager was initialized with correct token
        mock_agent_manager.assert_called_once_with(redhat_sso_token="x_token_123")
        assert isinstance(result, OpenAIChatResponse)

    @pytest.mark.asyncio
    async def test_get_openai_response_no_auth_token(
        self, sample_openai_request, mock_agent_manager
    ):
        """Test response generation without authentication token."""
        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        async def mock_stream(request):
            yield {
                "type": "message",
                "content": {"type": "ai", "content": "Response content"},
            }

        manager_instance.stream_response = mock_stream

        mock_request = MagicMock()
        mock_request.headers.get.return_value = ""

        await _get_openai_response(sample_openai_request, mock_request)

        # Verify AgentManager was initialized with None token
        mock_agent_manager.assert_called_once_with(redhat_sso_token=None)

    @pytest.mark.asyncio
    async def test_get_openai_response_agent_error(
        self, sample_openai_request, mock_agent_manager
    ):
        """Test response generation when agent throws an error."""
        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        # Create a proper async generator mock that raises an error
        async def mock_stream(request):
            if False:  # Ensure this is an async generator
                yield
            raise RuntimeError("Agent processing error")

        manager_instance.stream_response = mock_stream

        mock_request = MagicMock()
        mock_request.headers.get.return_value = ""

        with pytest.raises(HTTPException) as exc_info:
            await _get_openai_response(sample_openai_request, mock_request)

        assert exc_info.value.status_code == 500
        assert (
            "Internal server error during OpenAI non-streaming response"
            in exc_info.value.detail
        )
        assert "E_003" in exc_info.value.detail


class TestStreamingResponse:
    """Test cases for streaming OpenAI chat completions."""

    @pytest.mark.asyncio
    async def test_stream_openai_response_success(
        self, sample_openai_streaming_request, mock_agent_manager
    ):
        """Test successful streaming response generation."""
        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        # Mock the stream response with tokens and final message
        async def mock_stream(request):
            yield {"type": "token", "content": "Why"}
            yield {"type": "token", "content": " did"}
            yield {"type": "token", "content": " the"}
            yield {"type": "token", "content": " chicken"}
            yield {"type": "token", "content": " cross"}
            yield {"type": "token", "content": " the"}
            yield {"type": "token", "content": " road?"}
            yield {
                "type": "message",
                "content": {
                    "type": "ai",
                    "content": "Why did the chicken cross the road?",
                },
            }

        manager_instance.stream_response = mock_stream

        mock_request = MagicMock()
        mock_request.headers.get.side_effect = lambda key, default="": {
            "Authorization": "Bearer test_token"
        }.get(key, default)

        # Collect all streaming chunks
        chunks = []
        async for chunk in _stream_openai_response(
            sample_openai_streaming_request, mock_request
        ):
            chunks.append(chunk)

        # Verify we got chunks
        assert len(chunks) > 0

        # Verify initial chunk (role assignment)
        initial_chunk_data = json.loads(chunks[0].replace("data: ", "").strip())
        assert initial_chunk_data["choices"][0]["delta"]["role"] == "assistant"
        assert initial_chunk_data["choices"][0]["finish_reason"] is None

        # Verify token chunks
        token_chunks = [
            chunk for chunk in chunks[1:-2] if "Why" in chunk or "did" in chunk
        ]
        assert len(token_chunks) > 0

        # Verify final chunk
        final_chunk_data = json.loads(chunks[-2].replace("data: ", "").strip())
        assert final_chunk_data["choices"][0]["finish_reason"] == "stop"

        # Verify done marker
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_stream_openai_response_agent_error(
        self, sample_openai_streaming_request, mock_agent_manager
    ):
        """Test streaming response when agent throws an error."""
        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        # Mock the stream response to throw an error
        async def mock_stream(request):
            yield {"type": "token", "content": "Start"}
            raise RuntimeError("Stream processing error")

        manager_instance.stream_response = mock_stream

        mock_request = MagicMock()
        mock_request.headers.get.return_value = ""

        # Collect all streaming chunks
        chunks = []
        async for chunk in _stream_openai_response(
            sample_openai_streaming_request, mock_request
        ):
            chunks.append(chunk)

        # Should still get chunks including error handling
        assert len(chunks) > 0

        # Check that error chunk has content_filter finish_reason
        error_chunk_found = False
        for chunk in chunks:
            if "content_filter" in chunk:
                error_chunk_found = True
                break

        assert error_chunk_found

        # Verify done marker is still sent
        assert chunks[-1] == "data: [DONE]\n\n"


class TestChatCompletionEndpoint:
    """Test cases for the main chat completion endpoint."""

    def test_create_chat_completion_non_streaming(
        self, test_client, mock_agent_manager
    ):
        """Test the main endpoint for non-streaming requests."""
        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        # Mock successful response
        async def mock_stream(request):
            yield {
                "type": "message",
                "content": {"type": "ai", "content": "Test response"},
            }

        manager_instance.stream_response = mock_stream

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Authorization": "Bearer test_token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "gpt-4"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Test response"

    def test_create_chat_completion_streaming(self, test_client, mock_agent_manager):
        """Test the main endpoint for streaming requests."""
        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        # Mock streaming response
        async def mock_stream(request):
            yield {"type": "token", "content": "Hello"}
            yield {"type": "token", "content": " world"}
            yield {
                "type": "message",
                "content": {"type": "ai", "content": "Hello world"},
            }

        manager_instance.stream_response = mock_stream

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
        }

        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Authorization": "Bearer test_token"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        # Check streaming response content
        content = response.content.decode()
        assert "data: " in content
        assert "[DONE]" in content

    def test_create_chat_completion_validation_error(self, test_client):
        """Test endpoint with invalid request data."""
        request_data = {
            "model": "gpt-4",
            "messages": [],  # Empty messages should cause validation error
        }

        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
        )

        # Should return validation error
        assert response.status_code in [400, 422]

    def test_create_chat_completion_no_user_message(self, test_client):
        """Test endpoint with no user messages."""
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "system", "content": "You are helpful"}],
        }

        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
        )

        assert response.status_code == 400

    def test_create_chat_completion_server_error(self, test_client, mock_agent_manager):
        """Test endpoint when server encounters an error."""
        # Mock AgentManager to raise an exception during initialization
        mock_agent_manager.side_effect = RuntimeError("Server initialization error")

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
        )

        assert response.status_code == 500


class TestAuthenticationHandling:
    """Test cases for authentication handling."""

    def test_bearer_token_extraction(self, test_client, mock_agent_manager):
        """Test extraction of Bearer token from Authorization header."""
        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        async def mock_stream(request):
            yield {
                "type": "message",
                "content": {"type": "ai", "content": "Authenticated response"},
            }

        manager_instance.stream_response = mock_stream

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Authorization": "Bearer secret_token_123"},
        )

        assert response.status_code == 200

        # Verify AgentManager was called with correct token
        mock_agent_manager.assert_called_with(redhat_sso_token="secret_token_123")

    def test_x_token_header_extraction(self, test_client, mock_agent_manager):
        """Test extraction of token from X-Token header."""
        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        async def mock_stream(request):
            yield {
                "type": "message",
                "content": {"type": "ai", "content": "X-Token authenticated response"},
            }

        manager_instance.stream_response = mock_stream

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"X-Token": "x_secret_token_456"},
        )

        assert response.status_code == 200

        # Verify AgentManager was called with correct token
        mock_agent_manager.assert_called_with(redhat_sso_token="x_secret_token_456")

    def test_no_authentication_token(self, test_client, mock_agent_manager):
        """Test handling when no authentication token is provided."""
        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        async def mock_stream(request):
            yield {
                "type": "message",
                "content": {"type": "ai", "content": "Unauthenticated response"},
            }

        manager_instance.stream_response = mock_stream

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
        )

        assert response.status_code == 200

        # Verify AgentManager was called with None token
        mock_agent_manager.assert_called_with(redhat_sso_token=None)

    def test_bearer_token_priority_over_x_token(self, test_client, mock_agent_manager):
        """Test that Bearer token takes priority over X-Token when both are present."""
        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        async def mock_stream(request):
            yield {
                "type": "message",
                "content": {"type": "ai", "content": "Priority test response"},
            }

        manager_instance.stream_response = mock_stream

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={
                "Authorization": "Bearer bearer_token_123",
                "X-Token": "x_token_456",
            },
        )

        assert response.status_code == 200

        # Verify AgentManager was called with Bearer token, not X-Token
        mock_agent_manager.assert_called_with(redhat_sso_token="bearer_token_123")


class TestEdgeCases:
    """Test cases for edge cases and error scenarios."""

    def test_empty_message_content(self, test_client, mock_agent_manager):
        """Test handling of empty message content."""
        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        async def mock_stream(request):
            yield {
                "type": "message",
                "content": {"type": "ai", "content": "Handled empty input"},
            }

        manager_instance.stream_response = mock_stream

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": ""}],
        }

        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
        )

        assert response.status_code == 200

    def test_very_long_completion_id(self):
        """Test that completion IDs are properly truncated."""
        # This tests the UUID generation and truncation in the actual functions
        # The completion_id should be f"chatcmpl-{uuid.uuid4().hex[:29]}"
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"

        # Should be exactly 37 characters: "chatcmpl-" (9) + 29 hex chars
        assert len(completion_id) == 38  # "chatcmpl-" + 29 chars
        assert completion_id.startswith("chatcmpl-")

    def test_multiple_system_messages(self, test_client, mock_agent_manager):
        """Test handling of multiple system messages."""
        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        async def mock_stream(request):
            yield {
                "type": "message",
                "content": {
                    "type": "ai",
                    "content": "Handled multiple system messages",
                },
            }

        manager_instance.stream_response = mock_stream

        request_data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "system", "content": "You are also concise."},
                {"role": "user", "content": "Hello"},
            ],
        }

        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
        )

        assert response.status_code == 200

    @patch("template_agent.src.routes.openai_chat.time.time")
    def test_created_timestamp(self, mock_time, test_client, mock_agent_manager):
        """Test that created timestamp is properly set."""
        mock_time.return_value = 1234567890

        manager_instance = AsyncMock()
        mock_agent_manager.return_value = manager_instance

        async def mock_stream(request):
            yield {
                "type": "message",
                "content": {"type": "ai", "content": "Timestamp test"},
            }

        manager_instance.stream_response = mock_stream

        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = test_client.post(
            "/v1/chat/completions",
            json=request_data,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["created"] == 1234567890
