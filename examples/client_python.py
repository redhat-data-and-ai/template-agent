"""Python client example for Template Agent simplified streaming API.

This module provides a simple Python client for interacting with the
Template Agent's streaming API, demonstrating how to handle real-time
responses, different event types, and deep research mode.

Usage:
    python examples/client_python.py

    Or use as a library:
    from examples.client_python import TemplateAgentClient

    client = TemplateAgentClient()
    await client.stream_chat("Hello, world!", "thread-123", "session-123", "user-123")
"""

import asyncio
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp


class TemplateAgentClient:
    """Async Python client for Template Agent streaming API."""

    def __init__(
        self,
        base_url: str = "http://localhost:5002",
        headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize the client.

        Args:
            base_url: Base URL of the Template Agent API
            headers: Optional additional headers (e.g., for authentication)
        """
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **(headers or {}),
        }

    async def stream_chat(
        self,
        message: str,
        thread_id: str,
        session_id: str,
        user_id: str,
        stream_tokens: bool = True,
        deep_research_enabled: bool = False,
        deep_research_require_plan_approval: bool = False,
        timeout: int = 60,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat conversation with the agent.

        Args:
            message: User's input message
            thread_id: Conversation thread identifier
            session_id: Session identifier
            user_id: User identifier
            stream_tokens: Whether to stream individual tokens
            deep_research_enabled: Enable deep research pipeline
            deep_research_require_plan_approval: Require plan approval before research
            timeout: Request timeout in seconds (use 600 for deep research)

        Yields:
            Event dictionaries with 'type' and 'content' fields
        """
        request_data: Dict[str, Any] = {
            "message": message,
            "thread_id": thread_id,
            "session_id": session_id,
            "user_id": user_id,
            "stream_tokens": stream_tokens,
        }

        if deep_research_enabled:
            request_data["deep_research_enabled"] = True
            request_data["deep_research_require_plan_approval"] = (
                deep_research_require_plan_approval
            )

        timeout_config = aiohttp.ClientTimeout(total=timeout)

        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.post(
                f"{self.base_url}/v1/stream", json=request_data, headers=self.headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()

                    if not line_str:
                        continue

                    if line_str == "[DONE]":
                        break

                    try:
                        event = json.loads(line_str)
                        yield event
                    except json.JSONDecodeError:
                        continue

    async def send_message(
        self,
        message: str,
        thread_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: str = "python_client",
        stream_tokens: bool = True,
        deep_research_enabled: bool = False,
        deep_research_require_plan_approval: bool = False,
    ) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Send a message and return the complete response.

        Args:
            message: User's input message
            thread_id: Optional thread ID (generated if not provided)
            session_id: Optional session ID (uses thread_id if not provided)
            user_id: User identifier
            stream_tokens: Whether to stream individual tokens
            deep_research_enabled: Enable deep research pipeline
            deep_research_require_plan_approval: Require plan approval

        Returns:
            Tuple of (final_response_text, all_messages, deep_research_events)
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        if session_id is None:
            session_id = thread_id

        full_response = ""
        all_messages: List[Dict[str, Any]] = []
        dr_events: List[Dict[str, Any]] = []

        timeout = 600 if deep_research_enabled else 60

        async for event in self.stream_chat(
            message,
            thread_id,
            session_id,
            user_id,
            stream_tokens,
            deep_research_enabled=deep_research_enabled,
            deep_research_require_plan_approval=deep_research_require_plan_approval,
            timeout=timeout,
        ):
            event_type = event.get("type")
            content = event.get("content")

            if event_type == "token" and isinstance(content, str):
                full_response += content

            elif event_type == "message" and isinstance(content, dict):
                all_messages.append(content)
                if content.get("type") == "ai" and content.get("content"):
                    if not full_response:
                        full_response = content["content"]

            elif event_type == "deep_research_status" and isinstance(content, dict):
                dr_events.append(content)

            elif event_type == "error":
                error_msg = (
                    content.get("message", "Unknown error")
                    if isinstance(content, dict)
                    else str(content)
                )
                raise Exception(f"Agent error: {error_msg}")

        return full_response, all_messages, dr_events

    async def check_health(self) -> Dict[str, Any]:
        """Check if the API is healthy."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Health check failed: HTTP {response.status}")


async def example_streaming_chat():
    """Example of streaming chat with token updates."""
    print("=== Standard Chat Example ===")
    print("=" * 50)

    client = TemplateAgentClient()

    try:
        health = await client.check_health()
        print(f"API Status: {health.get('status', 'unknown')}")
    except Exception as e:
        print(f"API Health Check Failed: {e}")
        return

    thread_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    user_id = "python_example_user"

    print(f"\nThread ID: {thread_id}")
    print(f"Session ID: {session_id}")

    messages = [
        "Hello! Can you help me with some math?",
        "What is 15 * 24?",
        "Can you explain how you calculated that?",
    ]

    for i, message in enumerate(messages, 1):
        print(f"\n{'=' * 50}")
        print(f"Message {i}: {message}")
        print(f"{'=' * 50}")

        print("\nStreaming Response:")
        full_response = ""
        message_count = 0

        try:
            async for event in client.stream_chat(
                message=message,
                thread_id=thread_id,
                session_id=session_id,
                user_id=user_id,
                stream_tokens=True,
            ):
                event_type = event.get("type")
                content = event.get("content")

                if event_type == "token":
                    print(content, end="", flush=True)
                    full_response += content

                elif event_type == "message":
                    message_count += 1
                    msg_type = (
                        content.get("type", "unknown")
                        if isinstance(content, dict)
                        else "unknown"
                    )

                    if msg_type == "tool":
                        tool_id = content.get("tool_call_id", "unknown")
                        tool_content = content.get("content", "")
                        print(f"\n  Tool Result [{tool_id}]: {tool_content}")
                    elif msg_type == "ai" and content.get("tool_calls"):
                        tool_calls = content.get("tool_calls", [])
                        print(f"\n  Tool Calls: {len(tool_calls)} tools invoked")
                        for tool_call in tool_calls:
                            print(
                                f"   - {tool_call.get('name', 'unknown')}: {tool_call.get('args', {})}"
                            )

                elif event_type == "deep_research_status":
                    display_text = (
                        content.get("display_text", "")
                        if isinstance(content, dict)
                        else ""
                    )
                    if display_text:
                        print(f"\n  [Research] {display_text}")

                elif event_type == "error":
                    error_msg = (
                        content.get("message", "Unknown error")
                        if isinstance(content, dict)
                        else str(content)
                    )
                    print(f"\nError: {error_msg}")

        except Exception as e:
            print(f"\nStream Error: {e}")
            continue

        print(f"\n\nSummary: {len(full_response)} chars, {message_count} messages")

        if i < len(messages):
            await asyncio.sleep(2)


async def example_deep_research_chat():
    """Example of deep research mode with progress tracking."""
    print("\n\n=== Deep Research Example ===")
    print("=" * 50)

    client = TemplateAgentClient()

    try:
        health = await client.check_health()
        print(f"API Status: {health.get('status', 'unknown')}")
    except Exception as e:
        print(f"API Health Check Failed: {e}")
        return

    thread_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    user_id = "python_example_user"

    print(f"Thread ID: {thread_id}")
    print("Deep Research: ENABLED")

    query = "Tell me about Red Hat and its acquisition by IBM"
    print(f"\nQuery: {query}")
    print("-" * 50)

    full_response = ""
    dr_event_count = 0

    try:
        async for event in client.stream_chat(
            message=query,
            thread_id=thread_id,
            session_id=session_id,
            user_id=user_id,
            stream_tokens=False,
            deep_research_enabled=True,
            deep_research_require_plan_approval=False,
            timeout=600,
        ):
            event_type = event.get("type")
            content = event.get("content")

            if event_type == "deep_research_status" and isinstance(content, dict):
                dr_event_count += 1
                display_text = content.get("display_text", "")
                event_subtype = content.get("event_type", "")
                ui_visible = content.get("ui_visible", False)

                if ui_visible and display_text:
                    print(f"  [{event_subtype}] {display_text}")

                if event_subtype == "final_answer":
                    final = content.get("details", {}).get("final_answer", "")
                    if final:
                        full_response = final

            elif event_type == "message" and isinstance(content, dict):
                if content.get("type") == "ai" and content.get("content"):
                    if not full_response:
                        full_response = content["content"]

            elif event_type == "error":
                error_msg = (
                    content.get("message", "Unknown error")
                    if isinstance(content, dict)
                    else str(content)
                )
                print(f"\nError: {error_msg}")

    except Exception as e:
        print(f"\nStream Error: {e}")
        return

    print(f"\n{'=' * 50}")
    print(f"Deep research events: {dr_event_count}")
    print(f"Final answer: {len(full_response)} chars")
    if full_response:
        print("\n--- Answer Preview (first 500 chars) ---")
        print(full_response[:500])
        if len(full_response) > 500:
            print("...")

    # Follow-up query on the same thread to test triage/context-answer
    print(f"\n\n{'=' * 50}")
    follow_up = "Who founded Red Hat?"
    print(f"Follow-up Query (same thread): {follow_up}")
    print("-" * 50)

    dr_event_count = 0
    follow_up_response = ""

    try:
        async for event in client.stream_chat(
            message=follow_up,
            thread_id=thread_id,
            session_id=session_id,
            user_id=user_id,
            stream_tokens=False,
            deep_research_enabled=True,
            deep_research_require_plan_approval=False,
            timeout=600,
        ):
            event_type = event.get("type")
            content = event.get("content")

            if event_type == "deep_research_status" and isinstance(content, dict):
                dr_event_count += 1
                display_text = content.get("display_text", "")
                event_subtype = content.get("event_type", "")
                ui_visible = content.get("ui_visible", False)

                if ui_visible and display_text:
                    print(f"  [{event_subtype}] {display_text}")

                if event_subtype == "final_answer":
                    final = content.get("details", {}).get("final_answer", "")
                    if final:
                        follow_up_response = final

            elif event_type == "message" and isinstance(content, dict):
                if content.get("type") == "ai" and content.get("content"):
                    if not follow_up_response:
                        follow_up_response = content["content"]

    except Exception as e:
        print(f"\nStream Error: {e}")
        return

    print(f"\nFollow-up events: {dr_event_count}")
    print(f"Follow-up answer: {len(follow_up_response)} chars")
    if follow_up_response:
        print("\n--- Follow-up Answer Preview (first 500 chars) ---")
        print(follow_up_response[:500])
        if len(follow_up_response) > 500:
            print("...")


async def main():
    """Run all examples."""
    try:
        await example_streaming_chat()
        await example_deep_research_chat()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
