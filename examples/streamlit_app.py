"""Streamlit Demo App for Template Agent.

This application demonstrates how to integrate with the Template Agent's
simplified streaming API in a Streamlit application. It provides a clean
chat interface with real-time token streaming, deep research mode with
progress tracking, and message handling.

To run this app:
    streamlit run examples/streamlit_app.py

Make sure the Template Agent server is running on http://localhost:5002
"""

import json
import uuid
from typing import Any, Dict, List

import requests
import streamlit as st


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "user_id" not in st.session_state:
        st.session_state.user_id = "streamlit_user"


def stream_agent_response(
    message: str,
    thread_id: str,
    session_id: str,
    user_id: str,
    stream_tokens: bool = True,
    api_url: str = "http://localhost:5002",
    deep_research_enabled: bool = False,
    deep_research_require_plan_approval: bool = False,
) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stream response from the Template Agent using the simplified API.

    Args:
        message: User's input message
        thread_id: Conversation thread identifier
        session_id: Session identifier
        user_id: User identifier
        stream_tokens: Whether to stream individual tokens
        api_url: Base URL of the Template Agent API
        deep_research_enabled: Enable deep research pipeline
        deep_research_require_plan_approval: Require plan approval

    Returns:
        Tuple of (final_response, all_messages, deep_research_events)
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

    full_response = ""
    all_messages: List[Dict[str, Any]] = []
    dr_events: List[Dict[str, Any]] = []

    timeout = 600 if deep_research_enabled else 60

    try:
        response = requests.post(
            f"{api_url}/v1/stream",
            json=request_data,
            stream=True,
            timeout=timeout,
            headers={"Accept": "text/event-stream"},
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line.strip():
                continue

            if line.strip() == "[DONE]":
                break

            try:
                event = json.loads(line)
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
                    event_subtype = content.get("event_type", "")
                    if event_subtype == "final_answer":
                        final = content.get("details", {}).get("final_answer", "")
                        if final and not full_response:
                            full_response = final

                elif event_type == "error":
                    error_msg = (
                        content.get("message", "Unknown error")
                        if isinstance(content, dict)
                        else str(content)
                    )
                    st.error(f"Agent Error: {error_msg}")
                    break

            except json.JSONDecodeError:
                continue

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to agent: {e}")
        return "", [], []

    return full_response, all_messages, dr_events


def display_message(message: Dict[str, Any], role: str):
    """Display a message in the chat interface."""
    with st.chat_message(role):
        content = message.get("content", "")

        if content:
            st.write(content)

        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            with st.expander("Tool Calls", expanded=False):
                for tool_call in tool_calls:
                    st.json(
                        {
                            "tool": tool_call.get("name", "unknown"),
                            "args": tool_call.get("args", {}),
                            "id": tool_call.get("id", ""),
                        }
                    )

        metadata = message.get("response_metadata", {})
        if metadata:
            with st.expander("Metadata", expanded=False):
                st.json(metadata)


def display_deep_research_progress(dr_events: List[Dict[str, Any]]):
    """Display deep research progress in an expander."""
    if not dr_events:
        return

    visible = [e for e in dr_events if e.get("ui_visible", False)]
    if not visible:
        return

    with st.expander(f"Deep Research Progress ({len(visible)} events)", expanded=False):
        for evt in visible:
            event_subtype = evt.get("event_type", "")
            display_text = evt.get("display_text", "")

            if event_subtype == "started":
                st.info(display_text)
            elif event_subtype in ("completed", "final_answer"):
                st.success(display_text[:200])
            elif event_subtype == "error":
                st.error(display_text)
            elif "complete" in event_subtype:
                st.success(display_text)
            elif event_subtype.startswith("subquery"):
                st.write(f"  {display_text}")
            else:
                st.write(display_text)


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Template Agent Chat", page_icon="*", layout="wide")

    st.title("Template Agent Chat")
    st.markdown("Chat with the Template Agent using the simplified streaming API")

    initialize_session_state()

    with st.sidebar:
        st.header("Configuration")

        api_url = st.text_input(
            "API URL",
            value="http://localhost:5002",
            help="Base URL of the Template Agent API",
        )

        stream_tokens = st.checkbox(
            "Stream Tokens",
            value=True,
            help="Enable real-time token streaming for faster response display",
        )

        st.divider()

        st.subheader("Deep Research")
        deep_research_enabled = st.toggle(
            "Deep Research Mode",
            value=False,
            help="Enable multi-phase deep research pipeline for complex queries",
        )
        if deep_research_enabled:
            st.caption(
                "Deep research uses multiple agents to research your query "
                "in depth. Responses take longer but are more comprehensive."
            )

        st.divider()

        st.subheader("Session Info")
        st.text(f"Thread ID: {st.session_state.thread_id[:8]}...")
        st.text(f"Session ID: {st.session_state.session_id[:8]}...")
        st.text(f"User ID: {st.session_state.user_id}")

        if st.button("New Conversation"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        st.divider()

        st.subheader("API Status")
        try:
            health_response = requests.get(f"{api_url}/health", timeout=5)
            if health_response.status_code == 200:
                st.success("API Connected")
            else:
                st.error(f"API Error: {health_response.status_code}")
        except Exception as e:
            st.error(f"API Unreachable: {e}")

    st.subheader("Chat")

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            if message.get("dr_events"):
                display_deep_research_progress(message["dr_events"])
            display_message(message["content"], "assistant")

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            spinner_text = (
                "Deep research in progress (this may take a few minutes)..."
                if deep_research_enabled
                else "Agent is thinking..."
            )

            with st.spinner(spinner_text):
                full_response, all_messages, dr_events = stream_agent_response(
                    message=prompt,
                    thread_id=st.session_state.thread_id,
                    session_id=st.session_state.session_id,
                    user_id=st.session_state.user_id,
                    stream_tokens=stream_tokens and not deep_research_enabled,
                    api_url=api_url,
                    deep_research_enabled=deep_research_enabled,
                )

            if dr_events:
                display_deep_research_progress(dr_events)

            if full_response:
                response_placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": {
                            "type": "ai",
                            "content": full_response,
                            "messages": all_messages,
                        },
                        "dr_events": dr_events if dr_events else None,
                    }
                )
            else:
                response_placeholder.error("No response received from agent")

    with st.expander("Advanced Features", expanded=False):
        st.subheader("Raw Session Data")

        col1, col2 = st.columns(2)

        with col1:
            st.text("Session State:")
            st.json(
                {
                    "thread_id": st.session_state.thread_id,
                    "session_id": st.session_state.session_id,
                    "user_id": st.session_state.user_id,
                    "message_count": len(st.session_state.messages),
                    "deep_research_enabled": deep_research_enabled,
                }
            )

        with col2:
            st.text("Last Message Details:")
            if st.session_state.messages:
                st.json(st.session_state.messages[-1])

        if st.button("Export Conversation"):
            conversation_data = {
                "thread_id": st.session_state.thread_id,
                "session_id": st.session_state.session_id,
                "user_id": st.session_state.user_id,
                "messages": st.session_state.messages,
                "export_timestamp": str(uuid.uuid4()),
            }

            st.download_button(
                label="Download Conversation JSON",
                data=json.dumps(conversation_data, indent=2),
                file_name=f"conversation_{st.session_state.thread_id[:8]}.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
