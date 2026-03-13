# Template Agent Client Examples

This directory contains client examples demonstrating how to interact with the Template Agent's simplified streaming API. These examples show best practices for handling real-time streaming, different event types, deep research mode, and error scenarios.

## Available Examples

### 1. Streamlit Demo App (`streamlit_app.py`)

A full-featured chat application built with Streamlit:
- **Real-time chat interface** with message history
- **Deep Research toggle** in the sidebar for multi-phase research
- **Deep research progress** displayed as a collapsible event timeline
- **Token streaming visualization** for responsive UX
- **Session management** with thread and session persistence
- **Configuration panel** for API settings and debugging
- **Export functionality** for conversation data

**To Run:**
```bash
# Install Streamlit if not already installed
pip install streamlit requests

# Run the app
streamlit run examples/streamlit_app.py

# Open http://localhost:8501 in your browser
```

### 2. Python Async Client (`client_python.py`)

A robust async Python client for server-to-server communication:
- **Async/await support** using aiohttp
- **Deep research mode** with `deep_research_enabled` parameter
- **Streaming and non-streaming modes** for different use cases
- **Deep research event handling** with progress tracking
- **Follow-up query demonstration** showing triage/context-answer path
- **Comprehensive error handling** with detailed error messages
- **Session management** with automatic ID generation
- **Health checking** for API availability

**To Run:**
```bash
# Install dependencies
pip install aiohttp

# Run the example
python examples/client_python.py
```

**Usage as Library:**
```python
from examples.client_python import TemplateAgentClient

client = TemplateAgentClient()

# Simple message
response, messages, dr_events = await client.send_message("Hello!")

# Streaming chat
async for event in client.stream_chat("Hello!", "thread-123", "session-123", "user-123"):
    if event['type'] == 'token':
        print(event['content'], end='', flush=True)

# Deep research
async for event in client.stream_chat(
    "Tell me about Red Hat",
    "thread-123", "session-123", "user-123",
    deep_research_enabled=True,
    deep_research_require_plan_approval=False,
    timeout=600,
):
    if event['type'] == 'deep_research_status':
        print(event['content'].get('display_text', ''))
    elif event['type'] == 'message':
        content = event['content']
        if content.get('type') == 'ai':
            print(content['content'])
```

## API Reference

### Request Format

**Standard request:**

```json
{
  "message": "User's input message",
  "thread_id": "Conversation thread identifier",
  "session_id": "Session identifier",
  "user_id": "User identifier",
  "stream_tokens": true
}
```

**Deep research request:**

```json
{
  "message": "Tell me about Red Hat",
  "thread_id": "thread-123",
  "session_id": "session-123",
  "user_id": "user-456",
  "deep_research_enabled": true,
  "deep_research_require_plan_approval": false,
  "deep_research_model": "gemini-2.5-flash",
  "deep_research_max_subqueries": 10,
  "deep_research_max_mode": false
}
```

### Response Format

The API returns Server-Sent Events with this format:

```json
{"type": "message", "content": {"type": "ai", "content": "Hello"}}
{"type": "token", "content": " world"}
{"type": "deep_research_status", "content": {"stage": "started", "event_type": "started", "display_text": "Starting deep research...", "ui_visible": true, "details": {}}}
{"type": "error", "content": {"message": "Error occurred", "recoverable": false}}
[DONE]
```

**Event Types:**
| Type | Description |
|------|-------------|
| `message` | Complete messages (AI responses, tool calls, tool results) |
| `token` | Individual tokens for real-time streaming |
| `deep_research_status` | Deep research pipeline events (progress, findings, final answer) |
| `error` | Error messages with recovery information |
| `[DONE]` | Stream completion marker |

**Deep Research Event Sub-types** (in `content.event_type`):
| Event Type | Phase | Description |
|------------|-------|-------------|
| `started` | Init | Pipeline started |
| `agent_decision` | Various | Agent made a routing/analysis decision |
| `probe_complete` | Probe | Tool discovery finished |
| `plan_generated` | Plan | Research plan created |
| `subquery_start` | Research | Worker started investigating |
| `subquery_complete` | Research | Worker finished with findings |
| `synthesis_start` | Synthesis | Report generation started |
| `synthesis_complete` | Synthesis | Report ready |
| `review_start` | Review | Multi-persona review started |
| `review_complete` | Review | Review finished |
| `triage_decision` | Triage | Follow-up optimization decision |
| `final_answer` | Complete | Final research report |
| `completed` | Complete | Pipeline finished |
| `token_usage_update` | Complete | Token usage and cost summary |

## Getting Started

### Prerequisites

1. **Template Agent Server Running**
   ```bash
   cd template-agent
   python -m template_agent.src.main
   # Runs on http://localhost:5002
   ```

2. **Template MCP Server Running** (for tool access in deep research)
   ```bash
   cd template-mcp-server
   make local
   # Runs on http://localhost:5001
   ```

3. **Install Client Dependencies**
   ```bash
   pip install aiohttp requests streamlit
   ```

### Quick Test

Test the API is working:

```bash
# Health check
curl http://localhost:5002/health

# Standard streaming test
curl -X POST "http://localhost:5002/v1/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "message": "Hello!",
    "thread_id": "test-123",
    "session_id": "test-123",
    "user_id": "test-user",
    "stream_tokens": true
  }'

# Deep research test
curl -X POST "http://localhost:5002/v1/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about Red Hat",
    "thread_id": "test-dr-1",
    "session_id": "test",
    "user_id": "test-user",
    "deep_research_enabled": true,
    "deep_research_require_plan_approval": false
  }'
```

## Best Practices

### 1. Session Management
- Use consistent `thread_id` for multi-turn conversations
- Deep research follow-up queries on the same `thread_id` benefit from cached findings
- Use `session_id` to group related threads
- Generate UUIDs for unique identifiers

### 2. Error Handling
- Always handle `error` events in streams
- Check `recoverable` flag to determine retry logic
- Implement timeout and connection error handling

### 3. Token Streaming
- Set `stream_tokens: true` for real-time UX in standard mode
- Deep research mode sends `deep_research_status` events instead of tokens
- Buffer tokens appropriately for UI updates

### 4. Deep Research
- Use `deep_research_enabled: true` for complex, research-heavy queries
- Set `deep_research_require_plan_approval: false` for auto-execution
- Increase timeouts to 600s (deep research can take 1-5 minutes)
- Follow-up queries on the same thread use triage to answer from cache when possible

### 5. Performance
- Use appropriate timeouts for your use case (60s standard, 600s deep research)
- Handle stream interruption gracefully
- Consider connection pooling for high-volume usage

## Enterprise Features

All examples preserve enterprise features from the original implementation:

- **SSO Authentication**: Pass `X-Token` header for enterprise auth
- **Langfuse Tracing**: Automatic tracing and analytics
- **PostgreSQL Persistence**: Conversation history and checkpointing
- **Error Monitoring**: Comprehensive error logging and recovery

## Troubleshooting

### Common Issues

**Connection Refused**
- Ensure Template Agent server is running on http://localhost:5002
- Check firewall settings and port availability

**Deep Research Returns Empty/No Data**
- Ensure the MCP server is running on http://localhost:5001
- Verify `DEEP_RESEARCH_ENABLED=true` in the agent's `.env`
- Check that `TAVILY_API_KEY` is set in the MCP server's `.env` for web search

**Authentication Errors**
- Verify SSO token is valid (if using enterprise features)
- Check X-Token header format

**Streaming Issues**
- Ensure `Accept: text/event-stream` header is set
- Check for proxy/firewall interference with streaming
- Verify timeout settings are appropriate (600s for deep research)

**Token Streaming Not Working**
- Confirm `stream_tokens: true` in request
- Deep research does not use token streaming; it uses `deep_research_status` events
- Check for buffering issues in HTTP clients

### Debug Mode

Enable detailed logging in examples:

```python
# Python examples
import logging
logging.basicConfig(level=logging.DEBUG)

# Streamlit
st.set_option('client.showErrorDetails', True)
```

For more help, check the main project documentation or create an issue in the repository.
