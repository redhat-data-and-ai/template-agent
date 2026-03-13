# Template Agent

[![Python 3.12+](https://img.shields.io/badge/python-3.12,3.13-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/redhat-data-and-ai/template-agent/actions/workflows/test.yml/badge.svg)](https://github.com/redhat-data-and-ai/template-mcp-server/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/redhat-data-and-ai/template-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/redhat-data-and-ai/template-mcp-server)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A production-ready template for building AI agents with streaming capabilities, conversation management, and enterprise-grade features. Includes a generic **Deep Research** pipeline for multi-phase, multi-agent research with parallel workers, triage-based follow-up optimization, and Langfuse tracing.

## Features

- **Simplified Streaming API**: Clean, consistent event format for easy client integration
- **Real-time Streaming**: Server-Sent Events (SSE) with token and message streaming
- **Deep Research Mode**: Multi-phase research pipeline with planning, parallel worker execution, multi-persona review, and synthesis -- toggle on/off per request
- **Follow-up Optimization**: Triage node detects when cached findings can answer a follow-up query, skipping full research via a fast context-answer path
- **Multiple Client Examples**: Python async and Streamlit demo applications with deep research support
- **Conversation Management**: Multi-turn conversations with thread persistence
- **Enterprise Integration**: Langfuse tracing (graph-level and worker-level), PostgreSQL checkpointing, SSO support
- **Modular Architecture**: AgentManager abstraction with clean separation of concerns
- **Production Ready**: Health checks, error handling, cancellation support, and comprehensive logging
- **Google AI Integration**: Built-in support for Google Generative AI models (Gemini 2.5 Flash / Pro)

## Architecture

```mermaid
graph TB
    subgraph clients [Clients]
        UI[Web UI]
        API[API Client]
    end

    subgraph agent [Template Agent]
        subgraph apiLayer [API Layer]
            Health[Health Check]
            Stream[Stream Chat]
            History[Chat History]
            Threads[Thread Management]
            Feedback[Feedback]
        end

        subgraph coreLayer [Core Layer]
            Agent[Agent Engine]
            Manager[AgentManager]
            Utils[Message Utils]
            Prompt[Prompt Management]
        end

        subgraph deepResearch [Deep Research Pipeline]
            Router[Router]
            Complexity[Assess Complexity]
            Triage[Triage]
            ContextAnswer[Context Answer]
            Probe[Probe]
            Plan[Plan]
            Supervisor[Supervisor]
            Completeness[Completeness]
            Synthesize[Synthesize]
            Visualize[Visualize]
            Review[Review]
            Complete[Complete]
        end

        subgraph dataLayer [Data Layer]
            DB[(PostgreSQL)]
            Langfuse[Langfuse]
        end
    end

    subgraph external [External Services]
        Google[Google AI]
        MCP[MCP Server]
        SSO[SSO Auth]
    end

    UI --> Stream
    UI --> Health
    API --> Stream
    API --> Health

    Stream --> Manager
    Manager --> Agent
    Manager --> Router

    Router --> Complexity
    Complexity --> Triage
    Triage -->|context_sufficient| ContextAnswer
    Triage -->|full_research| Probe
    ContextAnswer --> Review
    Probe --> Plan
    Plan --> Supervisor
    Supervisor --> Completeness
    Completeness --> Synthesize
    Synthesize --> Visualize
    Visualize --> Review
    Review --> Complete

    Supervisor --> MCP
    Agent --> MCP
    Agent --> Google
    Supervisor --> Google
    Agent --> Langfuse
    Manager --> Langfuse
    Agent --> DB
```

## Streaming API

### Single Streaming Endpoint

```http
POST /v1/stream
Content-Type: application/json
Accept: text/event-stream
```

### Standard Request

```json
{
  "message": "User input message",
  "thread_id": "conversation-thread-id",
  "session_id": "session-id",
  "user_id": "user-identifier",
  "stream_tokens": true
}
```

### Deep Research Request

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

### Response Events

**Standard mode:**
```
{"type": "token", "content": "Hello"}
{"type": "token", "content": " world"}
{"type": "message", "content": {"type": "ai", "content": "Hello world"}}
[DONE]
```

**Deep research mode:**
```
{"type": "deep_research_status", "content": {"stage": "started", "event_type": "started", "display_text": "Starting deep research analysis...", "ui_visible": true, "details": {}}}
{"type": "deep_research_status", "content": {"stage": "agent_conversation", "event_type": "subquery_complete", "display_text": "Worker 1/7: ...", "ui_visible": true, "details": {...}}}
{"type": "deep_research_status", "content": {"stage": "completed", "event_type": "final_answer", "display_text": "Answer: ...", "ui_visible": true, "details": {"final_answer": "..."}}}
{"type": "message", "content": {"type": "ai", "content": "# Research Report\n..."}}
[DONE]
```

### Client Examples

Ready-to-use client examples are available in the [`examples/`](./examples/) directory:

- **[Streamlit Demo App](./examples/streamlit_app.py)** - Interactive chat with deep research support
- **[Python Async Client](./examples/client_python.py)** - Server-to-server integration with deep research

See the [examples README](./examples/README.md) for detailed usage instructions.

## Quick Start

### Prerequisites

- Python 3.12+
- Google AI API credentials
- Template MCP Server running (for tool access)
- PostgreSQL database (optional -- in-memory mode available)
- Langfuse account (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/redhat-data-and-ai/template-agent.git
   cd template-agent
   ```

2. **Create virtual environment**
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   uv pip install -e ".[dev]"
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run template-mcp-server** following https://github.com/redhat-data-and-ai/template-mcp-server

6. **Run the application**
   ```bash
   uv run python -m template_agent.src.main
   ```

## API Reference

### Endpoints

| Endpoint                  | Method | Description |
|---------------------------|--------|-------------|
| `/health`                 | GET | Health check |
| `/v1/stream`              | POST | Stream chat responses (standard and deep research) |
| `/v1/history/{thread_id}` | GET | Get conversation history |
| `/v1/threads/{user_id}`   | GET | List user threads |
| `/v1/feedback`            | POST | Record feedback |

### Standard Chat

```bash
curl -X POST "http://localhost:5002/v1/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how can you help me?",
    "thread_id": "thread_123",
    "user_id": "user_456",
    "stream_tokens": true
  }'
```

### Deep Research Chat

```bash
curl -X POST "http://localhost:5002/v1/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about Red Hat",
    "thread_id": "thread_123",
    "user_id": "user_456",
    "deep_research_enabled": true,
    "deep_research_require_plan_approval": false
  }'
```

### Health Check

```bash
curl "http://localhost:5002/health"
# Response: {"status": "healthy", "service": "Template Agent"}
```

## Deep Research Pipeline

The deep research mode executes a multi-phase LangGraph pipeline:

| Phase | Node | Description |
|-------|------|-------------|
| Routing | `router` | Validates the query and determines if deep research is needed |
| Complexity | `assess_complexity` | Classifies query complexity (simple/moderate/complex) and sets iteration bounds |
| Triage | `triage` | For follow-up queries: checks if cached findings can answer without new research |
| Context Answer | `context_answer` | Fast-path synthesis from cached findings (skips probe/plan/supervisor) |
| Probing | `probe` | Discovers available MCP tools and tests their capabilities |
| Planning | `plan` | Generates a research plan with subqueries based on tool capabilities |
| Research | `supervisor` | Orchestrates parallel workers via `asyncio.Semaphore` / `create_task` |
| Completeness | `completeness` | Evaluates research coverage and decides if more rounds are needed |
| Synthesis | `synthesize` | Aggregates findings into a structured report |
| Visualization | `visualize` | Generates charts/tables (optional) |
| Review | `review` | Multi-persona quality review (Factual Skeptic, User Advocate, Numerical Auditor, etc.) |
| Completion | `complete` | Saves findings to cache, emits final answer |

### Key Features

- **Parallel workers**: Subqueries execute concurrently with configurable concurrency limits
- **Follow-up optimization**: Triage detects when prior findings answer a follow-up, routing to a fast context-answer path
- **In-memory findings cache**: Findings persist across requests on the same thread within the process lifetime
- **Langfuse tracing**: Graph-level, worker-level, and node-level LLM call tracing
- **Cancellation support**: Active research sessions can be cancelled mid-flight
- **Token tracking**: Per-phase token usage and cost estimation

## Configuration

### Environment Variables

#### Server
| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_HOST` | `0.0.0.0` | Server bind address |
| `AGENT_PORT` | `5002` | Server port |
| `PYTHON_LOG_LEVEL` | `INFO` | Logging level |
| `USE_INMEMORY_SAVER` | `false` | Use in-memory checkpointer (no PostgreSQL needed) |

#### Database
| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `pgvector` | Database username |
| `POSTGRES_PASSWORD` | `pgvector` | Database password |
| `POSTGRES_DB` | `pgvector` | Database name |
| `POSTGRES_HOST` | `pgvector` | Database host |
| `POSTGRES_PORT` | `5432` | Database port |

#### MCP Server
| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVER_NAME` | `template-mcp-server` | MCP server identifier |
| `MCP_SERVER_URL` | `http://localhost:5001/mcp/` | MCP server endpoint |
| `MCP_TRANSPORT_PROTOCOL` | `streamable_http` | Transport protocol |
| `MCP_CONNECTION_TIMEOUT` | `30` | Connection timeout in seconds |
| `MCP_SSL_VERIFY` | `false` | Enable SSL verification for MCP |

#### Deep Research
| Variable | Default | Description |
|----------|---------|-------------|
| `DEEP_RESEARCH_ENABLED` | `true` | Enable deep research pipeline |
| `DEEP_RESEARCH_DEFAULT_MODEL` | `gemini-2.5-flash` | Default model for deep research LLM calls |
| `DEEP_RESEARCH_MAX_SUBQUERIES` | `10` | Maximum number of research subqueries |
| `DEEP_RESEARCH_MAX_TOOLS` | `50` | Maximum tools to include in prompts |
| `DEEP_RESEARCH_LLM_CALL_TIMEOUT_SECONDS` | `120` | Timeout for individual LLM calls |
| `DEEP_RESEARCH_REQUIRE_PLAN_APPROVAL` | `true` | Require user approval before executing plan |
| `DEEP_RESEARCH_ENABLE_VISUALIZATION` | `true` | Enable visualization node |
| `DEEP_RESEARCH_MAX_SESSION_SECONDS` | `600` | Maximum session duration (10 min) |
| `DEEP_RESEARCH_LLM_CONCURRENCY` | `5` | Maximum concurrent LLM calls |
| `LLM_INPUT_COST_PER_MILLION` | `1.25` | Cost per 1M input tokens (USD) |
| `LLM_OUTPUT_COST_PER_MILLION` | `10.0` | Cost per 1M output tokens (USD) |

#### Langfuse (optional)
| Variable | Default | Description |
|----------|---------|-------------|
| `LANGFUSE_PUBLIC_KEY` | - | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | - | Langfuse secret key |
| `LANGFUSE_BASE_URL` | - | Langfuse host URL |
| `LANGFUSE_TRACING_ENVIRONMENT` | `development` | Langfuse environment |

#### Google AI
| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_SERVICE_ACCOUNT_FILE` | - | Path to service account JSON |
| `GOOGLE_APPLICATION_CREDENTIALS_CONTENT` | - | Inline service account JSON |

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=template_agent.src --cov-report=html

# Run specific test file
pytest tests/test_prompt.py -v
```

## Deployment

### Podman Compose

```bash
podman-compose up -d --build
```

### Production Considerations

- **SSL/TLS**: Configure SSL certificates for HTTPS
- **Database**: Use managed PostgreSQL service
- **Monitoring**: Set up Langfuse for tracing
- **Scaling**: Configure horizontal pod autoscaling
- **Security**: Implement proper authentication

## Development

### Project Structure

```
template-agent/
├── template_agent/
│   ├── src/
│   │   ├── core/
│   │   │   ├── agent.py            # Agent initialization
│   │   │   ├── agent_utils.py      # Message utilities
│   │   │   ├── manager.py          # AgentManager (routes standard / deep research)
│   │   │   ├── prompt.py           # Prompt management
│   │   │   └── deep_research/      # Deep research pipeline
│   │   │       ├── streaming.py    # Graph builder & streaming orchestration
│   │   │       ├── agents.py       # ResearchContext factory & worker execution
│   │   │       ├── state.py        # State schema, ResearchContext dataclass
│   │   │       ├── events.py       # SSE event emitters
│   │   │       ├── prompts.py      # LLM prompt templates
│   │   │       ├── mode_config.py  # Model/mode configuration
│   │   │       ├── sentinel.py     # Loop circuit breaker
│   │   │       ├── cancel.py       # Cancellation store
│   │   │       ├── context_manager.py  # Hierarchical context window
│   │   │       ├── plan_store.py   # Plan persistence
│   │   │       ├── findings_store.py   # Cross-chat findings
│   │   │       ├── token_tracker.py    # Token usage re-export
│   │   │       ├── utils.py        # Shared utilities
│   │   │       └── nodes/          # Pipeline nodes
│   │   │           ├── probe.py
│   │   │           ├── triage.py
│   │   │           ├── complexity.py
│   │   │           ├── context_answer.py
│   │   │           ├── plan.py
│   │   │           ├── supervisor.py
│   │   │           ├── completeness.py
│   │   │           ├── synthesize.py
│   │   │           ├── visualize.py
│   │   │           ├── review.py
│   │   │           └── complete.py
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   ├── stream.py
│   │   │   ├── history.py
│   │   │   ├── threads.py
│   │   │   └── feedback.py
│   │   ├── api.py
│   │   ├── main.py
│   │   ├── schema.py
│   │   └── settings.py
│   └── utils/
│       ├── tracing.py              # Token tracking & tracked_invoke
│       └── pylogger.py             # Structured logging
├── examples/
│   ├── client_python.py
│   ├── streamlit_app.py
│   └── README.md
├── tests/
└── README.md
```

### Code Quality

```bash
# Run linting
ruff check .

# Run type checking
mypy template_agent/src/

# Run formatting
ruff format .

# Run pre-commit hooks
pre-commit run --all-files
```

## Related Projects

- [Template MCP Server](https://github.com/redhat-data-and-ai/template-mcp-server) - MCP tools server
- [Template UI](https://github.com/redhat-data-and-ai/template-ui) - React frontend
- [LangGraph](https://github.com/langchain-ai/langgraph) - Stateful LLM applications
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Langfuse](https://langfuse.com/) - LLM observability platform

---

**Built with care by the Red Hat Data & AI team**
