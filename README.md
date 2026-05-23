# Template Agent

[![Python 3.12+](https://img.shields.io/badge/python-3.12,3.13-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/redhat-data-and-ai/template-agent/actions/workflows/test.yml/badge.svg)](https://github.com/redhat-data-and-ai/template-mcp-server/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A template for building AI agents with SSE streaming, conversation management, and Langfuse tracing.

## Quick Start

**Prerequisites:** Python 3.12+, PostgreSQL, Google AI API credentials

```bash
git clone https://github.com/redhat-data-and-ai/template-agent.git
cd template-agent
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env          # edit with your config
uv run python -m deep_agent.src.main
```

**MCP Server Required:** The agent needs an MCP server for BMI calculations and email features.

- **For Testing:** Use the included Mock MCP Server: `make mock-mcp` (see [SETUP.md](./SETUP.md))
- **For Production:** Clone and run [template-mcp-server](https://github.com/redhat-data-and-ai/template-mcp-server)

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/v1/stream` | POST | Stream chat (SSE) |
| `/v1/users/{user_id}/history/{thread_id}` | GET | Conversation history |
| `/v1/users/{user_id}/threads` | GET | List user threads |
| `/v1/feedback` | POST | Record feedback |

```bash
curl -X POST http://localhost:5002/v1/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "thread_id": "t1", "user_id": "u1", "stream_tokens": true}'
```

Client examples (Streamlit, Python async) are in [`examples/`](./examples/).

## Configuration

| Variable | Default | Required |
|---|---|---|
| `POSTGRES_USER` | `pgvector` | Yes |
| `POSTGRES_PASSWORD` | `pgvector` | Yes |
| `POSTGRES_DB` | `pgvector` | Yes |
| `POSTGRES_HOST` | `pgvector` | Yes |
| `POSTGRES_PORT` | `5432` | Yes |
| `REDIS_URL` | `redis://redis:6379/0` | No |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | — | Yes |
| `LANGFUSE_PUBLIC_KEY` | — | No |
| `LANGFUSE_SECRET_KEY` | — | No |
| `LANGFUSE_BASE_URL` | — | No |
| `LANGFUSE_TRACING_ENVIRONMENT` | `development` | No |
| `SSL_KEYFILE` | — | No |
| `SSL_CERTFILE` | — | No |

Runtime configuration (cache TTLs, memory settings, agent identity) is in `config/agent/runtime/agent.yaml`.

## Project Structure

```
deep_agent/
├── src/
│   ├── core/
│   │   ├── agent.py          # Agent initialisation + subagent loading
│   │   ├── manager.py        # AgentManager, SSE streaming
│   │   ├── prompt.py         # System prompt loader
│   │   └── backend.py        # Shell backend (isolated venv)
│   ├── routes/
│   │   ├── stream.py         # POST /v1/stream
│   │   ├── history.py        # GET  /v1/users/{user_id}/history/{thread_id}
│   │   ├── threads.py        # GET  /v1/users/{user_id}/threads
│   │   ├── health.py         # GET  /health
│   │   └── feedback.py       # POST /v1/feedback
│   ├── api.py                # FastAPI app + lifespan
│   ├── main.py               # Uvicorn entry point
│   ├── schema.py             # Pydantic models
│   └── settings.py           # Env config
├── agent_config/
│   ├── orchestrator/         # Main agent configuration
│   │   └── main.md           # Orchestrator prompt (YAML + MD)
│   ├── subagents/            # Subagent definitions (YAML + MD)
│   └── skills/               # Skill documents per agent
└── tests/
    ├── agents/               # Agent-level tests (LLM-as-judge)
    ├── core/                 # Core unit tests
    └── routes/               # API route tests
```

## Testing & Quality

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=deep_agent.src --cov-report=html

# Agent-level tests only
pytest tests/agents/ -v

# Specific agent tests
pytest tests/agents/test_analyst.py -m analyst -v
pytest tests/agents/test_publisher.py -m publisher -v
pytest tests/agents/test_orchestrator.py -m orchestrator -v

# Single evaluation
pytest tests/agents/test_analyst.py -m analyst -k "eval-1" -v

# Code quality
ruff check . && ruff format .
pre-commit run --all-files
```

### Agent Tests

Tests are organized by agent (orchestrator + subagents), each with its associated skill and tools:

**Analyst** (`test_analyst.py`)
- Skill: `bmi-report`
- Tools: `calculate_bmi`, `search_web`
- Evals: `deep_agent/agent_config/skills/bmi-report/evals/evals.json`

**Publisher** (`test_publisher.py`)
- Skill: `email-formatter`
- Tools: `send_email`
- Evals: `deep_agent/agent_config/skills/email-formatter/evals/evals.json`

**Orchestrator** (`test_orchestrator.py`)
- Skill: `client-intake`
- Subagents: Analyst + Publisher
- Evals: `deep_agent/agent_config/skills/client-intake/evals/evals.json`

Test results saved to `tests/workspaces/{agent}-workspace/eval-{id}/`:
- `outputs/report.md` - Agent output
- `outputs/grading.json` - LLM judge evaluation results

## Deployment

### Custom FastAPI Server (default)

```bash
podman-compose up -d --build
```

For production: configure SSL certs (`AGENT_SSL_KEYFILE`, `AGENT_SSL_CERTFILE`), use managed PostgreSQL, and enable Langfuse tracing.

### Aegra / LangGraph Platform (with deep-agents-ui)

Serve the agent via the standard [LangGraph Platform](https://docs.langchain.com/langsmith/cli) API.
Compatible with [deep-agents-ui](https://github.com/langchain-ai/deep-agents-ui).

**Option A — Local dev (no Docker):**

```bash
# Terminal 1: Start the mock MCP server
make mock-mcp

# Terminal 2: Start LangGraph dev server
pip install "langgraph-cli[inmem]"
make aegra-dev

# Terminal 3: Start the UI
make aegra-clone-ui
make aegra-ui
```

Open http://localhost:3000, set **Deployment URL** = `http://127.0.0.1:2024`, **Assistant ID** = `agent`.

**Option B — Docker Compose:**

```bash
# Clone the UI
make aegra-clone-ui

# Start everything via LangGraph CLI
langgraph up -d compose.yaml --port 2024 --wait
```

**Option C — Build a standalone Docker image:**

```bash
make aegra-build
docker run -p 2024:8123 --env-file .env template-agent-aegra
```

#### Aegra directory structure

```
deep_agent/
└── aegra/
    ├── __init__.py       # Package metadata
    ├── graph.py          # Graph builder + exported agent (aegra.json entry)
    ├── state.py          # Extended state schema and health types
    ├── converters.py     # State conversion utilities
    └── nodes.py          # Error-handling node decorators
aegra.json                # Aegra Platform configuration
```

## Links

- [Issues](https://github.com/redhat-data-and-ai/template-agent/issues)
- [template-mcp-server](https://github.com/redhat-data-and-ai/template-mcp-server)
- [deep-agents-ui](https://github.com/langchain-ai/deep-agents-ui)
- [LangGraph Platform docs](https://docs.langchain.com/langsmith/cli)
