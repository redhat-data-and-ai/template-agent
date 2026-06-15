# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a template for building AI agents with SSE streaming, conversation management, and Langfuse tracing. It runs on the [deepagents](https://github.com/langchain-ai/deepagents) framework (version 0.4.12) and can be deployed either as:
- A custom FastAPI server (default)
- An Aegra/LangGraph Platform deployment (compatible with deep-agents-ui)

The template demonstrates a **multi-agent orchestration pattern** where an orchestrator delegates to specialized subagents (analyst, publisher) with skill-based execution.

## Development Commands

### Environment Setup
```bash
# Install dependencies (creates .venv and activates it)
make install

# Create .env from example
cp .env.example .env
# Edit .env with your credentials (Postgres, Google AI, optional Langfuse)
```

### Running the Agent

**Option A: Local with LangGraph Platform (recommended for development)**
```bash
# Terminal 1: Start Mock MCP Server
make mock-mcp

# Terminal 2: Start agent with Aegra dev server
make local
# API available at http://localhost:5002
```

**Option B: Full Docker Compose stack**
```bash
make dev
# Agent at http://localhost:5002
# Includes Postgres + Redis
```

**Option C: Full demo with UI + MCP Server + SSO auth**
```bash
make demo
# UI at http://localhost:8080 (SSO login)
# Agent at http://localhost:5002
# MCP Server at http://localhost:5001
```

### Testing

```bash
# All tests (unit + skills evals)
make test-all

# Unit tests only
make test

# Unit tests with coverage
make test-cov

# Skills evaluations only
make test-skills

# Specific agent evaluation
pytest tests/skills/test_analyst.py -m analyst -v
pytest tests/skills/test_publisher.py -m publisher -v
pytest tests/skills/test_orchestrator.py -m orchestrator -v

# Single evaluation
pytest tests/skills/test_analyst.py -m analyst -k "eval-1" -v
```

**Test workspace structure**: Results are saved to `tests/workspaces/{agent}-workspace/eval-{id}/`:
- `outputs/report.md` - Agent output
- `outputs/grading.json` - LLM judge evaluation

### Code Quality

```bash
# Run linter and formatter
ruff check . && ruff format .

# Run pre-commit hooks
pre-commit run --all-files

# Type checking (mypy configured in pyproject.toml)
mypy deep_agent/
```

### Deployment

**Kind (local Kubernetes)**
```bash
make kind      # Deploy full stack to local Kind cluster
make kind-down # Teardown
```

**OpenShift**
```bash
make deploy openshift NAMESPACE=your-project
make undeploy openshift NAMESPACE=your-project
```

## Architecture

### Directory Structure

```
deep_agent/
├── aegra/                      # Aegra/LangGraph Platform integration
│   ├── graph.py                # Async graph factory (per-request)
│   ├── auth.py                 # SSO auth handler (OIDC/JWT)
│   ├── mcp.py                  # MCP tool loading with SSO token forwarding
│   ├── startup.py              # Startup orchestrator (DB check, MCP init)
│   └── state.py                # Extended state schema
├── src/
│   ├── agent/                  # Agent configuration
│   ├── cache/                  # Redis caching (personalization, auth tokens)
│   ├── infrastructure/         # Providers, subagents, middleware, backend
│   ├── memory/                 # Postgres-backed memory persistence
│   ├── personalization/        # User-specific memories and rules
│   ├── streaming/              # SSE streaming utilities
│   ├── feedback/               # Feedback collection
│   └── settings.py             # Environment config (Pydantic BaseSettings)
├── utils/                      # Shared utilities (logging, Google creds)
└── __init__.py

config/
├── agent/
│   ├── PROMPT.md               # Orchestrator prompt (YAML frontmatter + Markdown)
│   ├── mcp.json                # MCP server registry
│   ├── runtime/
│   │   ├── agent.yaml          # Runtime config (model, middleware, caching)
│   │   └── secrets.example.yaml
│   ├── subagents/              # Subagent definitions (analyst.md, publisher.md)
│   └── skills/                 # Skill documents per agent
│       ├── client-intake/      # Orchestrator skill
│       ├── bmi-report/         # Analyst skill
│       └── email-formatter/    # Publisher skill
└── deployment/                 # UI config

tests/
├── skills/                     # Agent-level tests (LLM-as-judge)
├── unit/                       # Unit tests
└── workspaces/                 # Test execution workspaces
```

### Agent Configuration System

**Agent definitions** are in `config/agent/`:
- **Orchestrator**: `PROMPT.md` (YAML frontmatter + Markdown system prompt)
- **Subagents**: `subagents/{name}.md` (same format)
- **Skills**: `skills/{skill-name}/README.md` (loaded at runtime, injected into system prompt)

**YAML frontmatter fields** (in PROMPT.md / subagent .md files):
```yaml
---
name: orchestrator
description: Main coordinator for fitness assistant
model: gemini-2.5-pro           # Model name (resolved via providers config)
tools:                          # Built-in tools or MCP tool names
  - validate_email
skills:                         # Skills to load (from config/agent/skills/)
  - client-intake
mcps:                           # MCP servers to connect to (from mcp.json)
  - template-mcp-server
middleware:                     # Optional middleware overrides
  memory:
    enabled: true
---
```

**Runtime configuration** (`config/agent/runtime/agent.yaml`):
- Model resolution strategy (legacy vs deepagents)
- Provider configs (google_genai, anthropic_vertex, openai, vLLM)
- Harness profiles (per-model overrides: excluded_tools, excluded_middleware)
- Middleware pipeline (summarization, memory, PII redaction, retries, etc.)
- Caching (graph TTL, auth token TTL, personalization TTL)
- Async task settings

**Key design pattern**: Graph factory (`deep_agent/aegra/graph.py:agent`) is invoked **per-request** by Aegra. It:
1. Extracts SSO token from `ServerRuntime.user`
2. Loads orchestrator config from PROMPT.md
3. Injects personalization (user memories + rules) into system prompt
4. Loads MCP tools with SSO token forwarding
5. Builds subagents from subagent definitions
6. Compiles the deepagents graph
7. Caches the compiled graph (TTL from agent.yaml)

### Multi-Agent Pattern

**Orchestrator** (you) coordinates work:
- Delegates health analysis to **analyst** subagent (tools: `calculate_bmi`, `search_web`)
- Delegates email delivery to **publisher** subagent (tool: `send_email`)
- Validates email addresses using `validate_email` tool
- Creates TODO lists for task tracking

**Control flow**:
1. User request → Orchestrator creates TODO list
2. Orchestrator validates inputs (e.g., email addresses)
3. Orchestrator delegates to analyst subagent
4. Analyst completes analysis
5. Orchestrator delegates to publisher (if email requested)
6. Publisher sends email

**Critical constraint**: Publisher (step ②) is NEVER invoked until ALL other subagents complete.

### MCP Integration

**MCP servers** are defined in `config/agent/mcp.json`:
```json
{
  "mcpServers": {
    "template-mcp-server": {
      "url": "http://localhost:5001/mcp",
      "transport": "streamable_http",
      "enabled": true,
      "auth": true,
      "ssl_verify": false
    }
  }
}
```

**SSO token forwarding**: When a user is authenticated (Aegra deployment), their SSO access token is:
1. Extracted from `ServerRuntime.user.access_token`
2. Refreshed if needed (via `refresh_access_token`)
3. Set in MCP auth context (`set_mcp_auth_context`)
4. Forwarded to MCP servers in tool calls (Authorization header)

This enables **end-to-end auth**: User → UI → Agent → MCP Server (all using the same token).

### Testing Strategy

**Skill evaluations** (`tests/skills/`) use LLM-as-judge pattern:
- Each skill has `evals/evals.json` with test cases
- Test runner executes agent with test input
- LLM judge grades output against rubric
- Results saved to workspace directory

**Pytest markers**:
- `unit`: Fast isolated tests
- `integration`: Tests requiring external services
- `skills`: Skill evaluation tests
- `e2e`: End-to-end tests with running Aegra server

## Environment Variables

**Required**:
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT`: Database config
- `GOOGLE_SERVICE_ACCOUNT_FILE`: Path to GCP service account JSON (for Vertex AI)

**Optional**:
- `REDIS_URL`: Redis connection string (default: `redis://redis:6379/0`)
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`: Langfuse tracing
- `SSO_ISSUER_URL`, `SSO_CLIENT_ID`, `SSO_CLIENT_SECRET`: OIDC auth (for Aegra deployment)
- `VLLM_BASE_URL`, `VLLM_API_KEY`: vLLM endpoint (for custom models)

**Settings precedence**: Environment variables override `agent.yaml` defaults.

## Common Tasks

### Adding a New Subagent

1. Create `config/agent/subagents/{name}.md` with YAML frontmatter + system prompt
2. Create skill in `config/agent/skills/{skill-name}/README.md`
3. Add tools to the subagent's `tools:` list (built-in or MCP)
4. Update orchestrator's `PROMPT.md` to reference the subagent
5. Add evaluation test in `tests/skills/test_{name}.py`

### Adding a New Tool

**Built-in tools** (Python functions):
1. Define tool in `deep_agent/src/infrastructure/tools.py` or appropriate module
2. Register in tool registry (see existing tools for pattern)
3. Add to agent's `tools:` list in PROMPT.md frontmatter

**MCP tools**:
1. Deploy MCP server (see [template-mcp-server](https://github.com/redhat-data-and-ai/template-mcp-server))
2. Add server config to `config/agent/mcp.json`
3. Add server name to agent's `mcps:` list in PROMPT.md frontmatter
4. MCP tools are auto-discovered and exposed to the agent

### Changing Models

**Option 1: In PROMPT.md frontmatter**
```yaml
model: gemini-2.5-pro  # or claude-sonnet-4-6@default, gpt-4o, etc.
```

**Option 2: Via environment variable**
```bash
# Override for all agents
export DEFAULT_MODEL=claude-sonnet-4-6@default
```

**Supported providers**:
- **Google**: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-3.1-pro-preview`
- **Anthropic** (via Vertex AI): `claude-sonnet-4-6@default`, `claude-sonnet-4`
- **OpenAI**: `gpt-4o`, `gpt-4-turbo`, etc. (requires `OPENAI_API_KEY`)
- **vLLM/Ollama**: Any model name (routed to `VLLM_BASE_URL`)

### Modifying Middleware

Edit `config/agent/runtime/agent.yaml`:
```yaml
middleware:
  memory:
    enabled: true
    namespaces: ["memories"]
  pii:
    enabled: true
    rules:
      - type: credit_card
        strategy: mask
  # Add new middleware here
```

**Per-agent overrides**: Add `middleware:` section to PROMPT.md frontmatter.

### Debugging

**View logs**:
```bash
# Docker Compose
podman-compose logs -f template-agent

# Kind
kubectl -n template-agent logs -l component=agent -f

# Local dev server
# Logs go to stdout
```

**Enable debug logging**:
```bash
export PYTHON_LOG_LEVEL=DEBUG
export LANGGRAPH_LOG_LEVEL=DEBUG
```

**Check Langfuse traces**: Set `LANGFUSE_*` env vars and visit your Langfuse dashboard.

## Important Conventions

- **Never calculate BMI in orchestrator** - Always delegate to analyst subagent
- **TODO lists are mandatory** - Create TODO list BEFORE starting ANY work (simple or complex)
- **Email validation before sending** - Always use `validate_email` tool before delegating to publisher
- **Imperial to metric conversion** - Use formulas from `client-intake` skill (don't write your own)
- **Subagent execution order** - Publisher (email sender) is ALWAYS last
- **Test before claiming completion** - Run relevant tests (`make test-skills`) after making changes

## Dependencies

- **Python**: 3.12+ (managed by `uv`)
- **deepagents**: 0.4.12 (orchestration framework)
- **LangGraph**: Checkpoint-based state management
- **Langfuse**: Optional observability
- **PostgreSQL**: Conversation persistence (via `langgraph-checkpoint-postgres`)
- **Redis**: Caching (personalization, auth tokens)
- **MCP**: Model Context Protocol for tool integration
