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
| `REDIS_URL` | `redis://redis:6379/0` | No (Yes for `oauth` / `dcr` MCPs) |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | — | Yes |
| `LANGFUSE_PUBLIC_KEY` | — | No |
| `LANGFUSE_SECRET_KEY` | — | No |
| `LANGFUSE_BASE_URL` | — | No |
| `LANGFUSE_TRACING_ENVIRONMENT` | `development` | No |
| `SSL_KEYFILE` | — | No |
| `SSL_CERTFILE` | — | No |
| `MCP_TOKEN_ENCRYPTION_KEY` | — | Yes (for `oauth` / `dcr` MCPs) |
| `MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS` | — | No (decrypt-only; set during key rotation) |
| `AGENT_PUBLIC_BASE_URL` | `http://localhost:{AGENT_PORT}` | Yes (for `oauth` / `dcr` in production) |

Runtime configuration (cache TTLs, memory settings, agent identity) is in `config/agent/runtime/agent.yaml`.

## MCP Server Configuration

MCP servers are defined in `config/agent/mcp.json` and attached to agents via the
`mcps` frontmatter field in `config/agent/PROMPT.md` (orchestrator) or
`config/agent/subagents/*.md`. Subagents that omit `mcps` inherit the
orchestrator's MCP list.

### Auth modes

| `auth_mode` | When to use | How credentials work |
|---|---|---|
| `sso` (default) | MCP accepts the same SSO token as the agent (e.g. RH-SSO) | User's request Bearer token is forwarded to the MCP on every tool call |
| `oauth` | MCP has a pre-registered OAuth client | User connects once via the chat UI; per-user tokens are stored encrypted in Redis |
| `dcr` | MCP supports OAuth Dynamic Client Registration | Agent registers a client at startup/connect, then same per-user OAuth flow as `oauth` |

Set `"auth": false` to call an MCP without an `Authorization` header (public/local servers only).

### `mcp.json` schema

Each entry under `mcpServers` is keyed by a **server name** (referenced in agent frontmatter).
The name must match exactly in `mcps:` lists.

#### Common properties (all modes)

| Property | Type | Required | Default | Description |
|---|---|---|---|---|
| `url` | string | **Yes** | — | MCP server endpoint (e.g. `http://localhost:5001/mcp`) |
| `transport` | string | No | `streamable_http` | MCP transport (`streamable_http`, `http`, etc.) |
| `enabled` | boolean | No | `true` | When `false`, the server is ignored at runtime |
| `auth` | boolean | No | `true` | When `false`, no Bearer token is sent |
| `auth_mode` | string | No | `sso` | One of `sso`, `oauth`, `dcr` |
| `ssl_verify` | boolean | No | `true` | TLS certificate verification for MCP HTTP calls |
| `timeout` | number | No | `30` | Connection timeout in seconds |

#### SSO example

Use when the MCP server validates the same SSO access token as the agent (token pass-through).

```json
{
  "mcpServers": {
    "template-mcp-server": {
      "url": "http://localhost:5001/mcp",
      "transport": "streamable_http",
      "enabled": true,
      "auth": true,
      "auth_mode": "sso",
      "ssl_verify": false,
      "timeout": 30
    }
  }
}
```

**Required:** `url`
**No `oauth` block needed.**

#### OAuth example (pre-registered client)

Use when you already have a `client_id` (and optionally a client secret via environment
variable) from the MCP provider. Users connect through the chat UI before tools are available.

```json
{
  "mcpServers": {
    "my-oauth-mcp": {
      "url": "https://mcp.example.com/mcp",
      "transport": "streamable_http",
      "enabled": true,
      "auth": true,
      "auth_mode": "oauth",
      "ssl_verify": true,
      "timeout": 30,
      "oauth": {
        "client_id": "your-client-id",
        "client_secret_env": "MY_OAUTH_MCP_CLIENT_SECRET",
        "authorization_endpoint": "https://auth.example.com/authorize",
        "token_endpoint": "https://auth.example.com/token",
        "scopes": ["read", "write"]
      }
    }
  }
}
```

The OAuth redirect URI is derived at runtime from `AGENT_PUBLIC_BASE_URL` as
`{AGENT_PUBLIC_BASE_URL}/mcp/oauth/callback` — do not set `redirect_uri` in `mcp.json`.

**Required `oauth` fields for `auth_mode: "oauth"`:**

| Field | Required | Description |
|---|---|---|
| `client_id` | **Yes** | Pre-registered OAuth client ID |
| `authorization_endpoint` | **Yes** | Authorization URL |
| `token_endpoint` | **Yes** | Token exchange URL |
| `client_secret_env` | No | Name of an environment variable holding the client secret (preferred) |
| `scopes` | No | OAuth scopes (array of strings) |

**Security:** Never put `client_secret` in `mcp.json` — it is version-controlled and
easy to leak. Use `client_secret_env` and set the secret in `.env` or your secret
manager. Inline `client_secret` in config still works but logs a deprecation warning.

**HTTPS:** Set `AGENT_PUBLIC_BASE_URL` to `https://…` in production. Use
`http://localhost:{AGENT_PORT}` only for local development; authorization codes and
tokens can be intercepted when OAuth callbacks use plain HTTP.

#### DCR example (dynamic client registration)

Use when the MCP provider exposes a registration endpoint (e.g. Atlassian MCP, local
`template-mcp-server-rh-sso`). The agent registers itself and stores the resulting
`client_id` in Postgres — no manual `client_id` in config.

```json
{
  "mcpServers": {
    "jira-mcp-server-dcr": {
      "url": "https://mcp.atlassian.com/v1/mcp/authv2",
      "transport": "streamable_http",
      "enabled": true,
      "auth": true,
      "auth_mode": "dcr",
      "ssl_verify": true,
      "timeout": 30,
      "oauth": {
        "authorization_endpoint": "https://mcp.atlassian.com/v1/authorize",
        "token_endpoint": "https://cf.mcp.atlassian.com/v1/token",
        "registration_endpoint": "https://cf.mcp.atlassian.com/v1/register",
        "scopes": ["jira:read"]
      }
    }
  }
}
```

**Required `oauth` fields for `auth_mode: "dcr"`:**

| Field | Required | Description |
|---|---|---|
| `authorization_endpoint` | **Yes** | Authorization URL |
| `token_endpoint` | **Yes** | Token exchange and refresh URL |
| `registration_endpoint` | **Yes** | DCR registration URL |
| `scopes` | No | OAuth scopes requested at registration and authorize time |

`client_id` and `client_secret` are **not** set in config — they are created at runtime
and stored in the `mcp_oauth_clients` table.

#### Unauthenticated example

```json
{
  "mcpServers": {
    "local-mock-mcp": {
      "url": "http://localhost:8000/mcp",
      "transport": "streamable_http",
      "enabled": true,
      "auth": false
    }
  }
}
```

### Wiring MCPs to agents

Reference server names from `mcp.json` in agent frontmatter:

```yaml
---
name: jira
model: gemini-2.5-pro
mcps:
  - jira-mcp-server-dcr
tools:
  - searchJiraIssuesUsingJql
  - getJiraIssue
---
```

- **Orchestrator:** add `mcps:` to `config/agent/PROMPT.md` frontmatter.
- **Subagent:** add `mcps:` to `config/agent/subagents/<name>.md` frontmatter.
- **Inheritance:** subagents without `mcps` inherit the orchestrator's list.
- **Validation:** every name in `mcps` must exist in `mcp.json` and have `enabled: true`.

### Environment variables for OAuth / DCR

When using `auth_mode: "oauth"` or `"dcr"`:

```bash
# Fernet key for encrypting tokens at rest in Redis (required)
# Generate: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
MCP_TOKEN_ENCRYPTION_KEY=...

# Optional previous key during rotation (decrypt-only; see "Encryption key rotation" below)
# MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS=...

# Per-MCP OAuth client secret (auth_mode: oauth — set in .env, referenced via
# oauth.client_secret_env in mcp.json; never commit real secrets in mcp.json)
# MY_OAUTH_MCP_CLIENT_SECRET=your-client-secret

# Public URL where the agent is reachable (OAuth redirect URI and connect URLs)
# Local dev: http://localhost:5002  |  Production: https://your-agent.example.com
AGENT_PUBLIC_BASE_URL=http://localhost:5002

# Redis stores encrypted per-user OAuth tokens and short-lived PKCE state (required)
REDIS_URL=redis://localhost:6379/0

# Postgres stores DCR client records (required for auth_mode: dcr)
POSTGRES_HOST=...
```

In production behind a gateway or ingress, set `AGENT_PUBLIC_BASE_URL` to the
externally reachable **HTTPS** URL (e.g. `https://agent.example.com`). The OAuth
redirect URI is derived automatically as `{AGENT_PUBLIC_BASE_URL}/mcp/oauth/callback`.
Do not use `http://` outside local development — tokens can be intercepted in transit.

### Encryption key rotation

`MCP_TOKEN_ENCRYPTION_KEY` encrypts OAuth access/refresh tokens in Redis and DCR
client secrets in Postgres (`mcp_oauth_clients`). The agent supports **dual-key
decryption** so you can rotate without wiping stored credentials or forcing every
user to re-authenticate immediately.

**Planned rotation (zero-downtime):**

1. Generate a new Fernet key.
2. Set `MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS` to the **current** key and
   `MCP_TOKEN_ENCRYPTION_KEY` to the **new** key.
3. Rolling-restart all agent pods. New writes use the new key; reads succeed for
   ciphertext encrypted with either key.
4. Wait for natural re-encryption: tokens are rewritten with the new key on OAuth
   callback or token refresh. DCR `client_secret` values are rewritten on the next
   `upsert_client`.
5. When all rows have been touched (or after a maintenance window), remove
   `MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS` and restart again.

**Emergency rotation (key compromised):**

1. Generate a new Fernet key and set it as `MCP_TOKEN_ENCRYPTION_KEY`.
2. Delete stored credentials so nothing encrypted with the leaked key remains:
   - Redis tokens: `redis-cli --scan --pattern 'aegra:mcp_oauth_token:*' | xargs redis-cli DEL`
   - DCR clients: `DELETE FROM mcp_oauth_clients;`
3. Restart agent pods. Users must reconnect OAuth/DCR MCPs; DCR servers re-register
   on next connect.
4. Revoke OAuth clients at the identity provider if the old key exposure could have
   leaked plaintext tokens from Redis dumps or Postgres backups.

If decryption fails (wrong keys or corrupt data), the agent logs an error and the
affected MCP call returns an authorization error until the user reconnects.

### OAuth HTTP endpoints

These routes are registered on the Aegra custom app (`deep_agent/aegra/http_app.py`):

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/mcp/{name}/connect` | POST | Bearer (SSO) | Start OAuth flow; returns `{ "authorize_url": "..." }` |
| `/mcp/oauth/callback` | GET | None (state-bound) | OAuth redirect target; exchanges code and stores tokens |
| `/mcp/{name}/status` | GET | Bearer (SSO) | Returns `{ "connected": true/false }` for the current user |
| `/info` | GET | None | Returns agent name and list of OAuth/DCR MCP server names |

The chat UI calls connect/status through its BFF proxy. The OAuth provider redirects
the user's browser directly to `/mcp/oauth/callback` on the agent.

### Choosing an auth mode

| Scenario | Recommended mode |
|---|---|
| MCP shares RH-SSO / Keycloak with the agent | `sso` |
| MCP issued you a static OAuth client | `oauth` |
| MCP supports RFC 7591 dynamic registration (Atlassian, template-mcp-server-rh-sso) | `dcr` |
| Local dev mock with no auth | `auth: false` |

See `config/agent/mcp.json` for working examples of `sso`, `dcr`, and Jira/Atlassian MCP.

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
