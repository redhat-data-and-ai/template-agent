# Template Agent

[![Python 3.12+](https://img.shields.io/badge/python-3.12,3.13-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/redhat-data-and-ai/template-agent/actions/workflows/test.yml/badge.svg)](https://github.com/redhat-data-and-ai/template-agent/actions/workflows/test.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A template for building [Deep Agents](https://github.com/langchain-ai/deepagents) with the [LangGraph](https://langchain-ai.github.io/langgraph/) framework via the Aegra CLI. Includes orchestrator + subagents, MCP tool integration, conversation persistence, Langfuse tracing, and OpenTelemetry metrics.

## Features

**Agent capabilities:**
- Orchestrator with analyst and publisher subagents
- Skills: `client-intake`, `bmi-report`, `email-formatter`
- MCP auth modes: SSO pass-through, OAuth, and DCR

**Infrastructure:**
- Aegra dev server with Redis-backed SSE streaming
- PostgreSQL checkpoints, memory, and feedback storage
- Config-as-code in `config/agent/` (no Python edits for most changes)
- Container-ready with Red Hat UBI; OpenShift and Kind deployment overlays

## Quick Start

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), [Podman](https://podman.io/), Google Vertex AI credentials

```bash
git clone https://github.com/redhat-data-and-ai/template-agent.git
cd template-agent
make install     # creates venv, installs deps + pre-commit hooks
make local       # pgvector + redis in compose; agent on host → :5002
```

Verify in another terminal:

```bash
curl http://localhost:5002/health
```

Copy `.env.example` to `.env` before first run (or let `make local` create it) and set `GOOGLE_APPLICATION_CREDENTIALS_CONTENT`.

**MCP and UI are separate repos** — this project runs the agent and its dependencies (Postgres, Redis) only. Clone and run [template-mcp-server](https://github.com/redhat-data-and-ai/template-mcp-server) and [template-ui](https://github.com/redhat-data-and-ai/template-ui) when needed.

## API

The agent exposes the standard **LangGraph API** (assistant ID: `agent`, defined in `aegra.json`) plus custom routes on the Aegra HTTP app.

### LangGraph API

| Endpoint | Method | Description |
|---|---|---|
| `/ok` | GET | Server health |
| `/assistants/{assistant_id}` | GET | Assistant metadata |
| `/threads` | POST | Create conversation thread |
| `/threads/{thread_id}` | GET | Get thread state |
| `/threads/{thread_id}/runs` | POST | Run agent (sync) |
| `/threads/{thread_id}/runs/stream` | POST | Run agent (SSE stream) |

```bash
# Create a thread
curl -X POST http://localhost:5002/threads \
  -H "Content-Type: application/json" \
  -d '{}'

# Stream a message (replace THREAD_ID)
curl -N -X POST "http://localhost:5002/threads/THREAD_ID/runs/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "agent",
    "input": {"messages": [{"role": "human", "content": "Hello"}]},
    "stream_mode": "updates"
  }'
```

### Custom routes

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check (also `/healthz`, `/readyz`, `/livez`) |
| `/info` | GET | Agent name and OAuth/DCR MCP server list |
| `/feedback` | POST | Record user feedback (Langfuse + Postgres) |
| `/feedback/{thread_id}` | GET | List feedback for a thread |
| `/threads/{thread_id}/token-usage` | GET | Cumulative token usage for a thread |
| `/mcp/{name}/connect` | POST | Start OAuth/DCR flow for an MCP server |
| `/mcp/oauth/callback` | GET | OAuth redirect handler |
| `/mcp/{name}/status` | GET | MCP connection status for current user |

Use [template-ui](https://github.com/redhat-data-and-ai/template-ui) for a full chat experience against this API.

## Configuration

Configuration is split between **secrets/endpoints** (`.env`) and **operational settings** (`config/agent/runtime/agent.yaml`).

### Environment variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_HOST` | `localhost` | Postgres host (`pgvector` in compose) |
| `POSTGRES_PORT` | `5432` | Postgres port |
| `POSTGRES_DB` | `template_agent` | Database name |
| `POSTGRES_USER` | `postgres` | Database user |
| `POSTGRES_PASSWORD` | `postgres` | Database password |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL (required for OAuth/DCR MCPs) |
| `REDIS_BROKER_ENABLED` | `true` | Enable Redis-backed SSE broker |
| `GOOGLE_APPLICATION_CREDENTIALS_CONTENT` | — | Google service account JSON (required) |
| `ENABLE_AUTH` | `false` in `.env.example` | SSO/OIDC authentication |
| `SSO_ISSUER_URL` | — | OIDC issuer (Keycloak, Okta, etc.) |
| `SSO_CLIENT_ID` | — | OIDC client ID |
| `SSO_CLIENT_SECRET` | — | OIDC client secret |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse public key (optional) |
| `LANGFUSE_SECRET_KEY` | — | Langfuse secret key (optional) |
| `LANGFUSE_BASE_URL` | — | Langfuse host (optional) |
| `LANGFUSE_TRACING_ENVIRONMENT` | `development` | Langfuse environment label |
| `MCP_TOKEN_ENCRYPTION_KEY` | — | Fernet key for OAuth/DCR token encryption |
| `MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS` | — | Previous key during rotation (decrypt-only) |
| `AGENT_PUBLIC_BASE_URL` | `http://localhost:5002` | Public agent URL for OAuth callbacks |
| `CUSTOM_CA_FILE` | — | Host path to a PEM file with custom CA certs (compose only) |
| `SSL_KEYFILE` | — | TLS private key path (optional) |
| `SSL_CERTFILE` | — | TLS certificate path (optional) |

See [`.env.example`](./.env.example) for the full list including OpenTelemetry and MongoDB token-usage settings.

Runtime settings (cache, memory, providers, middleware, agent identity) live in [`config/agent/runtime/agent.yaml`](./config/agent/runtime/agent.yaml).

## MCP Server Configuration

MCP servers are defined in [`config/agent/mcp.json`](./config/agent/mcp.json) and attached to agents via the `mcps` frontmatter field in [`config/agent/PROMPT.md`](./config/agent/PROMPT.md) (orchestrator) or [`config/agent/subagents/*.md`](./config/agent/subagents/).

### Auth modes

| `auth_mode` | When to use | How credentials work |
|---|---|---|
| `sso` (default) | MCP accepts the same SSO token as the agent | User Bearer token forwarded on every tool call |
| `oauth` | MCP has a pre-registered OAuth client | User connects via chat UI; tokens stored encrypted in Redis |
| `dcr` | MCP supports OAuth Dynamic Client Registration | Agent registers at connect; per-user OAuth flow follows |

Set `"auth": false` for public/local MCP servers with no Authorization header.

### MCP URL by run mode

| Mode | `url` in `mcp.json` |
|---|---|
| `make local` (agent on host) | `http://localhost:5001/mcp` (default) |
| `make container` (MCP on host) | `http://host.containers.internal:5001/mcp` |

Alternate URLs are provided as `//` comments in `mcp.json` — uncomment the line you need.

### Wiring MCPs to agents

```yaml
---
name: analyst
model: gemini-2.5-pro
mcps:
  - template-mcp-server
tools:
  - calculate_bmi
  - search_web
---
```

- **Orchestrator:** add `mcps:` to `config/agent/PROMPT.md` frontmatter.
- **Subagent:** add `mcps:` to `config/agent/subagents/<name>.md` frontmatter.
- **Inheritance:** subagents without `mcps` inherit the orchestrator's list.
- **Validation:** every name in `mcps` must exist in `mcp.json` with `enabled: true`.

See `config/agent/mcp.json` for working SSO and DCR examples.

## Project Structure

```
template-agent/
├── aegra.json                  # Aegra / LangGraph framework entry point
├── config/agent/
│   ├── PROMPT.md               # Orchestrator prompt + frontmatter
│   ├── subagents/              # Subagent definitions
│   ├── skills/                 # Skill documents and evals
│   ├── mcp.json                # MCP server registry
│   ├── runtime/agent.yaml      # Runtime config (cache, memory, providers)
│   └── deployment/values.yaml  # OpenShift/ArgoCD deployment reference values
├── deep_agent/
│   ├── aegra/                  # Graph, HTTP app, MCP OAuth, entrypoint
│   └── src/                    # Config loader, cache, memory, token budget, etc.
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Aegra integration and e2e tests
│   └── skills/                 # LLM-as-judge skill evaluations
├── compose.yaml                # Postgres + Redis (+ agent with --profile container)
├── Containerfile
└── deployment/                 # OpenShift and Kind overlays
```

## Testing

```bash
make test           # unit tests
make test-all       # unit + skills evals
make test-skills    # skills evaluations only
make test-cov       # unit tests with coverage
```

Skills evals auto-discover from `config/agent/skills/*/evals/evals.json`. See [`config/agent/evals/README.md`](./config/agent/evals/README.md) for Promptfoo and Lightspeed eval options.

```bash
# Code quality
ruff check . && ruff format .
pre-commit run --all-files
```

## Custom CA Certificates

If your environment uses a corporate or internal certificate authority, the container can trust it at startup without rebuilding the image.

**Compose** — set `CUSTOM_CA_FILE` in `.env` to the host path of your PEM bundle:

```bash
# .env
CUSTOM_CA_FILE=./certs/ca.pem
```

**Kubernetes** — create a Secret and mount it, then set `CUSTOM_CA_PATH`:

```yaml
env:
  - name: CUSTOM_CA_PATH
    value: /etc/custom-ca/ca.pem
volumeMounts:
  - name: custom-ca
    mountPath: /etc/custom-ca
    readOnly: true
volumes:
  - name: custom-ca
    secret:
      secretName: custom-ca
```

**Fallback URL** — for non-orchestrated environments (e.g. `podman run`), set `CUSTOM_CA_URL` to download the PEM at startup:

```bash
podman run -e CUSTOM_CA_URL=https://certs.example.com/ca.pem ...
```

If neither variable is set, or the download fails, the container starts normally with default system certs.

## Deployment

```bash
make container
```

For production: configure TLS (`SSL_KEYFILE`, `SSL_CERTFILE`), use managed PostgreSQL and Redis, set `AGENT_PUBLIC_BASE_URL` to your HTTPS URL, and enable Langfuse tracing.

OpenShift manifests are in `deployment/overlays/openshift/`. Local full-stack Kubernetes testing: `make kind`.

## Links

- [Issues](https://github.com/redhat-data-and-ai/template-agent/issues)
- [template-mcp-server](https://github.com/redhat-data-and-ai/template-mcp-server)
- [template-ui](https://github.com/redhat-data-and-ai/template-ui)
- [LangGraph docs](https://langchain-ai.github.io/langgraph/)

## License

[Apache 2.0](LICENSE)
