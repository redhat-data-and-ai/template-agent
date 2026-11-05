# Working Examples

Complete, copy-paste-ready configurations for every pattern supported by the template-agent framework. Each example includes inline comments explaining the choices.

> **Prerequisite reading:** [Configuration Reference](./01-configuration-reference.md) covers the full schema. This guide shows how to assemble those options into real configurations.

---

## 1. Minimal `agent.yaml`

The smallest configuration that produces a working agent. Every field below is required or strongly recommended; everything else has sensible defaults.

```yaml
# config/agent/runtime/agent.yaml — Minimal

# ── Identity ──────────────────────────────────────────────────── [YAML-loaded]
# Display name shown in the chat UI and Langfuse traces.
name: "My Agent"

# ── Model ─────────────────────────────────────────────────────── [env-var]
# max_output_tokens caps the LLM response length per turn.
# 8192 is a safe default; raise for agents that produce long reports.
model:
  max_output_tokens: 8192

# ── Provider Strategy ─────────────────────────────────────────── [YAML-loaded]
# "legacy" uses built-in create_model() with Vertex AI (Gemini + Claude) + vLLM.
# Switch to "deepagents" when you need ProviderProfile-based model routing.
resolve_strategy: legacy

# ── Middleware ─────────────────────────────────────────────────── [YAML-loaded]
# Enable only the essentials. Everything else defaults to disabled.
middleware:
  human_approval:
    enabled: false          # No human-in-the-loop for local dev
  patch_tool_calls:
    enabled: true           # Fix malformed tool calls from the LLM
  model_call_limit:
    enabled: true
    run_limit: 50           # Safety net — abort after 50 LLM calls per run
  extra: []                 # No custom middleware

# ── Filesystem ─────────────────────────────────────────────────── [YAML-loaded]
# "state" is the simplest backend — thread-scoped scratch, no persistence.
filesystem:
  backend:
    type: state

# ── Server ─────────────────────────────────────────────────────── [env-var]
server:
  host: "0.0.0.0"
  port: 5002
```

---

## 2. Full-Featured `agent.yaml`

Production-ready configuration with every section enabled and annotated. Based on the actual `config/agent/runtime/agent.yaml` shipped with the template.

```yaml
# config/agent/runtime/agent.yaml — Full-Featured (Production)
#
# Unified runtime config for the template agent. Sections marked [YAML-loaded]
# are parsed by the Python config loader at startup. Sections marked [env-var]
# are read from environment variables via Pydantic BaseSettings — they appear
# here as the canonical reference for what those settings do and their defaults.
#
# Template users configure everything here. No Python code needed.
#
# OpenShift notes:
#   - Secrets (DB passwords, API keys) come via OpenShift Secrets -> env vars.
#   - Infrastructure endpoints (DB host, Redis host) come via ConfigMaps -> env vars.
#   - Everything else lives here.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Identity                                                        [YAML-loaded]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Display name for the agent — shown in UI, logs, and Langfuse traces.
name: "Health Assistant"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model                                                              [env-var]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
model:
  max_output_tokens: 8192   # Max tokens per LLM response. Raise for report-heavy agents.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Providers                                                      [YAML-loaded]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model resolution strategy:
#   legacy:     Use built-in create_model() — Vertex AI (Gemini + Claude) + vLLM.
#   deepagents: Use deepagents resolve_model() + ProviderProfile registry.
resolve_strategy: legacy

# Provider profiles — registered with deepagents.register_provider_profile().
# Only used when resolve_strategy: deepagents.
providers:
  google_genai:
    init_kwargs:
      # project: ${GCP_PROJECT}         # Uncomment and set your GCP project
      temperature: 0.0                   # Deterministic output
  anthropic_vertex:
    init_kwargs:
      # project: ${GCP_PROJECT}
      temperature: 0.0
  openai:
    init_kwargs:
      temperature: 0.0

# vLLM / OpenAI-compatible models:
#   Any model not in the built-in Gemini/Claude lists is routed to the
#   vLLM endpoint. Set VLLM_BASE_URL env var to your inference server.
#   Examples:
#     VLLM_BASE_URL=http://vllm-server:8000/v1
#     VLLM_BASE_URL=http://ollama:11434/v1

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Harness Profiles                                               [YAML-loaded]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-model runtime adjustments. Single source of truth — used by both
# the provider registry (deepagents HarnessProfile) and middleware resolution.
# Key format: "provider:model" or just "model" for provider-agnostic matching.
harness_profiles:
  gemini-2.5-pro:
    system_prompt_suffix: ""            # Append text to the system prompt for this model
    excluded_tools: []                  # Tools to hide from this model
    excluded_middleware: []             # Middleware class names to skip for this model
    general_purpose_subagent:
      enabled: true                     # Allow this model to use the task delegation tool
  gemini-2.5-flash:
    system_prompt_suffix: ""
    excluded_tools: []
    excluded_middleware: []
    general_purpose_subagent:
      enabled: true
  claude-sonnet-4:
    system_prompt_suffix: ""
    excluded_tools: []
    excluded_middleware:
      - patch_tool_calls               # Claude produces well-formed tool calls natively
    general_purpose_subagent:
      enabled: true

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Middleware Pipeline                                            [YAML-loaded]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Controls which deepagents middleware is active and how it behaves.
# Resolution order: defaults -> profile (matched from model: field) -> per-agent overrides
middleware:
  # ── Human Approval ──────────────────────────────────────────────
  human_approval:
    enabled: true             # Human-in-the-loop tool approval
    mode: all                 # 'all' = every tool; 'none' = disabled
    exclude:                  # Tools that skip approval (internal operations)
      - write_todos
      - compact_conversation

  # ── Built-in Middleware ─────────────────────────────────────────
  summarization_tool:
    enabled: true             # Agent can proactively trigger conversation summarization

  memory:
    enabled: true             # Cross-conversation memory via LangGraph Store
    namespaces:
      - "memories"            # Memory namespace(s) for this agent

  patch_tool_calls:
    enabled: true             # Auto-fix malformed tool calls from the LLM

  skills:
    enabled: true             # Enable skills system (reads from /skills/ directory)

  # ── Production Guardrails ───────────────────────────────────────
  model_call_limit:
    enabled: true
    run_limit: 50             # Max LLM calls per single run (prevents infinite loops)

  tool_call_limit:
    enabled: true
    run_limit: 200            # Max tool calls per single run

  model_retry:
    enabled: true
    max_retries: 3            # Retry transient LLM failures (rate limits, timeouts)
    backoff_factor: 2.0       # Exponential backoff multiplier
    initial_delay: 1.0        # First retry delay in seconds

  model_fallback:
    enabled: false            # Switch to fallback model on primary failure
    fallback_model: "google_genai:gemini-2.5-flash"
    # Requires GOOGLE_API_KEY for Developer API, or matching Vertex AI config.
    # Enable when fallback model uses same auth as primary.

  tool_retry:
    enabled: true
    max_retries: 2            # Retry failed tool calls (network issues, transient errors)
    tools:                    # Only retry these specific tools
      - template_calculate_bmi
      - template_search_web
      - template_send_email

  # ── Custom Middleware ───────────────────────────────────────────
  extra: []                   # List of dotted paths to custom middleware classes
  # Example:
  #   extra:
  #     - "my_project.middleware.logging:RequestLoggingMiddleware"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Async Tasks                                                    [YAML-loaded]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async_tasks:
  enabled: true               # Allow agent to run background tasks
  system_prompt: null          # Optional override for async task system prompt

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Filesystem & Storage                                           [YAML-loaded]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Backend types:
#   state (default): Thread-scoped scratch. Recommended for production / OpenShift.
#   composite:       Routes paths to different backends (scratch + persistent memory).
#   store:           Cross-thread persistent storage via LangGraph Store.
#   local_shell:     Real filesystem with isolated venv. LOCAL DEV ONLY.
filesystem:
  backend:
    type: composite           # Route different paths to different storage backends
    local_shell:
      timeout: 120            # Shell command timeout in seconds
      max_output_bytes: 100000
    store:
      enabled: true           # Enable LangGraph Store for persistent paths
      scope: user             # "user" = per-user storage; "thread" = per-conversation
    routes:                   # Path prefix -> backend mapping
      "/skills/": filesystem_readonly   # Skills are read-only
      "/memories/": store               # Memories persist across threads
      "/reports/": store                # Reports persist across threads
      "/": state                        # Everything else is thread-scoped scratch

  permissions:                # Fine-grained file operation permissions
    - operations: [read, glob, grep, ls]
      paths: ["config/**", "docs/**", "reports/**", "skills/**"]
      mode: allow
    - operations: [write, edit]
      paths: ["reports/**", "memories/**"]
      mode: allow
    - operations: [write, edit]
      paths: ["config/**", "*.py", "*.sh"]
      mode: deny              # Prevent agent from modifying config or code

  permission_inheritance: false  # Subagents do NOT inherit parent permissions

  settings:
    tool_token_limit_before_evict: 20000    # Evict tool output beyond this token count
    human_message_token_limit_before_evict: 50000
    max_execute_timeout: 3600               # Max execution time for any single operation

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cache                                                          [YAML-loaded]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cache:
  enabled: true
  model:
    enabled: true
    ttl: 600                  # Cache identical LLM requests for 10 minutes
    max_size: 50              # Max cached model responses
  personalization:
    enabled: true
    ttl: 120                  # User personalization cache TTL
  mcp:
    ttl: 300                  # MCP tool list cache — avoids reconnecting per request
  graph:
    ttl: 300                  # Compiled graph cache — avoids rebuilding per request
  redis:
    enabled: true             # Use Redis as cache backend (requires REDIS_URL)
  warming:
    enabled: true             # Pre-warm caches on startup
  metrics:
    enabled: true             # Expose cache hit/miss metrics

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Memory Processing                                              [env-var]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
memory:
  consolidation:
    enabled: true             # Merge duplicate/overlapping memories
  decay:
    enabled: true             # Fade old memories over time
    lambda: 0.05              # Decay rate (higher = faster decay)
  clustering:
    enabled: true             # Group related memories
    threshold: 0.4            # Similarity threshold for clustering
    min_cluster_size: 3       # Minimum memories to form a cluster
  relationships:
    enabled: true             # Track relationships between memories
  scheduler:
    interval_hours: 6         # Run memory processing every 6 hours
  max_inject: 20              # Max memories injected into context per turn

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Guardrail                                                      [YAML-loaded]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IBM Granite Guardian content safety checks.
guardrail:
  enabled: false              # Set true + provide GUARDIAN_API_BASE to activate
  model: "/data/granite-guardian-4.1-8b"
  # Credentials injected via env vars:
  #   GUARDIAN_API_BASE   — endpoint URL (required to activate guardrails)
  #   GUARDIAN_API_KEY    — API key ("EMPTY" for unauthenticated endpoints)
  #   GUARDIAN_SSL_VERIFY — true/false, default true

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Token Budget                                                   [YAML-loaded]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tracks cumulative LLM tokens per conversation thread_id in MongoDB.
token_budget:
  enabled: false              # Requires MONGODB_URI and MONGODB_DB env vars

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Platform                                                       [env-var]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
platform:
  audit:
    enabled: false            # Platform audit event logging
    buffer_max: 1000          # Max buffered audit events before flush

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OPA (Authorization)                                            [env-var]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
opa:
  enabled: false              # Open Policy Agent authorization
  url: http://opa:8181/v1/data/agent/authz
  timeout: 2.0                # OPA request timeout in seconds
  max_retries: 3              # Retries on OPA connection failure
  fail_open: false            # false = deny on OPA failure (recommended)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logging                                                        [env-var]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logging:
  level: INFO                 # DEBUG, INFO, WARNING, ERROR
  request:
    enabled: true             # Log incoming HTTP requests
    headers: true             # Include request headers in logs
    body: true                # Include request body in logs
    body_max_size: 10240      # Truncate body logs at 10KB

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Server                                                         [env-var]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
server:
  host: "0.0.0.0"             # Listen on all interfaces (required for containers)
  port: 5002                  # HTTP port
```

---

## 3. `mcp.json` Examples

MCP (Model Context Protocol) server configurations. Each server entry defines how the agent connects to an external tool provider. The file lives at `config/agent/mcp.json`.

### 3.1 SSO Pass-Through

The simplest auth mode. The agent forwards the user's SSO token to the MCP server. No OAuth dance required.

```json
{
    "mcpServers": {
        "my-tools": {
            "url": "http://localhost:5001/mcp",
            "transport": "streamable_http",
            "enabled": true,
            "auth": true,
            "auth_mode": "sso",
            "ssl_verify": false,
            "timeout": 30,
            "tool_prefix": "mytools"
        }
    }
}
```

| Field | Purpose |
|-------|---------|
| `url` | MCP server endpoint. Use `host.containers.internal` when agent runs in a container but the MCP server is on the host. |
| `transport` | Protocol. `streamable_http` is the standard choice. |
| `auth` | Must be `true` to enable any authentication. |
| `auth_mode` | `sso` forwards the user's SSO bearer token as-is. |
| `ssl_verify` | Set `false` only for local dev with self-signed certs. |
| `timeout` | Connection timeout in seconds. |
| `tool_prefix` | Prefixed to all tool names from this server (e.g., `mytools_search`). Prevents collisions when multiple servers expose tools with the same name. |

### 3.2 OAuth with Authorization Code Flow

For MCP servers that require their own OAuth token, obtained via a standard authorization_code grant. The agent performs the OAuth dance on behalf of the user.

```json
{
    "mcpServers": {
        "oauth-tools": {
            "url": "https://mcp.example.com/mcp",
            "transport": "streamable_http",
            "enabled": true,
            "auth": true,
            "auth_mode": "oauth",
            "ssl_verify": true,
            "timeout": 30,
            "tool_prefix": "ext",
            "oauth": {
                "authorization_endpoint": "https://idp.example.com/authorize",
                "token_endpoint": "https://idp.example.com/token",
                "client_id": "my-agent-client",
                "client_secret_env": "MY_OAUTH_MCP_CLIENT_SECRET",
                "scopes": ["openid", "profile", "tools.read", "tools.execute"]
            }
        }
    }
}
```

| Field | Purpose |
|-------|---------|
| `oauth.authorization_endpoint` | IdP authorization URL for the authorization_code grant. |
| `oauth.token_endpoint` | IdP token URL for exchanging the auth code for tokens. |
| `oauth.client_id` | OAuth client ID registered with the IdP. |
| `oauth.client_secret_env` | Name of the env var holding the client secret (never hardcode secrets in JSON). |
| `oauth.scopes` | OAuth scopes to request. Must include scopes required by the MCP server. |

> **Note:** Set `MCP_TOKEN_ENCRYPTION_KEY` in `.env` to encrypt access/refresh tokens stored in Redis. See [Configuration Reference](./01-configuration-reference.md) for key generation.

### 3.3 DCR (Dynamic Client Registration)

For MCP servers that support OAuth 2.0 Dynamic Client Registration (RFC 7591). The agent registers itself as an OAuth client at startup, then performs the authorization_code flow.

```json
{
    "mcpServers": {
        "dcr-tools": {
            "url": "http://localhost:5001/mcp",
            "transport": "streamable_http",
            "enabled": true,
            "auth": true,
            "auth_mode": "dcr",
            "ssl_verify": false,
            "timeout": 30,
            "tool_prefix": "dcr",
            "oauth": {
                "authorization_endpoint": "http://localhost:5001/auth/authorize",
                "token_endpoint": "http://localhost:5001/auth/token",
                "registration_endpoint": "http://localhost:5001/auth/register",
                "scopes": ["email", "openid", "profile", "session:role-any"]
            }
        }
    }
}
```

| Field | Purpose |
|-------|---------|
| `oauth.registration_endpoint` | RFC 7591 dynamic registration URL. The agent auto-registers a client here. |
| No `client_id` | Omitted because DCR assigns one dynamically. |
| No `client_secret_env` | The registration endpoint returns the secret; it is encrypted and stored automatically. |

> **Note:** DCR requires `MCP_TOKEN_ENCRYPTION_KEY` to encrypt the dynamically obtained client secret in Postgres.

### 3.4 API Key Authentication

For MCP servers that accept a static API key in the `Authorization` header instead of OAuth tokens.

```json
{
    "mcpServers": {
        "api-key-tools": {
            "url": "https://mcp.example.com/mcp",
            "transport": "streamable_http",
            "enabled": true,
            "auth": true,
            "auth_mode": "api_key",
            "auth_env_var": "MY_MCP_API_KEY",
            "ssl_verify": true,
            "timeout": 30,
            "tool_prefix": "ext"
        }
    }
}
```

| Field | Purpose |
|-------|---------|
| `auth_mode` | `api_key` sends the key as a Bearer token. |
| `auth_env_var` | Name of the env var holding the API key. The key is never stored in `mcp.json`. |

Then in your `.env`:

```bash
MY_MCP_API_KEY=sk-abc123...
```

### 3.5 Multi-Server Configuration

A realistic production setup combining different auth modes. Only enabled servers are connected at startup.

```json
{
    "mcpServers": {
        "internal-tools": {
            "url": "http://internal-mcp.svc:5001/mcp",
            "transport": "streamable_http",
            "enabled": true,
            "auth": true,
            "auth_mode": "sso",
            "ssl_verify": true,
            "timeout": 30,
            "tool_prefix": "internal"
        },
        "partner-api": {
            "url": "https://partner.example.com/mcp",
            "transport": "streamable_http",
            "enabled": true,
            "auth": true,
            "auth_mode": "api_key",
            "auth_env_var": "PARTNER_API_KEY",
            "ssl_verify": true,
            "timeout": 60,
            "tool_prefix": "partner"
        },
        "data-platform": {
            "url": "https://data.example.com/mcp",
            "transport": "streamable_http",
            "enabled": true,
            "auth": true,
            "auth_mode": "dcr",
            "ssl_verify": true,
            "timeout": 30,
            "tool_prefix": "data",
            "oauth": {
                "authorization_endpoint": "https://sso.example.com/authorize",
                "token_endpoint": "https://sso.example.com/token",
                "registration_endpoint": "https://sso.example.com/register",
                "scopes": ["openid", "data.read", "data.write"]
            }
        },
        "experimental-tools": {
            "url": "http://localhost:9000/mcp",
            "transport": "streamable_http",
            "enabled": false,
            "auth": false,
            "ssl_verify": false,
            "timeout": 10,
            "tool_prefix": "exp"
        }
    }
}
```

---

## 4. Subagent Examples

Subagents are defined as Markdown files in `config/agent/subagents/`. The YAML front matter configures the subagent; the Markdown body is its system prompt. The orchestrator delegates tasks to subagents via the `task` tool.

### 4.1 Default Subagent (Minimal)

An in-process, synchronous subagent with minimal configuration. Uses the `default` type, which inherits the orchestrator's model and runs inline.

```markdown
---
name: summarizer
type: default
description: >
  Summarizes long documents into concise bullet points.
  Use when the user provides a document and asks for a summary.
model: gemini-2.5-pro
tools:
  - template_search_web
skills:
  - summary-format
---

You are a Document Summarizer.

## General Behavior

Produce concise, accurate summaries of provided content. Each summary must
be a bullet-point list of key findings, no longer than 10 items.

## Input Requirement

| Field       | Type   | Required |
|-------------|--------|----------|
| document    | string | Yes      |
| max_bullets | int    | No       |

## Workflow

1. Read the provided document.
2. Identify key themes and findings.
3. Produce a bullet-point summary.

## Output Format

- Bullet-point list in Markdown.
- Each bullet is one concise sentence.
- Maximum 10 bullets unless max_bullets is specified.
```

### 4.2 Compiled Subagent (with Tools, Skills, and Model)

A compiled subagent with dedicated tools, skills, and a specific model. Based on the actual `analyst.md` shipped with the template.

```markdown
---
name: analyst
type: compiled
description: >
  Calculates BMI, classifies the result, and fetches category-specific
  health tips for Red Hat employees. Use when the user provides height
  and weight for BMI analysis.
model: gemini-2.5-pro
tools:
  - template_calculate_bmi
  - template_search_web
skills:
  - bmi-report
---

You are a BMI Analyst for Red Hat employees.

## General Behavior

Calculate and classify BMI using the provided tools — never compute values
inline or from internal knowledge. Read the **bmi-report** skill for
BMI categories and report structure. Tone must be encouraging and
non-judgmental. Never use words like "bad" or "failing."

## Input Requirement

| Field  | Type              | Required |
|--------|-------------------|----------|
| height | float, in **cm**  | Yes      |
| weight | float, in **kg**  | Yes      |

Both values must already be in metric units. Unit conversion is not handled here.

## Workflow

1. Calculate BMI via `calculate_bmi(height_cm, weight_kg)`.
2. Classify: Underweight (<18.5) · Normal (18.5-24.9) · Overweight (25-29.9) · Obese (30+).
3. Search for 3 health tips via `search_web` based on the BMI category.

## Output Format

- Use proper Markdown: headers, bold labels, bullet lists, and tables where
  they improve readability.
- BMI value rounded to one decimal place.
- Health tips as a numbered list, each tip one concise sentence.
- The disclaimer must appear as the final line of every report:
  "This is not medical advice. Consult a healthcare professional."

## Error Handling

| Failure                           | Action                                                    |
|-----------------------------------|-----------------------------------------------------------|
| `calculate_bmi` returns an error  | Report the error to the user. Do not estimate BMI manually. |
| `search_web` returns no results   | Return the report without tips and note that tips were unavailable. |
```

**Key differences from default:**

| Front matter field | `default` | `compiled` |
|--------------------|-----------|------------|
| `type` | `default` -- runs inline, shares orchestrator graph | `compiled` -- builds its own LangGraph, isolated state |
| `tools` | Optional (inherits from orchestrator if empty) | Required -- only listed tools are available |
| `skills` | Optional | Optional -- skills referenced in the prompt body |

### 4.3 Async Subagent (Remote Agent Protocol)

An asynchronous subagent that runs as a separate service, communicating via the Agent Protocol. Use this for long-running tasks or subagents hosted externally.

```markdown
---
name: report-generator
type: remote
description: >
  Generates comprehensive PDF reports from analysis data.
  Use for report generation requests that take more than 30 seconds.
graph_id: report-agent
url: http://report-service:5003
---

You are a Report Generator.

## General Behavior

Accept structured analysis data and produce a formatted PDF report.
This subagent runs asynchronously — the orchestrator receives a task ID
and polls for completion.

## Input Requirement

| Field       | Type   | Required |
|-------------|--------|----------|
| report_data | object | Yes      |
| format      | string | No       |

## Output Format

- Return a confirmation with the report download URL.
- Include a brief summary of what was generated.
```

| Front matter field | Purpose |
|--------------------|---------|
| `type: remote` | Runs as a separate Agent Protocol service, not in-process. |
| `graph_id` | The LangGraph deployment graph ID on the remote service. |
| `url` | Base URL of the remote Agent Protocol endpoint. |

### 4.4 Subagent with Custom Middleware Overrides

A subagent that overrides specific middleware settings from the parent. Useful when a subagent needs different guardrails (e.g., higher call limits for complex analysis).

```markdown
---
name: deep-researcher
type: compiled
description: >
  Performs in-depth multi-step research by querying multiple data sources.
  Use for complex research questions that require iterating over results.
model: gemini-2.5-pro
tools:
  - template_search_web
middleware:
  model_call_limit:
    run_limit: 100
  tool_call_limit:
    run_limit: 500
  model_retry:
    max_retries: 5
    backoff_factor: 3.0
    initial_delay: 2.0
---

You are a Deep Researcher.

## General Behavior

Iteratively search, analyze, and synthesize information from multiple
sources. You have higher call limits than other subagents because research
often requires many search iterations.

## Workflow

1. Break the research question into sub-questions.
2. Search for each sub-question independently.
3. Cross-reference findings for consistency.
4. Synthesize a final answer with citations.

## Output Format

- Structured Markdown with headers for each sub-question.
- Inline citations: `[Source: <url>]` after each claim.
- A "Confidence" section rating certainty (High / Medium / Low).
```

---

## 5. `pii.yaml` Examples

PII (Personally Identifiable Information) detection and scrubbing configuration. Lives at `config/agent/runtime/pii.yaml`.

**Valid values** (from the `PIIConfig` Pydantic model in `deep_agent/src/pii/config.py`):

| Field | Valid values |
|-------|-------------|
| `strategy` | `scrub` (reversible tokenization), `tokenize` (alias for scrub), `hash` (HMAC-SHA256), `redact` (one-way `***REDACTED***`), `mask` (partial mask, e.g., `****-1234`), `block` (reject request) |
| `provider` | `default` (stock LangChain PIIMiddleware), `regex` (token-map with built-in regex), `presidio` (token-map with Presidio NLP), `custom` (token-map with your own regex) |
| `trace_strategy` | `redact` (`***REDACTED***` in traces) or `hash` (`[HASH:abc123]` -- correlatable across requests) |

### 5.1 Basic PII Rules (Email Scrub, Credit Card Mask)

Minimal setup that protects the most common PII types.

```yaml
# config/agent/runtime/pii.yaml — Basic

enabled: true
trace_strategy: hash        # PII in Langfuse traces appears as [HASH:abc123]

rules:
  # Email addresses: reversible scrub.
  # "scrub" replaces with [MAIL_1], [MAIL_2], etc. and restores in LLM output.
  - name: email
    strategy: scrub
    provider: regex          # Built-in email regex — no custom pattern needed
    label: MAIL              # Token prefix: [MAIL_1], [MAIL_2], ...

  # Credit cards: one-way partial mask.
  # "mask" produces ****-****-****-1234 (keeps last 4 digits).
  - name: credit_card
    strategy: mask
    provider: default        # Stock LangChain PIIMiddleware
```

### 5.2 Custom Regex Rule (Employee ID Pattern)

Add a custom regex pattern for domain-specific PII.

```yaml
# config/agent/runtime/pii.yaml — Custom Regex

enabled: true
trace_strategy: hash

rules:
  - name: email
    strategy: scrub
    provider: regex
    label: MAIL

  # Custom employee ID pattern: EMP-123456
  # "custom" provider requires a "regex" field with your pattern.
  - name: employee_id
    strategy: scrub          # Reversible — LLM sees [EMP_ID_1], output restores original
    provider: custom         # Uses your regex, not a built-in pattern
    regex: '\bEMP-\d{6}\b'  # Matches EMP- followed by exactly 6 digits
    label: EMP_ID            # Token prefix: [EMP_ID_1], [EMP_ID_2], ...

  # Indian PAN card numbers: ABCDE1234F
  - name: pan_card
    strategy: block          # Reject the entire request if detected
    provider: custom
    regex: '\b[A-Z]{5}[0-9]{4}[A-Z]\b'
```

### 5.3 Block Strategy (Reject Requests with PII)

Strict mode: reject any request containing sensitive PII rather than allowing it through.

```yaml
# config/agent/runtime/pii.yaml — Strict Blocking

enabled: true
trace_strategy: redact       # Use ***REDACTED*** in traces (maximum privacy)

rules:
  # Block requests containing physical addresses (uses Presidio NLP).
  - name: address
    strategy: block
    provider: presidio       # Requires presidio-analyzer Python package

  # Block requests containing Social Security Numbers.
  - name: ssn
    strategy: block
    provider: custom
    regex: '\b\d{3}-\d{2}-\d{4}\b'

  # Block requests containing credit card numbers.
  - name: credit_card
    strategy: block
    provider: default

  # Emails are still allowed, but scrubbed (not blocked).
  - name: email
    strategy: scrub
    provider: regex
    label: MAIL
```

When `strategy: block` is triggered, the agent returns an error to the user indicating that the request was rejected due to PII detection. The request never reaches the LLM.

### 5.4 Mixed Providers (Default + Regex + Presidio + Custom)

A comprehensive configuration using all four provider types.

```yaml
# config/agent/runtime/pii.yaml — Mixed Providers

enabled: true
trace_strategy: hash

rules:
  # ── Default provider (stock LangChain PIIMiddleware) ────────────
  # One-way detection — runs all default-provider rules in parallel.
  - name: credit_card
    strategy: mask           # ****-****-****-1234
    provider: default

  - name: ip
    strategy: redact         # ***REDACTED***
    provider: default

  - name: url
    strategy: redact
    provider: default

  # ── Regex provider (built-in patterns, reversible token-map) ────
  # Uses built-in regex patterns. Supports scrub (reversible).
  - name: email
    strategy: scrub          # [MAIL_1] -> restored in output
    provider: regex
    label: MAIL

  - name: phone
    strategy: scrub
    provider: regex
    label: PHONE

  # ── Presidio provider (NLP-based detection) ─────────────────────
  # Uses Presidio analyzer for entity recognition. More accurate for
  # unstructured text but requires the presidio-analyzer package.
  - name: address
    strategy: block          # Reject requests with physical addresses
    provider: presidio

  - name: person
    strategy: scrub          # [PERSON_1] -> restored in output
    provider: presidio
    label: PERSON

  # ── Custom provider (your own regex) ────────────────────────────
  # For domain-specific patterns not covered by built-in detectors.
  - name: employee_id
    strategy: scrub
    provider: custom
    regex: '\bEMP-\d{6}\b'
    label: EMP_ID

  - name: internal_project_code
    strategy: redact
    provider: custom
    regex: '\bPROJ-[A-Z]{2,4}-\d{4}\b'

  - name: pan_card
    strategy: block
    provider: custom
    regex: '\b[A-Z]{5}[0-9]{4}[A-Z]\b'
```

---

## 6. `.env` for Langfuse

Langfuse provides LLM trace quality observability (what was asked, what was returned, cost). It auto-activates when its secrets are set as environment variables -- no YAML config needed.

### Development Setup (Minimal)

```bash
# .env — Langfuse Development

# --- Langfuse ---
# Sign up at https://cloud.langfuse.com and create a project to get these keys.
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# "development" tag in Langfuse UI — filters traces by environment.
LANGFUSE_TRACING_ENVIRONMENT=development
```

That is all you need. The Langfuse SDK reads these env vars directly; the agent detects their presence and activates tracing automatically.

### Production Setup (with Encryption, Custom Trace Name)

```bash
# .env — Langfuse Production

# --- Langfuse ---
LANGFUSE_PUBLIC_KEY=pk-lf-prod-xxxx-xxxx-xxxx-xxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-prod-xxxx-xxxx-xxxx-xxxxxxxxxxxx
# Self-hosted Langfuse instance (recommended for production data residency).
LANGFUSE_BASE_URL=https://langfuse.internal.example.com
LANGFUSE_TRACING_ENVIRONMENT=production

# --- User ID Encryption ---
# Encrypt user IDs in Langfuse traces for privacy compliance.
# Without this, SSO usernames appear in plain text in traces.
ENABLE_USER_ID_ENCRYPTION=true
# Generate: python -c "import secrets; print(secrets.token_hex(32))"
USER_ID_ENCRYPTION_KEY=a1b2c3d4e5f6...your-64-char-hex-key...

# --- OpenTelemetry (in addition to Langfuse) ---
# OTEL provides operational metrics (request counts, latency, errors).
# Langfuse provides LLM quality traces. Both can run simultaneously.
ENABLE_OTEL=true
OTEL_EXPORTER_OTLP_ENDPOINT=https://otel-collector.internal.example.com:4317
OTEL_SERVICE_NAME=health-assistant-prod
```

---

## 7. `observability.yaml` Examples

Controls OpenTelemetry (OTEL) metrics and distributed tracing export. Lives at `config/agent/runtime/observability.yaml`. Langfuse is configured entirely via env vars (see Section 6) and does not appear here.

### OTEL with Local Jaeger

For local development, export traces to a Jaeger instance running in Docker.

```yaml
# config/agent/runtime/observability.yaml — Local Jaeger

# Langfuse: auto-activates via env vars (LANGFUSE_PUBLIC_KEY, etc.)
# No YAML config needed for Langfuse.

# OpenTelemetry: export metrics and traces to local Jaeger.
otel:
  enabled: true               # Enable OTEL export
  exporter:
    endpoint: "http://localhost:4317"   # Jaeger OTLP gRPC endpoint
    insecure: true             # No TLS for local dev
  metrics:
    export_interval_ms: 5000   # Push metrics every 5 seconds
  tracing:
    fastapi_auto_instrument: true  # Auto-instrument FastAPI routes
```

Start Jaeger locally:

```bash
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

Then open `http://localhost:16686` to view traces.

### OTEL with Production Collector

For production, export to an OpenTelemetry Collector with authentication.

```yaml
# config/agent/runtime/observability.yaml — Production Collector

otel:
  enabled: true
  exporter:
    endpoint: "https://otel-collector.internal.example.com:4317"
    insecure: false            # TLS enabled in production
  metrics:
    export_interval_ms: 10000  # Push metrics every 10 seconds (lower frequency for prod)
  tracing:
    fastapi_auto_instrument: true
```

Production env vars (set via OpenShift ConfigMap/Secret):

```bash
# These env vars override the YAML values when set:
ENABLE_OTEL=true
OTEL_EXPORTER_OTLP_ENDPOINT=https://otel-collector.internal.example.com:4317
OTEL_EXPORTER_OTLP_INSECURE=false
OTEL_SERVICE_NAME=health-assistant-prod
OTEL_AUTH_TOKEN=your-collector-auth-token
OTEL_METRIC_EXPORT_INTERVAL=10000
```

> **Priority:** Env vars always override YAML values. This is intentional -- OpenShift ConfigMaps inject env vars, so the same `observability.yaml` works across environments without modification.

---

## 8. Custom Middleware Example

Custom middleware hooks into the agent's model call and tool call pipeline. This example shows a complete logging middleware that records every LLM request and response.

### Step 1: Create the Middleware Class

```python
# my_agent/middleware/request_logger.py
"""Custom middleware that logs every LLM model call with timing."""

from __future__ import annotations

import time
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(AgentMiddleware):
    """Log model calls and tool calls with timing information.

    This middleware wraps both sync and async paths. It logs:
    - Model name and message count before each LLM call
    - Latency and status after each LLM call
    - Tool name and arguments before each tool call
    - Tool result status and latency after each tool call
    """

    def _get_model_name(self, request: ModelRequest[Any]) -> str:
        """Extract model name from request."""
        model = request.model
        if isinstance(model, str):
            return model
        return (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or "unknown"
        )

    # ── Async Model Call Hook ─────────────────────────────────────

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[
            [ModelRequest[Any]], Awaitable[ModelResponse[Any]]
        ],
    ) -> ModelResponse[Any]:
        """Async hook: runs before and after every LLM call."""
        model_name = self._get_model_name(request)
        msg_count = len(request.messages)

        logger.info(
            "LLM request: model=%s, messages=%d",
            model_name,
            msg_count,
        )

        started = time.monotonic()
        try:
            response = await handler(request)
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            logger.info(
                "LLM response: model=%s, latency_ms=%.2f, status=success",
                model_name,
                elapsed_ms,
            )
            return response

        except Exception as exc:
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            logger.error(
                "LLM error: model=%s, latency_ms=%.2f, error=%s",
                model_name,
                elapsed_ms,
                exc,
            )
            raise

    # ── Sync Model Call Hook ──────────────────────────────────────

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Sync hook: used by in-process subagents (Runnable.invoke)."""
        model_name = self._get_model_name(request)
        msg_count = len(request.messages)

        logger.info(
            "LLM request (sync): model=%s, messages=%d",
            model_name,
            msg_count,
        )

        started = time.monotonic()
        try:
            response = handler(request)
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            logger.info(
                "LLM response (sync): model=%s, latency_ms=%.2f",
                model_name,
                elapsed_ms,
            )
            return response

        except Exception as exc:
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            logger.error(
                "LLM error (sync): model=%s, latency_ms=%.2f, error=%s",
                model_name,
                elapsed_ms,
                exc,
            )
            raise

    # ── Async Tool Call Hook ──────────────────────────────────────

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            Awaitable[ToolMessage | Command[Any]],
        ],
    ) -> ToolMessage | Command[Any]:
        """Async hook: runs before and after every tool call."""
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("args", {})

        logger.info(
            "Tool call: name=%s, args_keys=%s",
            tool_name,
            sorted(tool_args.keys()) if isinstance(tool_args, dict) else "N/A",
        )

        started = time.monotonic()
        try:
            result = await handler(request)
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            logger.info(
                "Tool result: name=%s, latency_ms=%.2f, status=success",
                tool_name,
                elapsed_ms,
            )
            return result

        except Exception as exc:
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            logger.error(
                "Tool error: name=%s, latency_ms=%.2f, error=%s",
                tool_name,
                elapsed_ms,
                exc,
            )
            raise
```

### Step 2: Register in `agent.yaml`

Add the dotted import path to the `middleware.extra` list. The framework calls `_import_middleware()` which imports the module, finds the class (or factory function), and instantiates it.

```yaml
# In config/agent/runtime/agent.yaml

middleware:
  # ... other middleware settings ...

  extra:
    # Format: "module.path:ClassName" or "module.path:factory_function"
    # The class must subclass AgentMiddleware.
    # The class (or factory) is called with no arguments — use __init__ defaults.
    - "my_agent.middleware.request_logger:RequestLoggingMiddleware"
```

### How It Works

The `_import_middleware()` function in `deep_agent/src/infrastructure/middleware.py` handles loading:

1. Splits the dotted path on `:` into `module_path` and `attr_name`.
2. Calls `importlib.import_module(module_path)` to import the module.
3. Gets the attribute (class or function) via `getattr(module, attr_name)`.
4. If callable, invokes it with no arguments: `factory_or_class()`.
5. Returns the middleware instance, which is appended to the pipeline.

The middleware pipeline executes in order: built-in middleware first (audit, OPA, safety, PII, guardrails), then `extra` middleware in the order listed.

### Available Hooks

| Hook | When it runs | Use case |
|------|-------------|----------|
| `awrap_model_call` / `wrap_model_call` | Around every LLM invocation | Logging, metrics, token counting, request modification |
| `awrap_tool_call` / `wrap_tool_call` | Around every tool invocation | Logging, authorization, input validation |
| `abefore_model` / `before_model` | Before model call (state-based) | PII scrubbing, content injection |
| `aafter_model` / `after_model` | After model call (state-based) | Safety filtering, response modification |

> **Tip:** Look at `deep_agent/src/audit/middleware.py` for a production-quality example of a middleware that audits both model calls and tool calls.

---

## Quick Reference: File Locations

| File | Purpose |
|------|---------|
| `config/agent/runtime/agent.yaml` | Main agent configuration |
| `config/agent/mcp.json` | MCP server connections and auth |
| `config/agent/subagents/*.md` | Subagent definitions (front matter + system prompt) |
| `config/agent/runtime/pii.yaml` | PII detection and scrubbing rules |
| `config/agent/runtime/observability.yaml` | OTEL metrics and tracing |
| `.env` | Secrets and infrastructure endpoints |
| `config/agent/PROMPT.md` | Main agent system prompt and LDAP groups |

For the full schema reference and all available options, see [Configuration Reference](./01-configuration-reference.md).
