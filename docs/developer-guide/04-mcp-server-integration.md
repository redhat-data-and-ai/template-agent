# MCP Server Integration

This guide covers how the agent connects to external MCP (Model Context Protocol) servers, discovers tools at runtime, and manages authentication flows.

---

## 1. MCP Overview

MCP (Model Context Protocol) is a standard protocol that lets LLM-based agents discover and invoke tools hosted by external services. Instead of hardcoding tool definitions, the agent connects to MCP servers at runtime, queries their available tools, and makes them callable by the LLM.

The agent uses [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters) for MCP client functionality. The core implementation lives in `deep_agent/aegra/mcp.py`.

### Connection Lifecycle

1. **Configuration** -- Server definitions are loaded from `config/agent/mcp.json` at startup.
2. **Parallel Connect** -- The agent connects to all enabled MCP servers concurrently via `get_mcp_tools()`.
3. **Tool Discovery** -- Each server's tool catalog is retrieved through the MCP protocol.
4. **Auth Injection** -- A `_TokenInjectorInterceptor` attaches the correct bearer token to every tool invocation (SSO, OAuth, or API key).
5. **Execution** -- When the LLM calls a tool, the request is proxied to the MCP server with proper authentication headers.

Failures at any stage are isolated per-server: if one MCP server is unreachable, the agent continues with tools from the remaining servers.

---

## 2. Configuration in `mcp.json`

### File Location

```
config/agent/mcp.json
```

### Format

The file uses JSONC (JSON with Comments). JavaScript-style `//` line comments are stripped before parsing by `_strip_jsonc_comments()` in the config loader.

### Top-Level Structure

```jsonc
{
    "mcpServers": {
        "<server-name>": {
            // per-server configuration
        }
    }
}
```

Each key under `mcpServers` is a unique server name used throughout the system as an identifier for routing, auth storage, logging, and frontmatter declarations.

### Per-Server Fields

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `url` | string | -- | Yes | MCP server endpoint (e.g., `http://localhost:5001/mcp`) |
| `transport` | string | `"streamable_http"` | No | Transport protocol for the MCP connection |
| `enabled` | boolean | `true` | No | Whether this server is active. Disabled servers are skipped entirely. |
| `auth` | boolean | `false` | No | Whether to send authentication headers with requests |
| `auth_mode` | string | `"sso"` | No | Authentication mode. One of: `sso`, `oauth`, `dcr`, `api_key` |
| `auth_env_var` | string | -- | No | Environment variable name containing the API key (required when `auth_mode: "api_key"`) |
| `ssl_verify` | boolean | `true` | No | SSL certificate verification. Forced `true` in production regardless of config. |
| `timeout` | integer | `30` | No | Connection timeout in seconds per server |
| `tool_prefix` | string | -- | No | Prefix added to tool names from this server (used as the connection key in `MultiServerMCPClient`) |
| `oauth` | object | -- | No | OAuth configuration block (required for `oauth` and `dcr` auth modes) |

### OAuth Sub-Fields

When `auth_mode` is `"oauth"` or `"dcr"`, the `oauth` block configures the OAuth 2.0 flow:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `authorization_endpoint` | string | Yes (except `client_credentials`) | OAuth authorization URL for the user consent screen |
| `token_endpoint` | string | Yes | OAuth token exchange endpoint |
| `registration_endpoint` | string | Yes (DCR only) | Dynamic Client Registration endpoint |
| `client_id` | string | Yes (OAuth only) | Pre-registered OAuth client ID |
| `client_secret_env` | string | No | Environment variable name containing the client secret (preferred over inline) |
| `redirect_uri` | string | Ignored | Always derived from `AGENT_PUBLIC_BASE_URL`. A warning is logged if set. |
| `grant_type` | string | `"authorization_code"` | One of: `authorization_code` (user-facing), `client_credentials` (service-to-service) |
| `scopes` | list[string] | No | OAuth scopes to request |

### Validation Rules

The config loader (`_validate_mcp_server()` in `deep_agent/src/agent/config/loader.py`) enforces these rules at startup:

- `auth_mode` must be one of `sso`, `oauth`, `dcr`, or `api_key`
- `oauth` and `dcr` modes require an `oauth` block
- `dcr` is incompatible with `grant_type: "client_credentials"` (use `oauth` instead)
- `oauth` mode requires `oauth.client_id`
- `dcr` mode requires `oauth.registration_endpoint`
- `token_endpoint` is always required for `oauth`/`dcr`
- `authorization_endpoint` is required unless using `client_credentials` grant
- Inline `client_secret` in mcp.json produces a warning; use `client_secret_env` instead

### Example: Minimal Configuration

```jsonc
{
    "mcpServers": {
        "my-server": {
            "url": "http://localhost:5001/mcp",
            "transport": "streamable_http",
            "enabled": true,
            "auth": true,
            "auth_mode": "sso",
            "timeout": 30
        }
    }
}
```

See also: [Configuration Reference](./01-configuration-reference.md) for the full config file inventory.

---

## 3. Authentication Modes

### 3.1 SSO Pass-Through (`auth_mode: "sso"`)

The simplest mode. The user's SSO/OIDC access token is forwarded directly to the MCP server as a `Bearer` token. No additional OAuth configuration is needed.

**How it works:**

1. The user authenticates with the agent via SSO.
2. `_resolve_connection_token()` returns the SSO token directly.
3. The `_TokenInjectorInterceptor` attaches `Authorization: Bearer <sso_token>` to every MCP tool call.
4. If the token is near expiry (< 60 seconds remaining), `refresh_access_token()` proactively refreshes it using the OIDC refresh grant before forwarding.

**Required configuration:**

- `auth: true`
- `auth_mode: "sso"` (or omit, since it is the default)

**Required environment variables:**

- `SSO_ISSUER_URL` -- OIDC issuer URL for token refresh
- `SSO_CLIENT_ID` -- OIDC client ID for token refresh

**Example `mcp.json`:**

```jsonc
{
    "mcpServers": {
        "internal-api": {
            "url": "https://internal-api.example.com/mcp",
            "transport": "streamable_http",
            "enabled": true,
            "auth": true,
            "auth_mode": "sso",
            "timeout": 30
        }
    }
}
```

### 3.2 OAuth (`auth_mode: "oauth"`)

Standard OAuth 2.0 with a pre-registered client. The agent uses a known `client_id` and optional `client_secret` to obtain tokens.

**How it works:**

1. On first use, the agent detects no stored token and raises `NeedsAuthorization`.
2. The UI shows a "Connect" button. The user clicks it, triggering `POST /mcp/{name}/connect`.
3. The agent builds an authorization URL with PKCE (`S256`) and redirects the user's browser.
4. After consent, the OAuth server redirects to `GET /mcp/oauth/callback` with an authorization code.
5. The agent exchanges the code for tokens and stores them encrypted in Redis.
6. Subsequent requests use the stored token; expired tokens are refreshed automatically.

**Required configuration:**

- `oauth.authorization_endpoint` -- the OAuth consent screen URL
- `oauth.token_endpoint` -- the token exchange endpoint
- `oauth.client_id` -- pre-registered client identifier
- `oauth.client_secret_env` -- environment variable for the client secret (never inline)

**Grant types:**

- `authorization_code` (default) -- interactive, user-facing OAuth flow with PKCE
- `client_credentials` -- service-to-service, no user interaction needed

**Example `mcp.json` (authorization_code):**

```jsonc
{
    "mcpServers": {
        "external-service": {
            "url": "https://api.external.com/mcp",
            "transport": "streamable_http",
            "enabled": true,
            "auth": true,
            "auth_mode": "oauth",
            "timeout": 30,
            "tool_prefix": "ext",
            "oauth": {
                "authorization_endpoint": "https://auth.external.com/authorize",
                "token_endpoint": "https://auth.external.com/token",
                "client_id": "agent-client-id",
                "client_secret_env": "EXTERNAL_OAUTH_CLIENT_SECRET",
                "grant_type": "authorization_code",
                "scopes": ["read", "write"]
            }
        }
    }
}
```

**Example `mcp.json` (client_credentials):**

```jsonc
{
    "mcpServers": {
        "backend-service": {
            "url": "https://backend.internal.com/mcp",
            "transport": "streamable_http",
            "enabled": true,
            "auth": true,
            "auth_mode": "oauth",
            "timeout": 30,
            "oauth": {
                "token_endpoint": "https://auth.internal.com/token",
                "client_id": "service-client-id",
                "client_secret_env": "BACKEND_OAUTH_SECRET",
                "grant_type": "client_credentials",
                "scopes": ["api.read"]
            }
        }
    }
}
```

With `client_credentials`, no `authorization_endpoint` is needed and no user interaction occurs. The agent acquires a token automatically at runtime using a distributed Redis lock to prevent concurrent token requests.

### 3.3 DCR -- Dynamic Client Registration (`auth_mode: "dcr"`)

The agent dynamically registers itself as an OAuth client with the authorization server. This avoids pre-registering a `client_id` -- the server issues one at runtime.

**How it works:**

1. On `POST /mcp/{name}/connect`, the agent checks if a DCR client record exists in Postgres.
2. If not, it calls the `registration_endpoint` to register a new client, receiving `client_id` and `client_secret`.
3. The credentials are encrypted and stored in the `mcp_oauth_clients` Postgres table.
4. The authorization flow then proceeds identically to the OAuth `authorization_code` flow.

**Required configuration:**

- `oauth.authorization_endpoint`
- `oauth.token_endpoint`
- `oauth.registration_endpoint` -- the RFC 7591 registration endpoint

**Constraints:**

- DCR is incompatible with `grant_type: "client_credentials"` -- use `auth_mode: "oauth"` for service-to-service flows.
- The redirect URI is derived from the `AGENT_PUBLIC_BASE_URL` environment variable (specifically `{AGENT_PUBLIC_BASE_URL}/mcp/oauth/callback`). Do not set `redirect_uri` in mcp.json.
- DCR must be enabled at the application level via `MCP_DCR_ENABLED=true`.

**Example `mcp.json`:**

```jsonc
{
    "mcpServers": {
        "partner-api": {
            "url": "https://partner.example.com/mcp",
            "transport": "streamable_http",
            "enabled": true,
            "auth": true,
            "auth_mode": "dcr",
            "timeout": 30,
            "tool_prefix": "partner",
            "oauth": {
                "authorization_endpoint": "https://partner.example.com/auth/authorize",
                "token_endpoint": "https://partner.example.com/auth/token",
                "registration_endpoint": "https://partner.example.com/auth/register",
                "scopes": ["email", "openid", "profile"]
            }
        }
    }
}
```

### 3.4 API Key (`auth_mode: "api_key"`)

Simple API key authentication. The key is read from an environment variable and sent as a `Bearer` token.

**How it works:**

1. `_resolve_connection_token()` reads the API key from the environment variable named in `auth_env_var`.
2. The key is sent as `Authorization: Bearer <api_key>` on every request.
3. No token refresh or OAuth flow is involved.

**Required configuration:**

- `auth_env_var` -- environment variable name containing the API key

**Example `mcp.json`:**

```jsonc
{
    "mcpServers": {
        "third-party-api": {
            "url": "https://api.thirdparty.com/mcp",
            "transport": "streamable_http",
            "enabled": true,
            "auth": true,
            "auth_mode": "api_key",
            "auth_env_var": "THIRD_PARTY_API_KEY",
            "timeout": 30,
            "tool_prefix": "tp"
        }
    }
}
```

Set the environment variable:

```bash
export THIRD_PARTY_API_KEY="sk-abc123..."
```

---

## 4. Token Management

Token storage and encryption are handled by `McpTokenStore` (`deep_agent/aegra/mcp_token_store.py`) and `mcp_crypto` (`deep_agent/aegra/mcp_crypto.py`).

### Token Encryption

All OAuth tokens (access tokens, refresh tokens, client secrets) are encrypted at rest using Fernet symmetric encryption before being stored in Redis or Postgres.

| Environment Variable | Required | Description |
|---------------------|----------|-------------|
| `MCP_TOKEN_ENCRYPTION_KEY` | Yes (for OAuth/DCR) | Fernet encryption key for token storage |
| `MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS` | No | Previous Fernet key for seamless key rotation |

Generate a Fernet key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### Token Storage

Tokens are stored in two backends depending on their type:

| Data | Backend | Key Pattern |
|------|---------|-------------|
| OAuth user tokens (access, refresh, expiry, scopes) | Redis (persistent, no TTL) | `mcp_oauth_token:{agent}:{user}:{mcp}` |
| DCR client credentials (client_id, client_secret) | Postgres (`mcp_oauth_clients` table) | `(agent_name, mcp_name)` primary key |

Redis is used for user tokens because they are frequently read and refreshed. Postgres is used for DCR client records because they are long-lived and rarely change.

### Token Refresh

The agent proactively refreshes tokens before they expire:

**SSO tokens** (`refresh_access_token()` in `mcp.py`):
- Checks if the JWT has fewer than 60 seconds remaining
- Uses a distributed Redis lock (`sso:refresh:{user_id}`) to prevent concurrent refresh races
- Falls back to an in-process `asyncio.Lock` when Redis is unavailable
- Cross-task shared token cache (`_user_token_cache`) lets concurrent requests see a fresh token immediately

**OAuth/DCR tokens** (`_refresh_mcp_token()` in `mcp_auth.py`):
- Checks if `expires_at` is within 30 seconds of now
- Uses a distributed lock (`mcp_token_refresh:{agent}:{user}:{mcp}`)
- If the lock is held by another task, polls storage for up to 5 seconds waiting for the refreshed token
- On failure, raises `NeedsAuthorization` to re-trigger the OAuth flow

### Key Rotation

To rotate the encryption key without downtime:

1. Set `MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS` to the current key value
2. Set `MCP_TOKEN_ENCRYPTION_KEY` to the new key
3. Deploy -- decryption tries the primary key first, then falls back to the previous key
4. After all tokens have been re-encrypted (on next refresh), remove `MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS`

---

## 5. Auth Wrapping and UI Interrupts

The `wrap_mcp_tools_for_auth()` function in `deep_agent/aegra/mcp_tool_auth.py` wraps every MCP tool so that authentication failures become resumable LangGraph interrupts instead of hard errors.

### How It Works

1. Every tool returned by `get_mcp_tools()` is wrapped by `_wrap_single_tool()`.
2. When a wrapped tool is invoked and `NeedsAuthorization` is raised, the wrapper catches it and calls LangGraph's `interrupt()` with a JSON payload:

```json
{
    "type": "mcp_auth_required",
    "mcp_name": "partner-api",
    "connect_url": "https://agent.example.com/mcp/partner-api/connect",
    "message": "Connect to partner-api to use these tools"
}
```

3. The UI receives this interrupt and shows a "Connect" button.
4. The user clicks "Connect", which opens a browser tab to the OAuth authorization URL.
5. After successful authorization, the OAuth callback stores the token and posts a `mcp_oauth_done` message to the opener window.
6. The UI resumes the conversation, and the tool is re-invoked -- this time with a valid token.

### Gemini JSON String Fix

The wrapper also applies `_fix_stringified_json_args()` before every tool invocation. Some models (notably Gemini) serialize nested objects as JSON strings instead of proper dicts when calling tools with complex input schemas. This function detects string-valued arguments whose schema type is `object` or `array` and parses them into their proper types. Arguments explicitly typed as `string` are never modified.

### Error Handling

The `ainvoke` wrapper (`_make_safe_ainvoke`) also catches non-auth MCP errors and returns a `ToolMessage` with `status="error"` instead of crashing the agent:

```
[TOOL_ERROR] tool_name failed: <error message>
```

This prevents a single tool failure from terminating the entire conversation.

---

## 6. SSL and Timeouts

### SSL Verification

Each server can set `ssl_verify` independently. In production, SSL verification is always enforced regardless of the config value. The `mcp_httpx_verify()` function logs an error and returns `true` when a production server has `ssl_verify: false`.

In development, setting `ssl_verify: false` disables certificate verification for that server's `httpx.AsyncClient`. This is useful for local MCP servers with self-signed certificates.

### Timeouts

The `timeout` field (default: 30 seconds) controls the connection timeout per server. This applies to:

- Initial tool discovery (`_connect_single_server`)
- The overall `asyncio.timeout()` wrapping the `MultiServerMCPClient` connection

If a connection times out, the agent retries once (2 attempts total) before recording a circuit breaker failure and moving on.

---

## 7. Circuit Breaker and Fault Tolerance

The MCP subsystem is designed for graceful degradation. A single failing server never takes down the agent.

### Circuit Breaker

`_get_mcp_breaker()` in `mcp.py` creates a circuit breaker (`CircuitBreaker`) for MCP connections:

- **Threshold:** 5 consecutive failures open the circuit
- **Reset timeout:** 60 seconds before the circuit transitions to half-open
- **Scope:** Global across all MCP servers (single breaker instance)
- When open, all MCP connections are skipped with a warning log

Each successful connection records a success; each failure (timeout, connection error) records a failure.

### Tool List Caching

Tool lists are cached per-user for `cache.mcp.ttl` seconds (configured in `agent.yaml`, default 300s). Subsequent requests within the TTL window return cached tools without reconnecting, eliminating approximately 3-4 seconds of overhead per request.

Cache keys are structured as `{user_id}:{comma-separated-server-names}`. The cache can be invalidated explicitly via `invalidate_mcp_tool_cache(user_id)`, which happens automatically after OAuth callback and disconnect events.

### Retry

Each server connection attempt is retried once on timeout (2 attempts total in `_connect_single_server`).

### Graceful Degradation

Connection failures return an empty tool list for that server. The agent continues with tools from other servers and any built-in tools. Specific behaviors by error type:

| Error Type | Behavior |
|-----------|----------|
| `NeedsAuthorization` | Returns an auth placeholder tool (see below) |
| HTTP 401/403 (OAuth/DCR server) | Returns an auth placeholder tool |
| HTTP 401/403 (SSO server) | Logs warning, returns empty list |
| Connection refused (optional server) | Records circuit breaker failure, returns empty list |
| Timeout (after 2 attempts) | Records circuit breaker failure, returns empty list |
| Other errors | Records circuit breaker failure, logs error with traceback, returns empty list |

### Auth Placeholder Tools

When an OAuth/DCR server has no usable token, `_create_auth_placeholder_tool()` injects a stub tool named `mcp__<server_name>`. When the LLM calls this stub:

1. It first attempts to resolve a token (in case one was obtained since discovery).
2. If successful, it invalidates the tool cache and asks the user to repeat their request.
3. If no token is available, it raises `NeedsAuthorization`, triggering the OAuth interrupt flow.

---

## 8. MCP Resource Tools

In addition to server-defined tools, the agent provides three built-in tools for reading MCP resources. These are defined in `deep_agent/aegra/mcp_resource_tools.py`.

### Available Tools

| Tool Name | MCP Method | Description |
|-----------|-----------|-------------|
| `mcp_list_resources` | `resources/list` | List available concrete resources from an MCP server |
| `mcp_list_resource_templates` | `resources/templates/list` | List URI templates with `{param}` placeholders |
| `mcp_read_resource` | `resources/read` | Read a resource by URI with line-based pagination |

### Reading Resources

`mcp_read_resource` extracts `contents[].text` from the MCP response and applies line-based pagination:

- `offset` (default: 0) -- zero-indexed line number to start from
- `limit` (default: 100) -- maximum number of lines to return
- Output is character-truncated to a 400,000 character budget (100K tokens at 4 chars/token)
- Binary blob contents are replaced with `[Binary content omitted (mime/type)]`

### Resource URI Allowlists

Subagents can restrict which resource URIs they access via the `resources` frontmatter field:

```yaml
---
resources:
  - "docs://guides/{topic}"
  - "config://settings"
---
```

When `resources` is set, only listed URIs (and URI templates) are accessible. Omitting the field means unrestricted access to all resources.

---

## 9. Health Checks

Per-server health monitoring is implemented in `deep_agent/aegra/mcp_health.py`.

### Behavior

- Pings each enabled MCP server's URL with an HTTP `GET`
- Any response with HTTP status < 500 is considered healthy
- Results are cached for **30 seconds** to avoid hammering dependencies
- Circuit breaker state is checked: if open, all servers report `breaker-open` without pinging
- Health check timeout is capped at 5 seconds (or the server's configured timeout, whichever is lower)

### OTEL Integration

When metrics export is enabled (`ENABLE_OTEL_METRICS=true`), the health checker emits an OpenTelemetry gauge:

- **Gauge name:** `mcp_server_health`
- **Values:** `1` (healthy) or `0` (unhealthy)
- **Attribute:** `mcp.server` = server name

### Aggregate Status

The overall MCP health is either `"ok"` (all servers healthy) or `"warning"` (some servers unhealthy). It is never `"error"`, so MCP failures cause the agent to report *degraded* rather than *unhealthy*.

---

## 10. HTTP API Routes

The agent exposes HTTP endpoints for OAuth flows and MCP resource/tool proxying in `deep_agent/aegra/mcp_routes.py`.

### OAuth Flow Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/mcp/{name}/connect` | Start the OAuth/DCR authorization flow. Returns `{ "authorize_url": "..." }`. |
| `GET` | `/mcp/oauth/callback` | OAuth redirect handler. Exchanges the authorization code for tokens, stores them, and returns an HTML page that notifies the opener window via `postMessage`. |
| `GET` | `/mcp/{name}/status` | Returns `{ "mcp_name": "...", "connected": true/false }` for the authenticated user. |
| `DELETE` | `/mcp/{name}/disconnect` | Clears stored OAuth tokens for the MCP server for the authenticated user. |
| `GET` | `/mcp/oauth/connections` | Returns connection status for all OAuth/DCR MCP servers. |
| `POST` | `/mcp/{name}/reregister` | Re-register DCR client credentials (admin/developer only). |

### Resource and Tool Proxy Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/mcp/{name}/resources/list` | Proxy MCP `resources/list` for app-side resource browsing |
| `POST` | `/mcp/{name}/resources/templates/list` | Proxy MCP `resources/templates/list` |
| `POST` | `/mcp/{name}/resources/read` | Proxy MCP `resources/read` with a `{ "uri": "..." }` body |
| `POST` | `/mcp/{name}/tools/list` | Proxy MCP `tools/list` for app metadata |
| `POST` | `/mcp/{name}/tools/call` | Proxy MCP `tools/call` for app-initiated tool execution. Enforces that the tool's `visibility` includes `app`. |

All endpoints require authentication (Bearer token in the `Authorization` header). In development mode with auth disabled, a hardcoded dev user ID is used.

---

## 11. Wiring MCP Tools to Agents

The `agent()` function in `deep_agent/aegra/graph.py` is the per-request graph factory that wires MCP tools into the agent.

### Orchestrator Wiring

On each request, the graph factory loads tools from all enabled MCP servers (or those declared in frontmatter), wraps them with auth interrupt handling, resolves frontmatter tool names to tool objects, and appends MCP resource tools (list, templates, read). This entire pipeline runs automatically — no Python code is needed.

### Frontmatter Declarations

The orchestrator and subagents use frontmatter to control MCP server access:

```yaml
---
name: my-agent
mcps:
  - template-mcp-server
  - partner-api
tools:
  - template__search_docs
  - template__create_ticket
resources:
  - "docs://guides/{topic}"
---
```

- **`mcps`** -- list of MCP server names (keys from `mcp.json`). Only these servers are connected. When omitted, the orchestrator connects to all enabled servers; subagents inherit the orchestrator's MCP servers.
- **`tools`** -- list of specific tool names to expose to this agent. Matched against the tools discovered from MCP servers and built-in tools.
- **`resources`** -- list of allowed resource URIs or URI templates. Restricts `mcp_read_resource` to only these patterns.

### Subagent Inheritance

Subagents inherit MCP servers from the orchestrator unless they declare their own `mcps:` list in frontmatter. This means:

- If the orchestrator declares `mcps: [server-a, server-b]` and a subagent has no `mcps:` field, the subagent can access tools from both servers.
- If the subagent declares `mcps: [server-a]`, it can only access tools from `server-a`.
- If the subagent declares `mcps: []`, it has no MCP tools.

### Tool Resolution Flow

1. `get_mcp_tools()` connects to servers and returns raw tool objects.
2. `wrap_mcp_tools_for_auth()` wraps each tool with auth interrupt handling.
3. `agent_config.resolve_tools()` matches frontmatter `tools:` names against the available tool objects.
4. Any MCP tools from declared servers that are not in the `tools:` list are added as extras (when `mcps:` is declared and tools were loaded).

This allows agents to either pick specific tools by name or accept all tools from declared servers.

---

## Further Reading

- [Configuration Reference](./01-configuration-reference.md) -- full inventory of all config files
- [Working Examples](./06-working-examples.md) -- end-to-end examples with MCP servers
