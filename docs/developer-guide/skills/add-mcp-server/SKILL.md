---
name: add-mcp-server
description: >
  Guides addition of new MCP server connections. Use when a user asks
  to connect, add, or integrate a new MCP server.
---

# Add MCP Server

Guide the user through adding a new MCP server entry to `config/agent/mcp.json`.

## When to Use

When the user asks to connect, add, or integrate a new MCP server.

## Step-by-Step Checklist

1. **Get the MCP server URL** from the user
2. **Determine the auth mode** — ask the user which applies:
   - `sso` — pass the user's SSO token (default)
   - `oauth` — MCP server has its own OAuth provider
   - `dcr` — MCP server supports dynamic client registration
   - `api_key` — MCP server uses a static API key
3. **Choose a `tool_prefix`** to namespace the server's tools (e.g. `jira`, `confluence`)
4. **Read the current** `config/agent/mcp.json`
5. **Add the new server entry** with the correct fields for the chosen auth mode
6. **Set up any required env vars** (for `api_key` mode or OAuth client secrets)
7. **Validate** the entry against the rules in `references/validation_rules.md`
8. **Write back** to `config/agent/mcp.json`

## Auth Mode Decision

| Scenario | Auth Mode |
|----------|-----------|
| MCP server trusts the user's SSO token | `sso` |
| MCP server has its own OAuth provider | `oauth` |
| MCP server supports dynamic client registration | `dcr` |
| MCP server uses a static API key | `api_key` |

## Resources

- **Field Reference:** `references/field_reference.md`
- **Validation Rules:** `references/validation_rules.md`
- **Templates:** `assets/sso_template.jsonc`, `assets/oauth_template.jsonc`, `assets/dcr_template.jsonc`, `assets/api_key_template.jsonc`

## Critical Requirements

- File is **JSONC** (supports `//` comments)
- `url` is **required** for every server
- `auth_mode: "oauth"` requires `oauth.authorization_endpoint`, `oauth.token_endpoint`, `oauth.client_id`
- `auth_mode: "dcr"` requires `oauth.authorization_endpoint`, `oauth.token_endpoint`, `oauth.registration_endpoint`
- `auth_mode: "dcr"` is **incompatible** with `grant_type: "client_credentials"`
- `auth_mode: "api_key"` requires `auth_env_var`
- **Never put secrets inline** — use `client_secret_env` (env var name), not `client_secret`
- `redirect_uri` is **ignored** (derived from `AGENT_PUBLIC_BASE_URL`)
- `ssl_verify` is **forced `true`** in production
