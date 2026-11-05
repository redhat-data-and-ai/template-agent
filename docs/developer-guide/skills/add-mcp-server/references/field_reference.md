# MCP Server Field Reference

Complete per-server field table for entries in `config/agent/mcp.json`.

## Server-Level Fields

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `url` | string | — | Yes | MCP server endpoint |
| `transport` | string | `streamable_http` | No | Transport protocol |
| `enabled` | boolean | `true` | No | Whether the server is active |
| `auth` | boolean | `false` | No | Whether to send auth headers |
| `auth_mode` | string | `sso` | No | Auth strategy: `sso`, `oauth`, `dcr`, `api_key` |
| `auth_env_var` | string | — | Cond. | Env var holding the API key (required for `api_key` mode) |
| `ssl_verify` | boolean | `true` | No | SSL certificate verification (forced `true` in production) |
| `timeout` | integer | `30` | No | Connection timeout in seconds |
| `tool_prefix` | string | — | No | Prefix added to all tool names from this server |

## OAuth Block Fields (`oauth.*`)

These fields live inside the `oauth` object on the server entry.

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `oauth.authorization_endpoint` | string | — | Cond. | Authorization URL. Required for `oauth`/`dcr` modes (except `client_credentials` grant) |
| `oauth.token_endpoint` | string | — | Cond. | Token URL. Required for `oauth`/`dcr` modes |
| `oauth.registration_endpoint` | string | — | Cond. | DCR registration URL. Required for `dcr` mode |
| `oauth.client_id` | string | — | Cond. | OAuth client ID. Required for `oauth` mode |
| `oauth.client_secret_env` | string | — | No | Env var name holding the client secret (never inline the secret) |
| `oauth.grant_type` | string | `authorization_code` | No | Grant type: `authorization_code` or `client_credentials` |
| `oauth.scopes` | list[string] | — | No | OAuth scopes to request |

## Conditional Requirements Summary

| Auth Mode | Required Fields |
|-----------|----------------|
| `sso` | `url` |
| `oauth` | `url`, `oauth.authorization_endpoint`, `oauth.token_endpoint`, `oauth.client_id` |
| `dcr` | `url`, `oauth.authorization_endpoint`, `oauth.token_endpoint`, `oauth.registration_endpoint` |
| `api_key` | `url`, `auth_env_var` |

> **Note:** `oauth.authorization_endpoint` is not required when `grant_type` is `client_credentials`, but `dcr` mode is incompatible with `client_credentials`.
