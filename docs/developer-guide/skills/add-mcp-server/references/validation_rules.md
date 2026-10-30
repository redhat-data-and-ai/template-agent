# MCP Server Validation Rules

Rules enforced by `_validate_mcp_server()` in `deep_agent/src/agent/config/loader.py`.

## Auth Mode Validation

- `auth_mode` must be one of: `sso`, `oauth`, `dcr`, `api_key`
- If `auth_mode` is not `oauth` or `dcr`, no further OAuth validation is performed

## OAuth/DCR Block Validation

These rules apply when `auth_mode` is `oauth` or `dcr`:

1. **OAuth block required** — `auth_mode` `oauth` or `dcr` requires an `oauth` object in the server config
2. **`oauth.token_endpoint` required** — always required for `oauth` and `dcr` modes
3. **`oauth.authorization_endpoint` required** — required unless `grant_type` is `client_credentials`
4. **`oauth.client_id` required for `oauth`** — required when `auth_mode` is `oauth`
5. **`oauth.registration_endpoint` required for `dcr`** — required when `auth_mode` is `dcr`

## Invalid Combinations

- **`auth_mode: "dcr"` + `grant_type: "client_credentials"`** is invalid. DCR is incompatible with client credentials grant. Use `auth_mode: "oauth"` instead.

## Warnings (Non-Fatal)

- **`oauth.redirect_uri` is ignored** — if set, a warning is logged. The redirect URI is always derived from `AGENT_PUBLIC_BASE_URL`.
- **`oauth.client_secret` is insecure** — if set inline, a warning is logged. Use `oauth.client_secret_env` with an environment variable name instead.

## Validation Order

The validator checks in this order:

1. `auth_mode` is valid
2. `oauth` block exists (for `oauth`/`dcr`)
3. `dcr` + `client_credentials` incompatibility
4. Required endpoint fields (`token_endpoint`, `authorization_endpoint`)
5. `redirect_uri` warning
6. `client_id` required for `oauth`
7. `client_secret` inline warning
8. `registration_endpoint` required for `dcr`
