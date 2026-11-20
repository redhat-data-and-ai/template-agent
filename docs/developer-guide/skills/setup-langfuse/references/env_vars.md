# Langfuse Environment Variables

Complete reference for all Langfuse-related environment variables.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LANGFUSE_PUBLIC_KEY` | Yes | — | Langfuse public key (from project settings) |
| `LANGFUSE_SECRET_KEY` | Yes | — | Langfuse secret key (from project settings) |
| `LANGFUSE_BASE_URL` | Yes | — | Langfuse server URL (e.g. `https://cloud.langfuse.com`) |
| `LANGFUSE_TRACING_ENVIRONMENT` | No | `development` | Environment tag for filtering traces in the dashboard |
| `LANGFUSE_TRACE_NAME` | No | agent.yaml `name` field (fallback: `template-agent`) | Override the trace name shown in the Langfuse UI |
| `ENABLE_USER_ID_ENCRYPTION` | No | `false` | Encrypt user IDs in traces using HMAC-SHA256 |
| `USER_ID_ENCRYPTION_KEY` | No | — | 32-byte hex key for user ID encryption (required if encryption enabled) |
| `SHUTDOWN_LANGFUSE_TIMEOUT_SECONDS` | No | `5` | Seconds to wait for pending traces to flush on shutdown |

## Source References

- `deep_agent/src/settings.py` — `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `LANGFUSE_TRACING_ENVIRONMENT` defaults (all `None`/`development`)
- `deep_agent/aegra/telemetry.py` — `_langfuse_configured()` checks for public + secret key; `_get_trace_name()` resolves trace name from agent.yaml > env var > `template-agent`
- `deep_agent/aegra/shutdown.py` — `SHUTDOWN_LANGFUSE_TIMEOUT_SECONDS` read from env with default `5`
- `deep_agent/aegra/auth.py` — `ENABLE_USER_ID_ENCRYPTION` and `USER_ID_ENCRYPTION_KEY` drive HMAC-SHA256 hashing of user IDs
- `.env.example` — documented examples for all variables
