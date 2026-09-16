---
name: setup-langfuse
description: >
  Guides Langfuse observability setup and configuration. Use when a
  user asks to set up, configure, or troubleshoot Langfuse tracing.
---

# Langfuse Observability Setup

Walk the user through enabling Langfuse tracing for their agent deployment.

## When to Use

When the user asks to set up, configure, enable, or troubleshoot Langfuse observability tracing.

## Step-by-Step Checklist

1. **Verify Langfuse account** — confirm the user has a Langfuse account and project created (cloud or self-hosted).
2. **Set required env vars** — all three are mandatory:
   - `LANGFUSE_PUBLIC_KEY` — from project settings
   - `LANGFUSE_SECRET_KEY` — from project settings
   - `LANGFUSE_BASE_URL` — Langfuse server URL (e.g. `https://cloud.langfuse.com`)
3. **Optionally set tracing environment** — `LANGFUSE_TRACING_ENVIRONMENT` (default: `development`). Used to filter traces by environment in the dashboard.
4. **Optionally override trace name** — `LANGFUSE_TRACE_NAME` (defaults to the `name` field in `agent.yaml`; falls back to `template-agent`).
5. **Configure PII scrubbing** (if needed) — edit `config/agent/runtime/pii.yaml`:
   - Set `enabled: true`
   - Set `trace_strategy: hash` (correlatable) or `trace_strategy: redact` (maximum privacy)
   - See `references/pii_trace_strategy.md` for details
6. **Configure user ID encryption** (if needed) — set `ENABLE_USER_ID_ENCRYPTION=true` and provide `USER_ID_ENCRYPTION_KEY` (32-byte hex key). User IDs in traces are HMAC-SHA256 hashed.
7. **Add env vars** — to `.env` file for local dev, or deployment config (k8s Secret / ConfigMap) for production.
8. **Verify** — start the agent, send a test message, and check the Langfuse dashboard for traces. Look for the log line `Langfuse auto-tracing registered` on startup.

## What Gets Traced Automatically

| Data | Details |
|------|---------|
| LLM calls | Model name, input/output tokens, latency |
| Tool calls | Tool name, arguments, result |
| HITL interrupt/resume | Stitched as a single trace across interrupt and resume |
| Session grouping | Grouped by `thread_id` for conversation continuity |
| User feedback scores | Thumbs up/down synced to Langfuse |

## Resources

- **Environment Variables:** `references/env_vars.md`
- **PII Trace Strategy:** `references/pii_trace_strategy.md`
- **Troubleshooting:** `references/troubleshooting.md`

## Critical Requirements

- **Auto-activation** — Langfuse activates automatically when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. No YAML config needed; purely env-var driven.
- **PII middleware order** — PII middleware must initialize before Langfuse so the `mask_otel_spans` callback can scrub span data before export. This is handled automatically by the startup order in `telemetry.py`.
- **Shutdown flush** — on shutdown, pending traces are flushed with a configurable timeout: `SHUTDOWN_LANGFUSE_TIMEOUT_SECONDS` (default: 5s). Increase if traces are lost on shutdown.
- **SDK log suppression** — Langfuse SDK loggers are suppressed to ERROR by default to reduce noise.
