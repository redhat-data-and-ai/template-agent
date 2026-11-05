# Langfuse Setup Guide

## 1. Overview

Langfuse provides LLM observability for this agent: tracing every model call, tool execution, and user interaction with cost tracking and user feedback collection.

The integration uses the **Langfuse v4 SDK** (`langfuse>=4.9.0`) and auto-activates when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set as environment variables. No YAML configuration is needed -- the entire integration is driven by environment variables and runtime code in `deep_agent/aegra/telemetry.py`.

Key capabilities:

- **LLM call tracing** -- model name, input/output messages, token usage, latency
- **Tool execution tracing** -- tool name, arguments, results as nested spans
- **HITL trace stitching** -- keeps a single trace across human-in-the-loop interrupt/resume cycles
- **Session grouping** -- all traces for a conversation grouped under the thread ID
- **PII scrubbing** -- configurable hashing or redaction of PII before spans reach Langfuse
- **User feedback** -- thumbs up/down scores attached to traces via the `/feedback` endpoint
- **Cost tracking** -- token usage and model costs visible in the Langfuse dashboard

## 2. Quick Setup

### Required Environment Variables

Set these three environment variables in your `.env` file or deployment configuration:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

The Langfuse v4 SDK reads these variables directly -- no further wiring is needed.

### Optional Environment Variables

```bash
# Environment tag for filtering traces in the dashboard (default: development)
LANGFUSE_TRACING_ENVIRONMENT=development

# Override the trace name shown in Langfuse UI
# Defaults to the agent name from agent.yaml
LANGFUSE_TRACE_NAME=my-agent
```

### Verification

After setting the environment variables, start the agent and send a test message. Then verify in the Langfuse dashboard:

1. Navigate to **Traces** in your Langfuse project.
2. Confirm a trace appears with your agent's name (or `LANGFUSE_TRACE_NAME` if set).
3. Expand the trace to see nested spans for LLM calls and tool executions.
4. Check that the **Session** column shows the `thread_id` from your request.

If traces do not appear, check the agent logs for messages containing `Langfuse credentials not set` or `Failed to register Langfuse tracing hook`. The Langfuse SDK loggers are suppressed to ERROR level by default (see [Section 8](#8-logger-noise-suppression)), so raise `PYTHON_LOG_LEVEL=DEBUG` temporarily if you need more detail.

## 3. What Gets Traced

Tracing is implemented in `deep_agent/aegra/telemetry.py` through two mechanisms that work together:

### Globally Registered CallbackHandler

The `setup_langfuse_tracing()` function registers a **process-wide** `HitlAwareCallbackHandler` (a subclass of Langfuse's `CallbackHandler`) via LangChain's `register_configure_hook`. This handler is stored in a `ContextVar` with a default value, so every LangChain run automatically picks it up without explicit wiring.

This callback captures:

| What | Details |
|------|---------|
| **LLM calls** | Model name, input/output messages, token usage (prompt + completion), latency |
| **Tool calls** | Tool name, arguments, results -- captured as nested spans under the LLM call |
| **HITL interrupt/resume** | Same trace is maintained across human-in-the-loop pauses (see [Section 4](#4-hitl-trace-stitching)) |
| **Subagent delegation** | `task` tool calls are classified as `subagent_delegation` in the trace hierarchy |

### LangfuseObservabilityProvider

The `LangfuseObservabilityProvider` class plugs into Aegra's `ObservabilityManager` and injects metadata into every `RunnableConfig`. The Langfuse `CallbackHandler` reads these keys automatically from `RunnableConfig.metadata`:

| Metadata Key | Source | Purpose |
|--------------|--------|---------|
| `langfuse_user_id` | Authenticated user identity (optionally encrypted) | Associates traces with a user |
| `langfuse_session_id` | `thread_id` from the request | Groups all traces for a conversation |
| `thread_id` | Same as above | Resume key for HITL interrupt/resume stitching |
| `langfuse_trace_name` | `agent.yaml` `name` field or `LANGFUSE_TRACE_NAME` env var | Human-readable trace name in the UI |
| `langfuse_tags` | Request trace ID | Tags traces with `trace_id:<value>` for correlation |

### Trace Name Resolution

The trace name is resolved in this order:

1. `agent.yaml` `name` field (via `agent_config.get_name()`)
2. `LANGFUSE_TRACE_NAME` environment variable
3. Fallback: `template-agent`

### Session Grouping

All traces for a conversation are grouped under `langfuse_session_id`, which is set to the `thread_id` from the request. This means every message in a multi-turn conversation appears as a separate trace within the same Langfuse session, making it easy to review the full conversation flow.

## 4. HITL Trace Stitching

### The Problem

LangGraph implements human-in-the-loop (HITL) via soft interrupts (`GraphInterrupt`). When a graph is interrupted and later resumed with `Command(resume=...)`, LangGraph ends the current chain run and starts a new one. Without trace stitching, each interrupt/resume cycle creates a **separate trace** in Langfuse, fragmenting the conversation into disconnected pieces.

### The Solution

The `_build_hitl_aware_handler_class()` function in `telemetry.py` builds a `HitlAwareCallbackHandler` that extends Langfuse's `CallbackHandler` to keep one orchestrator span open across HITL pauses. The mechanism works in three steps:

**1. On `GraphInterrupt` (nested `on_chain_error`):**

- Detects the interrupt by checking for a `GraphInterrupt` exception with a non-null `parent_run_id`.
- Saves the root observation in an LRU-bounded `_open_hitl_roots` dict, keyed by the thread ID (resume key).
- Marks the root run ID in `_keep_open_root_run_ids` so that `span.end()` is skipped.

**2. On `on_chain_end` for an interrupted root:**

- Updates the observation output but does **not** call `span.end()` -- the observation stays open.
- Clears the resume key and propagation context, then resets the root run state.

**3. On `Command(resume=...)` root `on_chain_start`:**

- Detects the resume by checking `_is_langgraph_resume(inputs)`.
- Looks up the open observation from `_open_hitl_roots` using the resume key.
- Rebinds the new LangChain `run_id` to the previously open observation via `_attach_observation()`.
- All subsequent child spans attach under the original orchestrator trace.

### Bounded Storage

The `_open_hitl_roots` dict is an `OrderedDict` with a maximum capacity of 1024 entries (`_MAX_OPEN_HITL_ROOTS`). When the dict is full, the oldest entry is evicted and its observation is ended. This prevents unbounded memory growth from abandoned HITL sessions that are never resumed.

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| **New message while HITL root is open** | The stale open root is ended before starting the new trace |
| **Root hard failure** | The open root is removed from the dict (never left dangling for the next resume) |
| **Rebind failure** | The open root is ended, a warning is logged, and the handler falls through to create a new root span |
| **Eviction on full** | Oldest entries are evicted with `span.end()` called on each |

### Thread Safety Note

The `_open_hitl_roots`, `_keep_open_root_run_ids`, and `_runs` structures use plain dicts/sets without locking. Concurrent LangChain runs sharing the same handler may race on these structures. A per-thread or per-asyncio-task approach would be required for full thread safety.

## 5. PII Scrubbing of Traces

Before span data is exported to Langfuse, PII is scrubbed from all span attributes using the `mask_otel_spans` callback.

### Configuration

PII scrubbing for traces is controlled by the `trace_strategy` field in `pii.yaml` (default: `hash`):

The two strategies:

| Strategy | Replacement | Example | Correlation |
|----------|-------------|---------|-------------|
| `hash` | `[HASH:abc123]` | `john.doe@example.com` becomes `[HASH:a1b2c3]` | Deterministic -- same PII always produces the same hash, enabling cross-request correlation |
| `redact` | `***REDACTED***` | `john.doe@example.com` becomes `***REDACTED***` | No correlation possible |

### Implementation

The `mask_otel_spans` callback is registered on the Langfuse client during `setup_langfuse_tracing()`:

1. The function checks if a PII scrubber is available via `get_scrubber()`.
2. If present, it creates a `_mask_otel_spans` callback that iterates over all span attributes.
3. For each string attribute, it applies either `scrub_for_trace_hash` (hash strategy) or `scrub_one_way` (redact strategy).
4. Modified attributes are returned as `OtelSpanPatch` objects with `set_attributes`.
5. The callback is passed to the `Langfuse()` constructor via the `mask_otel_spans` parameter.

### Startup Order Requirement

PII middleware **must** be initialized before Langfuse setup. The agent's startup sequence enforces this automatically. If PII middleware is not initialized (e.g., `pii.yaml` is missing or `enabled: false`), the span-masking callback is not registered and traces will contain raw PII.

### PII Rules

PII detection rules are defined in the `pii` section of `agent.yaml`. Each rule specifies a name, optional regex, strategy (`scrub`, `hash`, `redact`, `mask`, or `block`), and provider (`regex`, `presidio`, `custom`, or `default`). Only non-default provider rules are passed to the token-map scrubber; `provider: default` rules are handled by the stock PII middleware.

## 6. User Identity in Traces

### How User ID Is Set

The `LangfuseObservabilityProvider.get_metadata()` method sets `langfuse_user_id` from the authenticated user identity. This value appears in the Langfuse dashboard on each trace, enabling per-user filtering and analysis.

### Optional Encryption

For privacy compliance, user IDs can be encrypted before they reach Langfuse:

```bash
ENABLE_USER_ID_ENCRYPTION=true
USER_ID_ENCRYPTION_KEY=your-32-byte-hex-key
```

When enabled, user IDs are hashed with **HMAC-SHA256** (truncated to 16 hex characters). This is deterministic — the same user always maps to the same encrypted ID, enabling user-level analysis without exposing the actual identity. When encryption is disabled (the default), the raw user ID is passed through unchanged.

## 7. Feedback Integration

User feedback (thumbs up/down) is recorded as Langfuse scores via the `/feedback` endpoint, implemented in `deep_agent/aegra/feedback.py`.

### Endpoint

```
POST /feedback
```

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `trace_id` | `string` | Yes | Trace ID to attach feedback to (hex format, no hyphens) |
| `name` | `string` | Yes | Score name/identifier (e.g., `user-rating`, `thumbs-up`, `thumbs-down`) |
| `value` | `float` | Yes | Score value (0.0 to 1.0) |
| `thread_id` | `string` | No | Thread ID for Langfuse trace lookup and Postgres persistence |
| `message_id` | `string` | No | Message ID for Postgres persistence |
| `kwargs` | `object` | No | Additional parameters, including `comment` for free-text feedback |

### Langfuse Score Recording

Feedback is written to Langfuse as a `BOOLEAN` score attached to the relevant trace.

### Trace ID Resolution

Because the Langfuse SDK auto-generates trace IDs that differ from LangGraph run IDs, the agent queries the Langfuse API to find the latest trace in the session using the `thread_id`. If a matching trace is found, its ID is used instead of the client-provided `trace_id`, ensuring feedback scores attach to the correct trace in the dashboard.

### Graceful Degradation

When Langfuse is not configured (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` not set):

- The `/feedback` endpoint still accepts requests and returns `200 OK`.
- Feedback is persisted to **Postgres** (if `thread_id` and `message_id` are provided and a database is configured).
- A log entry `feedback_skipped_langfuse_unconfigured` is emitted.

This means feedback collection works independently of Langfuse availability.

## 8. Logger Noise Suppression

The Langfuse SDK can produce verbose log output during normal operation. To prevent this from cluttering application logs, the following loggers are automatically set to `ERROR` level during logging setup in `deep_agent/utils/pylogger.py`:

```
langfuse
langfuse.client
langfuse.api
langfuse.callback
```

These loggers are part of the `OBSERVABILITY_LOGGERS` set, which is included in `ERROR_ONLY_LOGGERS`. The suppression is applied during `_configure_third_party_loggers()`, called as part of `get_python_logger()`.

This means you will only see Langfuse log messages at ERROR level or above. If you need to debug Langfuse connectivity or behavior, temporarily set `PYTHON_LOG_LEVEL=DEBUG` -- note that this raises the application log level globally, not just for Langfuse.

## 9. Shutdown Flush Behavior

Proper shutdown ensures that all pending Langfuse data (buffered spans, scores) is flushed before the process exits. The agent has two shutdown paths, both implemented in `deep_agent/aegra/shutdown.py`.

### Async Shutdown (SIGTERM Handler)

When the agent receives SIGTERM (normal container shutdown), the `_shutdown_langfuse()` function runs:

1. Calls `get_langfuse_client()` to get the existing client.
2. Runs `client.shutdown()` in a thread via `asyncio.to_thread()`.
3. Wraps the call in `asyncio.wait_for()` with a configurable timeout.

```bash
# Timeout for Langfuse shutdown (default: 5 seconds)
SHUTDOWN_LANGFUSE_TIMEOUT_SECONDS=5
```

### Sync Shutdown (atexit)

The `_shutdown_langfuse_sync()` function runs as an `atexit` callback:

1. Checks if Langfuse is configured via `_langfuse_configured()`.
2. Gets the existing client via `langfuse.get_client()`.
3. Calls `client.shutdown()` (or `client.flush()` if `shutdown` is not available).

**Important:** The atexit handler avoids creating a new Langfuse client during interpreter shutdown. If the client was never initialized, it returns `skipped` rather than attempting to construct one (which would trigger `RuntimeError: cannot schedule new futures`).

### Overall Drain Period

The agent's graceful shutdown sequence is time-budgeted:

```bash
# Total drain period for in-flight requests (default: 15 seconds)
SHUTDOWN_DRAIN_SECONDS=15

# Overall Kubernetes termination grace period (default: 60 seconds)
SHUTDOWN_GRACE_PERIOD_SECONDS=60
```

The shutdown module validates at import time that `SHUTDOWN_DRAIN_SECONDS + SHUTDOWN_LANGFUSE_TIMEOUT_SECONDS` leaves at least 5 seconds of headroom before the Kubernetes SIGKILL deadline.

### Idempotency

Both shutdown paths call idempotent cleanup -- whichever runs first performs the work, and the second is a no-op. The `_shutting_down` and `_shutdown_complete` flags coordinate between the two paths.

## 10. OpenTelemetry Integration (Separate from Langfuse)

The agent has **two independent observability systems** that complement each other:

| System | Purpose | Configuration |
|--------|---------|---------------|
| **Langfuse** | LLM-specific tracing (prompts, completions, token usage, costs) | Environment variables only |
| **OpenTelemetry** | Operational metrics and distributed tracing (HTTP latency, thread counts, stream metrics) | `config/agent/runtime/observability.yaml` + env var overrides |

These systems coexist without conflict. Langfuse traces LLM calls with prompt/completion detail; OpenTelemetry traces infrastructure spans and exports metrics.

### OTEL Configuration

OpenTelemetry configuration lives in `config/agent/runtime/observability.yaml`:

```yaml
otel:
  enabled: false
  exporter:
    endpoint: "http://localhost:4317"
    insecure: true
  metrics:
    export_interval_ms: 5000
  tracing:
    fastapi_auto_instrument: true
```

| Field | Description | Default |
|-------|-------------|---------|
| `otel.enabled` | Enable OTLP export (metrics stay in-memory when false) | `false` |
| `otel.exporter.endpoint` | OTLP gRPC collector endpoint | `http://localhost:4317` |
| `otel.exporter.insecure` | Skip TLS verification for the exporter | `true` |
| `otel.metrics.export_interval_ms` | Metric export interval in milliseconds (1000-60000) | `5000` |
| `otel.tracing.fastapi_auto_instrument` | Enable automatic FastAPI span creation | `true` |

### Environment Variable Overrides

Environment variables take precedence over the YAML configuration:

| Env Var | Overrides |
|---------|-----------|
| `ENABLE_OTEL` | `otel.enabled` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `otel.exporter.endpoint` |
| `OTEL_EXPORTER_OTLP_INSECURE` | `otel.exporter.insecure` |
| `OTEL_METRIC_EXPORT_INTERVAL` | `otel.metrics.export_interval_ms` |

### Key Metrics

The agent exports the following OpenTelemetry metrics (prefixed with the agent name):

- `{prefix}_conversations_total` -- conversation count by status
- `{prefix}_conversation_duration_seconds` -- conversation duration histogram
- `{prefix}_active_conversations` -- currently active conversations
- `{prefix}_messages_total` -- messages sent/received
- `{prefix}_stream_duration_seconds` -- streaming response duration
- `{prefix}_time_to_first_token_seconds` -- time to first streamed token
- `{prefix}_stream_errors_total` -- stream failures by type
- `{prefix}_threads_created_total` -- thread creation count
- `{prefix}_threads_active` -- currently active threads
- `{prefix}_threads_deleted_total` -- thread deletion count
- `{prefix}_graph_build_duration_seconds` -- graph compilation duration

## 11. Environment Variable Reference

### Langfuse Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LANGFUSE_PUBLIC_KEY` | Yes | -- | Langfuse project public key (`pk-lf-...`) |
| `LANGFUSE_SECRET_KEY` | Yes | -- | Langfuse project secret key (`sk-lf-...`) |
| `LANGFUSE_BASE_URL` | Yes | -- | Langfuse server URL (e.g., `https://cloud.langfuse.com`) |
| `LANGFUSE_TRACING_ENVIRONMENT` | No | `development` | Environment tag for trace filtering in the dashboard |
| `LANGFUSE_TRACE_NAME` | No | agent.yaml `name` | Override the trace name shown in the Langfuse UI |

### User Identity Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENABLE_USER_ID_ENCRYPTION` | No | `false` | Encrypt user IDs with HMAC-SHA256 before sending to Langfuse |
| `USER_ID_ENCRYPTION_KEY` | No | -- | 32-byte hex key for user ID encryption (required when encryption is enabled) |

### Shutdown Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SHUTDOWN_LANGFUSE_TIMEOUT_SECONDS` | No | `5` | Timeout for Langfuse flush during shutdown |
| `SHUTDOWN_DRAIN_SECONDS` | No | `15` | Drain period for in-flight requests before cleanup |
| `SHUTDOWN_GRACE_PERIOD_SECONDS` | No | `60` | Overall termination grace period (should match Kubernetes `terminationGracePeriodSeconds`) |

### PII Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PII_HASH_KEY` | No | -- | HMAC key for PII hash strategy (used when `trace_strategy: hash`) |
| `PII_TOKEN_MAP_TTL_DAYS` | No | `7` | TTL in days for PII token map entries (1-365) |

### OpenTelemetry Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENABLE_OTEL` | No | `false` | Enable OpenTelemetry metric and trace export |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | `http://localhost:4317` | OTLP gRPC collector endpoint |
| `OTEL_EXPORTER_OTLP_INSECURE` | No | `true` | Skip TLS verification for OTLP exporter |
| `OTEL_METRIC_EXPORT_INTERVAL` | No | `5000` | Metric export interval in milliseconds |
| `ENABLE_OTEL_METRICS` | No | `false` | Enable OTEL metrics export (separate from `ENABLE_OTEL`) |
| `ENABLE_OTEL_TRACES` | No | `false` | Enable OTEL trace export |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | No | -- | Separate OTLP endpoint for traces (e.g., Jaeger) |
| `OTEL_SERVICE_NAME` | No | `template-agent` | Service name for OTEL resource attributes |
| `OTEL_AUTH_TOKEN` | No | -- | Authentication token for OTLP exporter |
| `OTEL_METRIC_EXPORT_INTERVAL_MILLIS` | No | `10000` | Metric export interval (alternative to `OTEL_METRIC_EXPORT_INTERVAL`) |

## 12. Working Examples

### Development `.env` (Minimal)

```bash
# Langfuse -- minimum required for tracing
LANGFUSE_PUBLIC_KEY=pk-lf-your-dev-key
LANGFUSE_SECRET_KEY=sk-lf-your-dev-secret
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# Optional -- defaults to "development"
LANGFUSE_TRACING_ENVIRONMENT=development
```

### Production `.env` (Full)

```bash
# Langfuse -- required
LANGFUSE_PUBLIC_KEY=pk-lf-your-prod-key
LANGFUSE_SECRET_KEY=sk-lf-your-prod-secret
LANGFUSE_BASE_URL=https://langfuse.your-company.com

# Environment tag for dashboard filtering
LANGFUSE_TRACING_ENVIRONMENT=production

# Custom trace name (overrides agent.yaml name)
LANGFUSE_TRACE_NAME=my-production-agent

# User ID encryption for privacy compliance
ENABLE_USER_ID_ENCRYPTION=true
USER_ID_ENCRYPTION_KEY=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4

# PII hash key for trace scrubbing
PII_HASH_KEY=your-hmac-key-for-pii-hashing

# Shutdown tuning
SHUTDOWN_LANGFUSE_TIMEOUT_SECONDS=10
SHUTDOWN_DRAIN_SECONDS=20
SHUTDOWN_GRACE_PERIOD_SECONDS=60
```

### Cross-Reference

For additional working examples covering the full agent configuration, see [Working Examples](./06-working-examples.md). For the complete configuration reference including YAML settings, see [Configuration Reference](./01-configuration-reference.md).
