# Observability Guide

This guide covers OpenTelemetry (OTEL) metrics and tracing for the template agent.

## Overview

The agent supports three complementary observability layers:

1. **Langfuse** — LLM-specific tracing (prompts, completions, tokens, costs)
2. **Agent lifecycle OTEL** (`deep_agent/aegra/otel.py`) — conversations, streams, threads, graph builds
3. **Token budget OTEL** (`deep_agent/src/observability/otel_setup.py`) — per-thread/per-user token usage export

Langfuse and OTEL coexist without conflict. Langfuse traces LLM calls; OTEL traces infrastructure and exports operational metrics.

## Architecture

The agent exports telemetry **directly** to OTLP backends. There is no in-repo OTEL Collector deployment.

```
Local dev:   Agent --OTLP/gRPC--> Jaeger (:4317)
OpenShift:   Agent --OTLP/gRPC--> otel-gateway / managed observability service
```

Agent metrics export via OTLP, not Prometheus scrape.

## Local Development

### Quick Start

Start Jaeger:

```bash
docker compose --profile observability up
```

This launches **Jaeger** — trace UI at http://localhost:16686 (OTLP gRPC on `:4317`).

### Enable OTEL in Agent

Uncomment in `compose.yaml` under `template-agent`:

```yaml
environment:
  - ENABLE_OTEL=true
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
  - ENABLE_OTEL_TRACES=true
  - OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://jaeger:4317
```

Or set in `.env`:

```bash
ENABLE_OTEL=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
ENABLE_OTEL_TRACES=true
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://jaeger:4317
```

### View Traces

**Jaeger UI**: http://localhost:16686

1. Select the service name from your agent config (for example `health_assistant`)
2. Click **Find Traces**
3. Inspect HTTP spans (FastAPI auto-instrumentation) and custom spans

### View Metrics

Agent lifecycle metrics export via OTLP when `ENABLE_OTEL=true`. For local debugging, use the in-memory snapshot API exposed through the OTEL module (`get_metrics_snapshot()`).

Token budget metrics export separately when `ENABLE_OTEL_METRICS=true` and `OTEL_EXPORTER_OTLP_ENDPOINT` are set (see token budget docs in `deep_agent/src/token_budget/`).

## OpenShift Deployment

### Configuration

Enable via ConfigMap (env vars override `config/agent/runtime/observability.yaml`):

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
data:
  ENABLE_OTEL: "true"
  OTEL_EXPORTER_OTLP_ENDPOINT: "otel-gateway:4327"
  ENABLE_OTEL_TRACES: "true"
  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: "jaeger-collector.observability.svc:4317"
  OTEL_AUTH_TOKEN: "<bearer-token-if-required>"
```

Point endpoints at your cluster's OTLP ingress or managed observability backend. No collector manifests ship with this template.

### Verify OTEL Setup

```bash
oc logs deployment/agent | grep -i otel
curl -s http://<agent-service>/health | jq .checks.otel
```

The health check reports initialization status, enabled flag, endpoint, and SDK version.

## Available Metrics

Metric names use a **dynamic prefix** derived from the agent display name in config (for example `health_assistant_conversations_total` for agent name "Health Assistant").

### Conversations

- `{prefix}_conversations_total{status}` — conversations by status
- `{prefix}_active_conversations` — currently active conversations
- `{prefix}_conversation_duration_seconds` — conversation duration

### Messages

- `{prefix}_messages_total{direction,message_type}` — messages sent/received

### Streaming

- `{prefix}_stream_tokens_total` — tokens streamed
- `{prefix}_stream_duration_seconds` — stream duration
- `{prefix}_stream_errors_total{error_type}` — stream failures
- `{prefix}_time_to_first_token_seconds` — time to first token

### Threads

- `{prefix}_threads_created_total` — threads created
- `{prefix}_threads_active` — active threads
- `{prefix}_threads_deleted_total` — threads deleted
- `{prefix}_thread_messages_count` — messages per thread

### Graph builds

- `{prefix}_graph_build_duration_seconds{cache_hit,mcp_tool_count,...}` — graph compilation timing (wired in `graph.py`)

## Instrumentation Status

The OTEL module defines `record_*` helpers for lifecycle metrics. Most are **not yet wired** to runtime handlers.

### Working now

- OTLP export when `ENABLE_OTEL=true`
- FastAPI distributed tracing (HTTP spans, W3C trace context)
- Graph build metrics (`record_graph_built()` in `graph.py`)
- In-memory metric snapshot for debugging
- Health check OTEL status (`/health` → `checks.otel`)
- Token budget OTEL export (separate flags — see `ENABLE_OTEL_METRICS`)

### Requires wiring

Add instrumentation calls at:

1. **Conversations** — `record_conversation_started()` / `record_conversation_completed()`
2. **Messages** — `record_message_sent()`
3. **Streams** — `record_stream_started()`, `record_first_token()`, `record_stream_completed()`, `record_stream_error()`
4. **Threads** — `record_thread_created()`, `record_thread_deleted()`, `record_thread_messages()`

See `deep_agent/aegra/otel.py` for the full API.

### Example

```python
from deep_agent.aegra.otel import record_conversation_started, record_conversation_completed

async def handle_conversation(thread_id: str):
    start_mono = record_conversation_started(attributes={"thread_id": thread_id})
    try:
        # ... conversation logic ...
        record_conversation_completed(start_mono, status="completed")
    except Exception:
        record_conversation_completed(start_mono, status="error")
        raise
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_OTEL` | `false` | Enable agent lifecycle OTEL export (`aegra/otel.py`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP gRPC endpoint for lifecycle metrics/traces |
| `OTEL_EXPORTER_OTLP_INSECURE` | `true` | Disable TLS for OTLP connection |
| `OTEL_METRIC_EXPORT_INTERVAL` | `5000` | Metric export interval in ms (1000–60000) |
| `ENABLE_OTEL_METRICS` | `false` | Enable token budget metrics export |
| `ENABLE_OTEL_TRACES` | `false` | Enable trace export via `otel_setup.py` |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | `""` | Separate traces endpoint (defaults to Jaeger in dev) |
| `OTEL_SERVICE_NAME` | `template-agent` | Service name for token budget OTEL resource |
| `OTEL_AUTH_TOKEN` | `""` | Bearer token for authenticated OTLP endpoints |
| `APPLICATION_VERSION` | — | Overrides `service.version` resource attribute |

### YAML Configuration

File: `config/agent/runtime/observability.yaml`

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

Environment variables override YAML values.

## Troubleshooting

### Metrics report zero

Expected until `record_*` helpers are wired into runtime handlers. Graph build metrics should increment on agent graph compilation.

### Traces not in Jaeger

1. Confirm `ENABLE_OTEL=true` or `ENABLE_OTEL_TRACES=true` with a valid traces endpoint
2. Check agent logs for `OTEL enabled` or `template_agent_otel_tracing_enabled`
3. Verify Jaeger OTLP receiver: `COLLECTOR_OTLP_ENABLED=true` (set in compose)
4. Check `/health` → `checks.otel` for initialization status

### OTLP connection failures

```bash
# From agent container/network namespace
curl -v telnet://jaeger:4317
```

Verify endpoint hostname matches compose service name (`jaeger`, not `localhost`, from inside the agent container).
