# Observability

Three signals, three destinations, and the keys that join them. This is
the page to read at 03:00 when an alert has fired and you need to get
from "something is wrong" to the turn that broke.

## Where each signal goes

| Signal | Emitted by | Leaves the pod via | Lands in |
|---|---|---|---|
| Logs | `deep_agent/utils/pylogger.py` (structlog, JSON to stdout) | container stdout, collected at the node | your log backend |
| Metrics | `deep_agent/aegra/otel.py` (~15 instruments) | OTLP, **only when `ENABLE_OTEL=true`** | your metrics backend |
| In-flight gauge | `deep_agent/aegra/inflight_metrics.py` | Prometheus `/metrics`, scraped by the `ServiceMonitor` | cluster monitoring, and KEDA |
| LLM traces | Langfuse `CallbackHandler` (`deep_agent/aegra/telemetry.py`) | Langfuse SDK, direct HTTP | Langfuse |

The template does not name a vendor. The seam is OTLP: point
`OTEL_EXPORTER_OTLP_ENDPOINT` at whatever receives it in your cluster — a
collector, or a vendor agent's OTLP ingest — and set `ENABLE_OTEL=true`.
Both live in `config/agent/deployment/values.yaml` under `config:`.

**Until you do that, the metrics do not exist anywhere.** The agent is
instrumented regardless, but with OTLP off everything in `otel.py` stays
in an in-memory reader, readable only through `/api/metrics` on one pod.
The only series that reaches the cluster is `aegra_runs_in_flight`, which
exists for autoscaling. Every SLI in [slo.md](slo.md) depends on the
export being on.

Langfuse is separate on purpose: it holds prompts, completions, token
counts and cost, which are not metrics and do not belong in a time series.
It activates on its own when `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`
/ `LANGFUSE_BASE_URL` are present.

## The join keys

Every log line carries these. Two of them also reach the other backends,
which is what makes a pivot possible.

| Field | In logs | In traces | In Langfuse | What it identifies |
|---|---|---|---|---|
| `otel_trace_id` | yes | **native trace id** | metadata | one request, W3C |
| `trace_id` | yes | span attr `app.trace_id` | metadata | one request, application-level |
| `request_id` | yes | — | — | one HTTP call, from the gateway |
| `thread_id` | yes | — | `session_id` | one conversation |
| `user_id` | yes | — | `user_id` (encrypted) | who |
| `service` | yes | `service.name` | — | which agent |

`trace_id` is the application's own identifier: the UI sends it as
`X-Trace-ID`, the agent echoes it back on the response and forwards it to
MCP servers. `otel_trace_id` is the W3C one, read from the active span.
They are usually the same value — when no `X-Trace-ID` arrives, the agent
adopts the W3C id rather than minting a fresh UUID, precisely so that the
two do not diverge — but both are logged because when they *do* differ,
you need each one to reach its own backend.

## Pivoting

**Metric alert → the requests behind it.** The alert gives you a service, a
time window and usually a `status_code` or `model`. Query logs for the same
window and labels, and take an `otel_trace_id` off any matching line.

**Log line → the trace.** Search the trace backend for `otel_trace_id`. It
is the native trace id there, not an attribute, so this is an ID lookup.

**Log line → what the model was asked.** Search Langfuse metadata for
`trace_id`, or filter by `session_id` = `thread_id` to get the whole
conversation rather than one turn.

**A user complaint → everything.** Start from `thread_id`, which the UI
shows. It reaches logs directly and Langfuse as `session_id`.

**A trace → the logs.** Search logs for the trace id as `otel_trace_id`.

## What is deliberately not here

- **No alert definitions.** They belong with your monitors, not in the
  template. [slo.md](slo.md) specifies the SLIs precisely enough to write
  them from.
- **No dashboards.** Same reason.
- **No log shipper.** Collection is a cluster concern — the agent writes
  JSON to stdout and stops there. If your cluster has no node-level
  collector, logs end when the pod does, and with `scaleToZero` or an HPA
  scale-down that is sooner than you think.

## Gotchas

- `ENABLE_PROMETHEUS_METRICS` and `ENABLE_OTEL` are different switches for
  different pipelines. The first publishes one gauge for autoscaling and
  defaults on; the second exports everything else and defaults off.
  Turning the second on does not make the first redundant: KEDA and the
  HPA read the scraped gauge.
- `OTEL_METRIC_EXPORT_INTERVAL` sets the resolution of every metric alert
  you can write. The code default is 5s, which is fine locally and is 12x
  the cardinality in production for resolution nothing acts on. The
  OpenShift overlay sets 60000.
- PII is scrubbed before it reaches Langfuse and the audit log — see
  `config/agent/runtime/pii.yaml`. Log lines are scrubbed by the same
  machinery, but anything you add with a raw f-string bypasses it.
- `service.name` comes from `SERVICE_NAME`, and the pod labels come from
  `app.serviceName` in `values.yaml`. Set both when you scaffold. Left at
  the default, every agent built from this template is called `agent`.
