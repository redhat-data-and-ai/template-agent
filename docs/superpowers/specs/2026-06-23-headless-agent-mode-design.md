# Headless Agent Mode + Event-Driven Triggers

## Summary

Add a headless agent mode (`mode: headless` in `agent.yaml`) that disables the Aegra HTTP server and runs the agent as a background worker. An `EventTriggerMiddleware` consumes events from three trigger sources (webhook, cron, queue consumer) and invokes the agent graph per event. Results are fanned out to configurable output sinks. All components are tested with unit tests, integration tests, and a full headless startup test.

## Configuration

New top-level keys in `config/agent/runtime/agent.yaml`:

```yaml
mode: server  # server (default, full Aegra) | headless

triggers:
  webhook:
    enabled: false
    host: "0.0.0.0"
    port: 8888
    path: "/trigger"
  cron:
    enabled: false
    jobs:
      - name: "daily-report"
        schedule: "0 9 * * *"
        payload:
          task: "generate_daily_report"
  queue:
    enabled: false
    backend: "redis_streams"
    stream: "agent-tasks"
    consumer_group: "agent-workers"
    consumer_name: "worker-1"

output_sinks:
  - type: stdout
  - type: file
    path: "/var/log/agent/output.jsonl"
  - type: webhook
    url: "https://downstream.example.com/results"
    headers:
      Authorization: "Bearer ${WEBHOOK_TOKEN}"
  - type: redis
    stream: "agent-results"
```

### Behavior

- `mode: server` — default. Full Aegra API starts. Current behavior, nothing changes.
- `mode: headless` — Aegra HTTP server does not start. `EventTriggerMiddleware` takes over as the runtime.
- `triggers:` — only meaningful when `mode: headless`. Each trigger type is independently enabled.
- `output_sinks:` — list of sinks. All enabled sinks receive every output (fan-out). If empty, defaults to stdout.

### Pydantic Models

- `WebhookTriggerConfig` — host, port, path
- `CronJobConfig` — name, schedule, payload
- `CronTriggerConfig` — enabled, jobs list
- `QueueTriggerConfig` — enabled, backend, stream, consumer_group, consumer_name
- `TriggerConfig` — webhook, cron, queue
- `OutputSinkConfig` — type, plus type-specific fields (path, url, headers, stream)
- `HeadlessConfig` — mode, triggers, output_sinks (top-level container)

## Component Architecture

```
deep_agent/src/triggers/
├── __init__.py
├── config.py          # Pydantic models for triggers + sinks
├── middleware.py       # EventTriggerMiddleware (orchestrates lifecycle)
├── sources/
│   ├── __init__.py
│   ├── protocol.py    # TriggerSource protocol (async iterator → TriggerEvent)
│   ├── webhook.py     # Minimal HTTP listener trigger
│   ├── cron.py        # APScheduler-based trigger
│   └── queue.py       # QueueConsumer protocol + Redis Streams implementation
├── sinks/
│   ├── __init__.py
│   ├── protocol.py    # OutputSink protocol
│   ├── stdout.py
│   ├── file.py
│   ├── webhook.py     # POST results to URL
│   └── redis.py       # Publish to Redis Stream
└── runtime.py         # HeadlessRuntime (adapts graph factory)
```

### TriggerSource Protocol

```python
class TriggerSource(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def __aiter__(self) -> AsyncIterator[TriggerEvent]: ...
```

Each source yields `TriggerEvent` objects. Sources run as independent async tasks.

### TriggerEvent

```python
@dataclass
class TriggerEvent:
    name: str
    payload: dict
    source: str          # "webhook" | "cron" | "queue"
    metadata: dict
    timestamp: datetime
```

### TriggerResult

```python
@dataclass
class TriggerResult:
    event: TriggerEvent
    output: Any
    duration_ms: float
    success: bool
    error: str | None
```

### EventTriggerMiddleware

Owns the full lifecycle:

- **start()** — reads config, instantiates enabled trigger sources and output sinks, starts all sources
- **run()** — main loop consuming TriggerEvents from all sources via `asyncio.TaskGroup`, invokes agent graph per event, fans out TriggerResults to all sinks
- **stop()** — stops accepting new events, waits for in-flight invocations (configurable drain timeout), flushes all sinks

### QueueConsumer Protocol

```python
class QueueConsumer(Protocol):
    async def consume(self) -> AsyncIterator[QueueMessage]: ...
    async def ack(self, message: QueueMessage) -> None: ...
    async def close(self) -> None: ...
```

Ships with `RedisStreamsConsumer` as default implementation. Users implement this protocol for Kafka/RabbitMQ/SQS.

### OutputSink Protocol

```python
class OutputSink(Protocol):
    async def emit(self, result: TriggerResult) -> None: ...
    async def close(self) -> None: ...
```

Four implementations: `StdoutSink`, `FileSink`, `WebhookSink`, `RedisSink`.

### HeadlessRuntime

Implements the same interface shape as Aegra's `ServerRuntime` without SSO. Constructed with a configurable identity (service account or anonymous). Passed to the existing `graph.py:agent()` factory unchanged.

Switching to direct `ServerRuntime` construction later (option 1) is ~30 lines of change — delete `HeadlessRuntime` class and construct `ServerRuntime` with synthetic values.

## Data Flow

### Startup

1. `python -m deep_agent.headless` — new entry point
2. Loads `agent.yaml`, checks `mode: headless`
3. Runs same `check_prerequisites()` as Aegra (DB, model provider, Redis)
4. Creates `HeadlessRuntime` (no SSO, service account identity)
5. Calls `graph.py:agent(headless_runtime)` once to get compiled graph
6. `EventTriggerMiddleware.start()` — instantiates and starts all enabled trigger sources + sinks

### Main Loop

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Webhook     │     │  Cron        │     │  Queue       │
│  Listener    │     │  Scheduler   │     │  Consumer    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────┬───────┴───────────────────┘
                   ▼
           TriggerEvent stream
                   │
                   ▼
        EventTriggerMiddleware
         (asyncio.TaskGroup)
                   │
                   ▼
          graph.ainvoke(event.payload)
                   │
                   ▼
             TriggerResult
                   │
           ┌───────┼───────┬───────┐
           ▼       ▼       ▼       ▼
         stdout   file   webhook  redis
```

- Each trigger source runs as an independent async task
- Events are consumed as they arrive — no batching
- Graph invocation is per-event (one at a time by default)
- All enabled sinks receive every result (fan-out)
- Errors in graph invocation are caught, logged, and emitted as failed TriggerResults — the loop continues

### Shutdown

1. Signal received (SIGTERM/SIGINT)
2. `EventTriggerMiddleware.stop()` — stops accepting new events
3. Waits for in-flight graph invocations to complete (configurable drain timeout)
4. Flushes all output sinks
5. Runs same cleanup as Aegra (Langfuse flush, Redis close, cache clear)

### Error Handling

- **Trigger source failure** (e.g., Redis disconnects) — logged, reconnect with backoff, other sources unaffected
- **Graph invocation failure** — `TriggerResult(success=False, error=...)` sent to all sinks
- **Sink failure** — logged, other sinks unaffected (one sink failing doesn't block others)

## Testing Strategy

### Unit Tests (`tests/unit/triggers/`)

| Test file | Covers |
|---|---|
| `test_config.py` | Pydantic model validation — trigger configs, sink configs, defaults, invalid values |
| `test_middleware.py` | EventTriggerMiddleware lifecycle — start/stop, event consumption, fan-out to sinks, error handling |
| `test_webhook_source.py` | Webhook listener — starts/stops HTTP, yields TriggerEvent from POST body |
| `test_cron_source.py` | Cron source — schedules jobs, fires TriggerEvent on schedule, cancellation |
| `test_queue_source.py` | RedisStreamsConsumer + QueueConsumer protocol compliance |
| `test_stdout_sink.py` | JSON to stdout |
| `test_file_sink.py` | JSONL append, directory creation, flush |
| `test_webhook_sink.py` | POST result, HTTP error handling |
| `test_redis_sink.py` | Publish to stream |
| `test_runtime.py` | HeadlessRuntime construction, graph factory acceptance |

All unit tests are mock-based — no real services needed.

### Integration Tests (`tests/integration/triggers/`)

| Test file | Covers |
|---|---|
| `test_redis_streams.py` | Real Redis Streams — produce message, consumer picks it up, ack works, consumer group behavior, reconnect after disconnect |
| `test_webhook_listener.py` | Real HTTP POST to webhook listener, verify TriggerEvent arrives in middleware |
| `test_redis_sink_integration.py` | Real Redis — emit TriggerResult, read it back from stream |
| `test_end_to_end.py` | Full pipeline: push event to Redis Stream → EventTriggerMiddleware consumes → graph invoked (mocked) → result appears in output sink (real Redis or real file) |

### Headless Startup Test (`tests/integration/triggers/test_headless_startup.py`)

- Starts full headless process (`python -m deep_agent.headless`) as a subprocess
- Verifies prerequisites check runs (DB, model provider)
- Verifies trigger sources start (cron scheduled, queue consumer connected, webhook listener bound)
- Sends a test event via webhook POST and via Redis Stream push
- Asserts output appears in configured sink
- Sends SIGTERM, verifies graceful shutdown (in-flight drained, sinks flushed, process exits 0)

### Infrastructure

- Redis: same docker-compose Redis service already in stack (`make dev` brings it up)
- Pytest marker: `@pytest.mark.integration` — skipped by `make test`, included by `make test-all`
- Fixture: `redis_client` that flushes test keys before/after each test
- Fixture: `headless_process` that starts/stops the headless worker with a test config

### New Makefile Targets

```makefile
test-triggers:       ## Unit tests for triggers only
    pytest tests/unit/triggers/ -v

test-integration:    ## Integration tests (requires Redis + DB)
    pytest tests/integration/ -m integration -v

test-headless:       ## Full headless startup test
    pytest tests/integration/triggers/test_headless_startup.py -v
```

## Files Changed

### New Files

- `deep_agent/src/triggers/__init__.py`
- `deep_agent/src/triggers/config.py`
- `deep_agent/src/triggers/middleware.py`
- `deep_agent/src/triggers/runtime.py`
- `deep_agent/src/triggers/sources/__init__.py`
- `deep_agent/src/triggers/sources/protocol.py`
- `deep_agent/src/triggers/sources/webhook.py`
- `deep_agent/src/triggers/sources/cron.py`
- `deep_agent/src/triggers/sources/queue.py`
- `deep_agent/src/triggers/sinks/__init__.py`
- `deep_agent/src/triggers/sinks/protocol.py`
- `deep_agent/src/triggers/sinks/stdout.py`
- `deep_agent/src/triggers/sinks/file.py`
- `deep_agent/src/triggers/sinks/webhook.py`
- `deep_agent/src/triggers/sinks/redis.py`
- `deep_agent/headless.py` — headless worker entry point (`python -m deep_agent.headless`)
- `tests/unit/triggers/` — 10 test files
- `tests/integration/triggers/` — 5 test files

### Modified Files

- `config/agent/runtime/agent.yaml` — add `mode`, `triggers`, `output_sinks` sections
- `Makefile` — add `test-triggers`, `test-integration`, `test-headless` targets
- `pyproject.toml` — add `integration` pytest marker
