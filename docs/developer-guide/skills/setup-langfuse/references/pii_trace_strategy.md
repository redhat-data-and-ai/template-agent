# PII Trace Strategy for Langfuse

PII scrubbing for Langfuse traces is configured in `config/agent/runtime/pii.yaml` via the `trace_strategy` field.

## Strategies

| Strategy | Output | Use Case |
|----------|--------|----------|
| `hash` | `[HASH:abc123]` | Deterministic — same PII always produces the same hash, enabling cross-request correlation for debugging |
| `redact` | `***REDACTED***` | One-way removal — no correlation possible, maximum privacy |

## How It Works

The `mask_otel_spans` callback is registered on the Langfuse client during setup (`telemetry.py`). It runs before span data is exported, so PII never reaches Langfuse.

1. On startup, `setup_pii_middleware()` initializes the PII scrubber from `pii.yaml` rules.
2. `setup_langfuse_tracing()` checks if a scrubber is active and registers a `mask_otel_spans` callback.
3. Before each span batch is sent, the callback iterates over span attributes and applies the configured `trace_strategy` (hash or redact).

## Configuration

In `config/agent/runtime/pii.yaml`:

```yaml
enabled: true
trace_strategy: hash   # or "redact"
rules:
  - name: email
    strategy: scrub
    provider: regex
  # ... additional rules
```

## Source References

- `deep_agent/src/pii/config.py` — `PIIConfig.trace_strategy` field definition (type: `Literal["redact", "hash"]`, default: `hash`)
- `deep_agent/aegra/telemetry.py` — `mask_otel_spans` callback registration and scrub logic
- `config/agent/runtime/pii.yaml` — runtime configuration file
