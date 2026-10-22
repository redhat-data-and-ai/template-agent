# Langfuse Troubleshooting

Common issues and solutions when working with Langfuse tracing.

## No traces appearing

- Verify both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. Both are required for auto-activation.
- Check agent startup logs for `Langfuse credentials not set — auto-tracing disabled`. This means the keys are missing or empty.
- Verify `LANGFUSE_BASE_URL` points to the correct Langfuse instance.
- Confirm the keys belong to the correct Langfuse project.

## Noisy Langfuse logs

- Langfuse SDK loggers are suppressed to ERROR level by default.
- To temporarily enable verbose output for debugging, set `PYTHON_LOG_LEVEL=DEBUG`.
- Remember to revert to `INFO` after troubleshooting.

## Traces not grouped by conversation

- Ensure `thread_id` is consistent across all messages in the same conversation.
- The `LangfuseObservabilityProvider` injects `thread_id` as both `langfuse_session_id` and `thread_id` in metadata. Verify your client is sending a stable thread ID.

## PII appearing in traces

- Enable PII scrubbing in `config/agent/runtime/pii.yaml` by setting `enabled: true`.
- Choose `trace_strategy: hash` (correlatable) or `trace_strategy: redact` (maximum privacy).
- Verify PII middleware initialized successfully by checking for `PII middleware initialised` in startup logs.

## Slow shutdown / traces lost on shutdown

- Increase `SHUTDOWN_LANGFUSE_TIMEOUT_SECONDS` (default: 5s) to allow more time for pending traces to flush.
- Check logs for `Langfuse shutdown timed out` — this indicates the timeout is too short for the volume of pending traces.

## HITL traces appearing as separate traces

- The agent uses a shared `HitlAwareCallbackHandler` that stitches interrupt and resume into a single trace.
- If traces split on HITL resume, verify the handler is initializing correctly: look for `Langfuse auto-tracing registered (shared HITL-aware handler for interrupt/resume)` in startup logs.
- Ensure `thread_id` is preserved across the interrupt/resume cycle.
