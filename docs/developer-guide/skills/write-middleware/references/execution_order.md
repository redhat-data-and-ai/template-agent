# Middleware Pipeline Execution Order

## `build_middleware_list()` Order

The `build_middleware_list()` function in `deep_agent/src/infrastructure/middleware.py` builds middleware in this order. Middleware earlier in the list runs first (outermost wrapper for wrap hooks).

1. **AuditMiddleware** -- always included if platform audit is enabled (`deep_agent.src.audit.middleware`)
2. **OPAMiddleware** -- always included if OPA authorization is enabled (`deep_agent.src.opa.middleware`)
3. **GeminiSafetyLogMiddleware** -- always included; detects and replaces Gemini safety-blocked empty responses
4. **Custom PII middleware** -- from the global scrubber, when `pii.enabled` is true (`deep_agent.src.pii.middleware`)
5. **SummarizationToolMiddleware** -- when `middleware.summarization_tool.enabled` is true; gives the agent a tool to proactively trigger conversation summarization
6. **Guard middleware** (in order):
   - **ModelCallLimitMiddleware** -- caps LLM calls per run (`middleware.model_call_limit`)
   - **ToolCallLimitMiddleware** -- caps tool calls per run (`middleware.tool_call_limit`)
   - **ModelRetryMiddleware** -- retries transient LLM failures with backoff (`middleware.model_retry`)
   - **ModelFallbackMiddleware** -- switches to fallback model on primary failure (`middleware.model_fallback`)
   - **ToolRetryMiddleware** -- retries specific tool failures (`middleware.tool_retry`)
7. **Stock PII rules** -- default-provider rules wrapped in `ParallelPIIMiddleware` for concurrent execution
8. **Extra middleware** -- from `middleware.extra` list in `agent.yaml` -- **THIS IS WHERE CUSTOM MIDDLEWARE GOES**

## Auto-Included by Deepagents Framework

These middleware are added by `create_deep_agent()` itself, not by `build_middleware_list()`. They do not need to be registered in `middleware.extra`:

- **SubAgentMiddleware** -- manages subagent lifecycle and delegation
- **SummarizationMiddleware** -- automatic conversation summarization at token thresholds
- **PatchToolCallsMiddleware** -- fixes malformed tool calls from the LLM (can be excluded via `excluded_middleware`)
- **FilesystemMiddleware** -- provides filesystem access to the agent
- **TodoListMiddleware** -- manages the agent's internal task list
- **MemoryMiddleware** -- cross-thread persistent memory (enabled when `middleware.memory.enabled` is true)

## HTTP-Level Middleware (Starlette)

These are separate from the agent middleware pipeline. They run at the HTTP request level:

- **SecurityHeadersMiddleware** -- OWASP security headers (X-Content-Type-Options, X-Frame-Options, HSTS, CSP)
- **RequestSizeLimitMiddleware** -- body size limit (default 10MB, configurable via `REQUEST_BODY_MAX_SIZE` env var)
- **RequestContextMiddleware** -- correlation headers (X-Trace-ID, X-Request-ID, X-Org-ID)

## Safety Boundary

Not AgentMiddleware but related to the middleware pipeline:

- **SafetyAwareRunnable** -- wraps compiled subagent graphs, catches `ContentSafetyError` exceptions raised by guard middleware and returns a safe user-facing message

## Vertex AI Safety Settings

Environment variables that interact with `GeminiSafetyLogMiddleware`:

| Env Var | Default | Purpose |
|---------|---------|---------|
| `SAFETY_DANGEROUS_CONTENT` | `BLOCK_MEDIUM_AND_ABOVE` | Dangerous content filter threshold |
| `SAFETY_HATE_SPEECH` | `BLOCK_MEDIUM_AND_ABOVE` | Hate speech filter threshold |
| `SAFETY_HARASSMENT` | `BLOCK_MEDIUM_AND_ABOVE` | Harassment filter threshold |
| `SAFETY_SEXUALLY_EXPLICIT` | `BLOCK_LOW_AND_ABOVE` | Sexually explicit content filter threshold |
