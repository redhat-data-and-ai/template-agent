# Built-in Middleware Reference

## Agent Middleware (deepagents pipeline)

| Name | Module | Purpose | Config Key | Hooks Used |
|------|--------|---------|------------|------------|
| AuditMiddleware | `deep_agent.src.audit.middleware` | Emits platform audit events for LLM calls, MCP tool calls, memory writes, and subagent delegations | Always on when audit enabled (env-based) | `wrap_model_call`, `awrap_model_call`, `wrap_tool_call`, `awrap_tool_call` |
| OPAMiddleware | `deep_agent.src.opa.middleware` | Enforces OPA authorization policy on model outputs and tool results; blocks policy-violating content with retries | Always on when OPA enabled (env-based) | `abefore_model`, `awrap_model_call`, `awrap_tool_call` |
| GeminiSafetyLogMiddleware | `deep_agent.src.infrastructure.middleware` | Detects Gemini safety-blocked empty responses (candidate-level and prompt-level blocks) and replaces them with a user-facing refusal message | Always included | `after_model`, `aafter_model` |
| PIIMiddleware | `deep_agent.src.pii.middleware` | Scrubs PII from model inputs using token-map anonymization, restores PII in model outputs; blocks input containing PII with `strategy: block` rules | `pii.enabled` in agent config | `abefore_agent`, `before_agent`, `awrap_model_call`, `wrap_model_call` |
| ParallelPIIMiddleware | `deep_agent.src.infrastructure.middleware` | Runs stock LangChain PII type checks concurrently instead of sequentially | `middleware.pii.rules` (default provider) | `abefore_model`, `before_model` |
| ModelCallLimitMiddleware | `langchain.agents.middleware` | Caps total LLM calls per agent run | `middleware.model_call_limit` | (framework-managed) |
| ToolCallLimitMiddleware | `langchain.agents.middleware` | Caps total tool calls per agent run | `middleware.tool_call_limit` | (framework-managed) |
| ModelRetryMiddleware | `langchain.agents.middleware` | Retries transient LLM failures with exponential backoff; skips retry for `ContentSafetyError` | `middleware.model_retry` | (framework-managed) |
| ModelFallbackMiddleware | `langchain.agents.middleware` | Switches to a fallback model when the primary model fails | `middleware.model_fallback` | (framework-managed) |
| ToolRetryMiddleware | `langchain.agents.middleware` | Retries failures for specific named tools | `middleware.tool_retry` | (framework-managed) |
| SummarizationToolMiddleware | `deepagents.middleware.summarization` | Gives the agent a tool to proactively trigger conversation summarization | `middleware.summarization_tool` | (framework-managed) |

## Auto-Included Middleware (deepagents framework)

These are added by `create_deep_agent()` and cannot be registered via `middleware.extra`:

| Name | Purpose | Can Be Excluded |
|------|---------|-----------------|
| SubAgentMiddleware | Manages subagent lifecycle and delegation | No |
| SummarizationMiddleware | Automatic conversation summarization at token thresholds | No |
| PatchToolCallsMiddleware | Fixes malformed tool calls from the LLM | Yes (`excluded_middleware: [PatchToolCallsMiddleware]`) |
| FilesystemMiddleware | Provides filesystem access to the agent | No |
| TodoListMiddleware | Manages the agent's internal task list | No |
| MemoryMiddleware | Cross-thread persistent memory | Controlled via `middleware.memory.enabled` |

## HTTP-Level Middleware (Starlette)

Not agent middleware; runs at the HTTP request/response level:

| Name | Purpose | Config |
|------|---------|--------|
| SecurityHeadersMiddleware | OWASP security headers (X-Content-Type-Options, X-Frame-Options, HSTS, CSP) | Always active |
| RequestSizeLimitMiddleware | Body size limit | `REQUEST_BODY_MAX_SIZE` env var (default 10MB) |
| RequestContextMiddleware | Correlation headers (X-Trace-ID, X-Request-ID, X-Org-ID) | Always active |
