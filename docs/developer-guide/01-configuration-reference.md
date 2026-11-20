# Configuration Reference

This document is the authoritative reference for every configuration field in the template-agent framework. All operational configuration is done through YAML and JSON files in the `config/agent/` directory -- no Python code changes are needed to customise agent behaviour. Secrets and infrastructure endpoints are injected via environment variables.

Each field table uses the format: **Field | Type | Default | Required | Description**. Sections are annotated `[YAML-loaded]` (parsed by the Python config loader at startup) or `[env-var]` (read from environment variables via Pydantic `BaseSettings`).

---

## File Overview

| File | Format | Purpose |
|------|--------|---------|
| `config/agent/runtime/agent.yaml` | YAML | Unified runtime config: identity, providers, middleware, filesystem, cache, guardrails, OPA, and more |
| `config/agent/mcp.json` | JSONC | MCP (Model Context Protocol) server registry |
| `config/agent/runtime/pii.yaml` | YAML | PII detection and scrubbing rules |
| `config/agent/runtime/observability.yaml` | YAML | OpenTelemetry exporter, metrics, and tracing config |
| `config/agent/runtime/ui.yaml` | YAML | Frontend BFF server, security, features, and observability |
| `config/agent/runtime/secrets.example.yaml` | YAML | Documentation-only reference for required secrets (never stores actual values) |
| `config/agent/PROMPT.md` | Markdown + YAML frontmatter | Orchestrator prompt, model selection, tool/skill/MCP binding, access control |
| `.env.example` | dotenv | Environment variable reference for secrets and infrastructure endpoints |

---

## `config/agent/runtime/agent.yaml`

The unified runtime configuration file. Parsed once at startup by `AgentConfig`. Individual sections are extracted and validated against Pydantic models.

### Identity [YAML-loaded]

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `name` | `string` | `"Agent"` | No | Display name for the agent instance. Used in logs, UI, and as the DCR/token key fallback when `DEPLOYED_AGENT_NAME` is not set. |

### Model [env-var]

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `model.max_output_tokens` | `int` | `8192` | No | Maximum tokens the LLM may generate per response. Mapped to `MAX_OUTPUT_TOKENS` env var. |

### Providers [YAML-loaded]

Controls how the framework resolves LLM model instances. See [Middleware Extension Guide](./02-middleware-extension-guide.md) for how providers interact with middleware.

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `resolve_strategy` | `"legacy" \| "deepagents"` | `"legacy"` | No | Model resolution strategy. `legacy` uses the built-in `create_model()` function (Vertex AI for Gemini/Claude, vLLM for others). `deepagents` uses the `resolve_model()` function with the ProviderProfile registry. |
| `providers` | `map[string, object]` | `{}` | No | Named provider profiles. Each key is a provider name (e.g., `google_genai`, `anthropic_vertex`, `openai`). Only used when `resolve_strategy: deepagents`. |
| `providers.<name>.init_kwargs` | `map[string, any]` | `{}` | No | Keyword arguments passed to `init_chat_model()` for this provider. Common keys: `project`, `temperature`, `location`. Supports `${ENV_VAR}` expansion. |

### Harness Profiles [YAML-loaded]

Per-model runtime adjustments applied at graph build time. The key format is `"provider:model"` or just `"model"` for provider-agnostic matching (e.g., `gemini-2.5-pro` or `claude-sonnet-4`).

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `harness_profiles.<model>` | `object` | -- | No | Profile block for a specific model. Matched against the `model` field from agent/subagent frontmatter. |
| `harness_profiles.<model>.system_prompt_suffix` | `string` | `""` | No | Text appended to the system prompt when this model is active. Use for model-specific instructions. |
| `harness_profiles.<model>.excluded_tools` | `list[string]` | `[]` | No | Tool names to exclude from the agent's toolset when this model is active. |
| `harness_profiles.<model>.excluded_middleware` | `list[string]` | `[]` | No | Middleware names to disable for this model (e.g., `patch_tool_calls` for Claude models that handle tool calls natively). |
| `harness_profiles.<model>.general_purpose_subagent` | `object` | `{enabled: true}` | No | Configuration for the auto-added general-purpose subagent. |
| `harness_profiles.<model>.general_purpose_subagent.enabled` | `bool` | `true` | No | Whether to attach a general-purpose subagent to this model's agent graph. |
| `harness_profiles.<model>.general_purpose_subagent.description` | `string` | `null` | No | Custom description for the general-purpose subagent. |
| `harness_profiles.<model>.general_purpose_subagent.system_prompt` | `string` | `null` | No | Custom system prompt for the general-purpose subagent. |

### Middleware Pipeline [YAML-loaded]

Controls which deepagents middleware components are active and their parameters. Resolution order: global defaults (this section) -> profile (matched from the agent's `model` field) -> per-agent overrides in frontmatter.

#### `middleware.human_approval`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `human_approval.enabled` | `bool` | `true` | No | Master switch for human-in-the-loop tool approval. When enabled, the agent pauses before executing tool calls and waits for user approval. |
| `human_approval.mode` | `"all" \| "none"` | `"all"` | No | `all` requires approval for every tool call (except those in `exclude`). `none` disables approval entirely. |
| `human_approval.exclude` | `list[string]` | `[]` | No | Tool names exempted from approval prompts (e.g., `write_todos`, `compact_conversation`). |

#### `middleware.summarization_tool`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `summarization_tool.enabled` | `bool` | `true` | No | Activates `SummarizationToolMiddleware`, which provides context summarisation capabilities to the agent. |

#### `middleware.memory`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `memory.enabled` | `bool` | `true` | No | Activates `MemoryMiddleware` for cross-conversation memory injection and persistence. |
| `memory.namespaces` | `list[string]` | `["memories"]` | No | Store namespaces the memory middleware reads from and writes to. |

#### `middleware.patch_tool_calls`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `patch_tool_calls.enabled` | `bool` | `true` | No | Activates `PatchToolCallsMiddleware`. Automatically excluded for Claude models via harness profiles since Claude handles tool calls natively. |

#### `middleware.skills`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `skills.enabled` | `bool` | `true` | No | Activates `SkillsMiddleware`, which injects skill reference documents into the agent's context when skills are configured in frontmatter. |

#### `middleware.model_call_limit`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `model_call_limit.enabled` | `bool` | `true` | No | Activates `ModelCallLimitMiddleware` to cap LLM invocations per run, preventing runaway loops. |
| `model_call_limit.run_limit` | `int` | `50` | No | Maximum number of LLM calls allowed in a single agent run. |

#### `middleware.tool_call_limit`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `tool_call_limit.enabled` | `bool` | `true` | No | Activates `ToolCallLimitMiddleware` to cap tool invocations per run. |
| `tool_call_limit.run_limit` | `int` | `200` | No | Maximum number of tool calls allowed in a single agent run. |

#### `middleware.model_retry`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `model_retry.enabled` | `bool` | `true` | No | Activates `ModelRetryMiddleware` to retry LLM calls on transient failures (rate limits, timeouts, 5xx). |
| `model_retry.max_retries` | `int` | `3` | No | Maximum number of retry attempts per failed LLM call. |
| `model_retry.backoff_factor` | `float` | `2.0` | No | Exponential backoff multiplier between retries. |
| `model_retry.initial_delay` | `float` | `1.0` | No | Initial delay in seconds before the first retry. |

#### `middleware.model_fallback`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `model_fallback.enabled` | `bool` | `false` | No | Activates `ModelFallbackMiddleware` to switch to a backup model when the primary model fails. |
| `model_fallback.fallback_model` | `string` | `""` | Yes (when enabled) | Fallback model identifier in `provider:model` format (e.g., `google_genai:gemini-2.5-flash`). The fallback model must use the same auth configuration as the primary. |

#### `middleware.tool_retry`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `tool_retry.enabled` | `bool` | `false` | No | Activates `ToolRetryMiddleware` to retry specific tools on failure. |
| `tool_retry.max_retries` | `int` | `2` | No | Maximum retry attempts per failed tool call. |
| `tool_retry.tools` | `list[string]` | `[]` | No | Explicit list of tool names eligible for retry. Only these tools are retried; all others fail immediately. |

#### `middleware.extra`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `extra` | `list[string]` | `[]` | No | Additional middleware class names to include in the pipeline. Use for custom middleware extensions. |

### Async Tasks [YAML-loaded]

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `async_tasks.enabled` | `bool` | `true` | No | Activates `AsyncSubAgentMiddleware`, enabling the orchestrator to dispatch long-running tasks to background subagents. |
| `async_tasks.system_prompt` | `string` | `null` | No | Custom system prompt for async subagent tasks. When `null`, inherits the orchestrator's system prompt. |

### Filesystem & Storage [YAML-loaded]

Validated by `FilesystemFileConfig` in `deep_agent/src/agent/config/filesystem.py`.

#### `filesystem.backend`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `backend.type` | `"state" \| "composite" \| "store" \| "local_shell"` | `"state"` | No | Backend type. `state`: thread-scoped ephemeral scratch (recommended for production). `composite`: routes paths to different backends. `store`: cross-thread persistent via LangGraph Store. `local_shell`: real filesystem with isolated venv (local dev only). |
| `backend.local_shell.timeout` | `int` | `120` | No | Shell command execution timeout in seconds. |
| `backend.local_shell.max_output_bytes` | `int` | `100000` | No | Maximum bytes captured from shell command output before truncation. |
| `backend.state.enabled` | `bool` | `false` | No | Whether the state backend sub-component is explicitly enabled. |
| `backend.store.enabled` | `bool` | `false` | No | Whether the store backend sub-component is explicitly enabled. |
| `backend.store.scope` | `"user" \| "assistant" \| "org"` | `"user"` | No | Scope for store-backed persistent files. `user`: per-user isolation. `assistant`: shared across all users. `org`: shared within an organisation. |
| `backend.routes` | `map[string, string]` | `{}` | No | Path-prefix-to-backend routing table for the `composite` backend type. Keys are path prefixes (e.g., `"/skills/"`); values are backend names (`filesystem_readonly`, `store`, `state`). The first matching prefix wins; `"/"` is the catch-all. |

#### `filesystem.permissions`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `permissions` | `list[object]` | `[]` | No | Ordered list of permission rules controlling which filesystem operations the agent may perform. First matching rule wins. |
| `permissions[].operations` | `list[string]` | -- | Yes | Operations this rule applies to: `read`, `write`, `edit`, `glob`, `grep`, `ls`. |
| `permissions[].paths` | `list[string]` | -- | Yes | Glob patterns for paths this rule covers (e.g., `"config/**"`, `"*.py"`). |
| `permissions[].mode` | `"allow" \| "deny"` | `"allow"` | No | Whether this rule permits or blocks the operations on the matched paths. |

#### `filesystem.permission_inheritance` and `filesystem.settings`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `permission_inheritance` | `bool` | `false` | No | When `true`, subagents inherit the orchestrator's filesystem permissions. When `false`, each agent uses only its own permissions. |
| `settings.tool_token_limit_before_evict` | `int` | `20000` | No | Token count threshold for tool output. Outputs exceeding this limit are evicted from conversation history to stay within context windows. |
| `settings.human_message_token_limit_before_evict` | `int` | `50000` | No | Token count threshold for human messages before eviction. |
| `settings.max_execute_timeout` | `int` | `3600` | No | Maximum execution timeout in seconds for filesystem operations. |

### Cache [YAML-loaded]

Validated by `CacheFileConfig` in `deep_agent/src/agent/config/cache.py`. All TTL fields have Pydantic range constraints.

| Field | Type | Default | Range | Required | Description |
|-------|------|---------|-------|----------|-------------|
| `cache.enabled` | `bool` | `true` | -- | No | Master switch for the entire cache subsystem. When `false`, all cache layers are disabled. |
| `cache.model.enabled` | `bool` | `true` | -- | No | Cache instantiated LLM model objects to avoid re-creating them per request. |
| `cache.model.ttl` | `int` | `600` | 10--7200 | No | Time-to-live in seconds for cached model instances. |
| `cache.model.max_size` | `int` | `50` | 1--100 | No | Maximum number of model instances to cache simultaneously. |
| `cache.personalization.enabled` | `bool` | `true` | -- | No | Cache user personalization data (memories, rules) to avoid re-fetching from the store per request. |
| `cache.personalization.ttl` | `int` | `120` | 10--3600 | No | Time-to-live in seconds for cached personalization data. |
| `cache.mcp.ttl` | `int` | `300` | 10--3600 | No | Time-to-live in seconds for the MCP tool list cache. Avoids reconnecting to MCP servers per request. |
| `cache.graph.ttl` | `int` | `300` | 10--3600 | No | Time-to-live in seconds for compiled LangGraph instances. Avoids rebuilding the graph per request. |
| `cache.redis.enabled` | `bool` | `true` | -- | No | Enable Redis as an L2 (remote) cache layer. Requires `REDIS_URL` to be configured. |
| `cache.warming.enabled` | `bool` | `true` | -- | No | Pre-warm caches at startup (model instances, graph compilation, MCP tool lists). |
| `cache.metrics.enabled` | `bool` | `true` | -- | No | Expose cache hit/miss metrics via the observability pipeline. |

### Memory Processing [env-var]

Controls background memory consolidation, decay, clustering, and injection. These settings are documented in `agent.yaml` as the canonical reference but are read at runtime.

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `memory.consolidation.enabled` | `bool` | `true` | No | Enable periodic consolidation of similar memories into summarised entries. |
| `memory.decay.enabled` | `bool` | `true` | No | Enable time-based memory decay. Older, less-accessed memories gradually lose relevance weight. |
| `memory.decay.lambda` | `float` | `0.05` | No | Decay rate parameter. Higher values cause faster memory relevance decay. |
| `memory.clustering.enabled` | `bool` | `true` | No | Enable semantic clustering of memories for more efficient retrieval. |
| `memory.clustering.threshold` | `float` | `0.4` | No | Cosine similarity threshold for grouping memories into the same cluster. |
| `memory.clustering.min_cluster_size` | `int` | `3` | No | Minimum number of memories required to form a cluster. |
| `memory.relationships.enabled` | `bool` | `true` | No | Enable relationship extraction between memories (e.g., user preferences linked to topics). |
| `memory.scheduler.interval_hours` | `int` | `6` | No | How often (in hours) the background memory processing scheduler runs. |
| `memory.max_inject` | `int` | `20` | No | Maximum number of memories injected into the system prompt per conversation turn. |

### Guardrail [YAML-loaded]

Validated by `GuardrailsConfig` in `deep_agent/src/guardrails/config.py`. Controls Granite Guardian content safety checks.

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `guardrail.enabled` | `bool` | `false` | No | Master switch for Granite Guardian content safety checks. When `false` or absent, guardrails are completely inactive. |
| `guardrail.model` | `string` | `null` | Yes (when enabled) | Model path or identifier for the Granite Guardian model (e.g., `"/data/granite-guardian-4.1-8b"`). Validation fails if `enabled: true` and `model` is not set. |

Runtime credentials for the guardrail endpoint are injected via environment variables:

| Env Var | Default | Description |
|---------|---------|-------------|
| `GUARDIAN_API_BASE` | `null` | Endpoint URL for the Granite Guardian model server. Guardrails only activate when this is set. |
| `GUARDIAN_API_KEY` | `"EMPTY"` | API key for the guardian endpoint. Use `"EMPTY"` for unauthenticated vLLM endpoints. |
| `GUARDIAN_SSL_VERIFY` | `true` | Whether to verify TLS certificates when connecting to the guardian endpoint. Set `false` only in development. |

### Token Budget [YAML-loaded]

Validated by `TokenBudgetConfig` in `deep_agent/src/token_budget/config.py`. Tracks cumulative LLM token usage per conversation `thread_id` in MongoDB.

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `token_budget.enabled` | `bool` | `false` | No | Enable per-thread token usage tracking. Requires a MongoDB connection (`MONGODB_URI`). |

### Platform / Audit [env-var]

| Field | Type | Default | Range | Required | Description |
|-------|------|---------|-------|----------|-------------|
| `platform.audit.enabled` | `bool` | `false` | -- | No | Enable audit event publishing. Env var: `PLATFORM_AUDIT_ENABLED` (default `true` in Settings). |
| `platform.audit.buffer_max` | `int` | `1000` | 1--100,000 | No | Maximum audit events buffered before flushing. Env var: `PLATFORM_AUDIT_BUFFER_MAX`. |

### OPA (Authorization) [env-var]

Validated by `OpaFileConfig` in `deep_agent/src/agent/config/opa.py`. Environment variables override YAML values when explicitly set.

| Field | Type | Default | Range | Required | Description |
|-------|------|---------|-------|----------|-------------|
| `opa.enabled` | `bool` | `false` | -- | No | Enable OPA policy evaluation for authorization. Env var: `OPA_ENABLED`. |
| `opa.url` | `string` | `"http://localhost:8181/v1/data/agent/authz"` | -- | No | OPA policy evaluation endpoint URL. Env var: `OPA_URL`. |
| `opa.timeout` | `float` | `2.0` | > 0 | No | HTTP timeout in seconds for OPA requests. Env var: `OPA_TIMEOUT`. |
| `opa.max_retries` | `int` | `0` | >= 0 | No | Maximum retry attempts for failed OPA requests. YAML default is `3`; Pydantic model default is `0`. Env var: `OPA_MAX_RETRIES`. |
| `opa.fail_open` | `bool` | `false` | -- | No | When `true`, OPA errors allow the request (fail open). When `false`, OPA errors deny the request (fail closed). |

### Logging [env-var]

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `logging.level` | `string` | `"INFO"` | No | Python log level. Valid values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Env var: `PYTHON_LOG_LEVEL`. |
| `logging.request.enabled` | `bool` | `true` | No | Enable HTTP request/response logging. Env var: `REQUEST_LOGGING_ENABLED`. |
| `logging.request.headers` | `bool` | `true` | No | Include request headers in logs. Env var: `REQUEST_LOG_HEADERS`. |
| `logging.request.body` | `bool` | `true` | No | Include request body in logs. Env var: `REQUEST_LOG_BODY`. |
| `logging.request.body_max_size` | `int` | `10240` | No | Maximum request body size in bytes to log. Bodies exceeding this are truncated. Env var: `REQUEST_LOG_BODY_MAX_SIZE`. |

### Server [env-var]

| Field | Type | Default | Range | Required | Description |
|-------|------|---------|-------|----------|-------------|
| `server.host` | `string` | `"0.0.0.0"` | -- | No | Bind address for the agent HTTP server. Env var: `AGENT_HOST`. |
| `server.port` | `int` | `5002` | 1024--65535 | No | Port for the agent HTTP server. Env var: `AGENT_PORT`. |

---

## `config/agent/mcp.json`

MCP (Model Context Protocol) server registry in JSONC format (supports `//` line comments). Defines external tool servers the agent connects to at runtime. Each entry under `mcpServers` registers one MCP server.

### `mcpServers.<name>`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `url` | `string` | -- | Yes | URL of the MCP server endpoint (e.g., `http://localhost:5001/mcp`). |
| `transport` | `string` | -- | Yes | Transport protocol. Typically `"streamable_http"` for HTTP-based MCP servers. |
| `enabled` | `bool` | -- | Yes | Whether this MCP server is active. Set `false` to disable without removing the entry. |
| `auth` | `bool` | -- | No | Whether authentication is required to connect to this MCP server. |
| `auth_mode` | `"sso" \| "oauth" \| "dcr" \| "api_key"` | `"sso"` | No | Authentication mode. `sso`: pass through the authenticated user's SSO token. `oauth`: pre-registered OAuth client credentials. `dcr`: Dynamic Client Registration (auto-registers with the MCP server). `api_key`: static API key from an environment variable. |
| `ssl_verify` | `bool` | `true` | No | Whether to verify TLS certificates. Set `false` only for local development. Forced to `true` when `ENVIRONMENT=production`. |
| `timeout` | `int` | `30` | No | Connection and request timeout in seconds. |
| `tool_prefix` | `string` | -- | No | Prefix added to all tool names from this server to avoid name collisions (e.g., prefix `"template"` turns `calculate_bmi` into `template_calculate_bmi`). |
| `auth_env_var` | `string` | -- | No | Environment variable name containing the API key. Required when `auth_mode: api_key`. |

### `mcpServers.<name>.oauth`

Required when `auth_mode` is `"oauth"` or `"dcr"`. Configures the OAuth 2.0 flow for this MCP server.

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `oauth.authorization_endpoint` | `string` | -- | Yes (for auth code flow) | OAuth authorization endpoint URL. Not required when `grant_type: client_credentials`. |
| `oauth.token_endpoint` | `string` | -- | Yes | OAuth token endpoint URL for exchanging codes/credentials for tokens. |
| `oauth.registration_endpoint` | `string` | -- | Yes (for `dcr`) | Dynamic Client Registration endpoint. Required only when `auth_mode: dcr`. |
| `oauth.scopes` | `list[string]` | `[]` | No | OAuth scopes to request (e.g., `["email", "openid", "profile"]`). |
| `oauth.client_id` | `string` | -- | Yes (for `oauth`) | Pre-registered OAuth client ID. Required when `auth_mode: oauth`; not needed for `dcr` (auto-registered). |
| `oauth.client_secret_env` | `string` | -- | No | Environment variable name containing the client secret. Preferred over `client_secret` for security. |
| `oauth.client_secret` | `string` | -- | No | OAuth client secret (insecure -- use `client_secret_env` instead). Generates a warning if present. |
| `oauth.grant_type` | `string` | `"authorization_code"` | No | OAuth grant type. `authorization_code` (default) for user-interactive flows; `client_credentials` for service-to-service. Not compatible with `auth_mode: dcr`. |
| `oauth.redirect_uri` | `string` | -- | No | Ignored if present -- redirect URI is always derived from `AGENT_PUBLIC_BASE_URL`. Generates a warning. |

---

## `config/agent/runtime/pii.yaml`

PII (Personally Identifiable Information) detection and scrubbing configuration. Validated by `PIIConfig` in `deep_agent/src/pii/config.py` and `deep_agent/src/agent/config/middleware.py`. If this file does not exist, PII processing is disabled.

### Top-Level Fields

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `enabled` | `bool` | `false` | No | Master switch for PII processing. Set `false` to keep the config file but disable processing. |
| `trace_strategy` | `"redact" \| "hash"` | `"hash"` | No | How PII appears in Langfuse traces. `redact`: replaces with `***REDACTED***`. `hash`: replaces with `[HASH:abc123]` for cross-request correlation. |

### `rules[]`

Each rule defines a PII type to detect and how to handle it.

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `rules[].name` | `string` | -- | Yes | Identifier for this PII type (e.g., `credit_card`, `email`, `pan_card`). Also accepts legacy `type` field as alias. |
| `rules[].strategy` | `"scrub" \| "mask" \| "hash" \| "redact" \| "block"` | `"scrub"` | No | How to handle detected PII. `scrub`: reversible tokenization (`[EMAIL_1]`, restored in LLM output). `mask`: partial mask preserving last 4 chars. `hash`: one-way HMAC-SHA256. `redact`: one-way `***REDACTED***`. `block`: reject the entire request if this PII type appears. |
| `rules[].provider` | `"default" \| "regex" \| "presidio" \| "custom"` | `"regex"` | No | Detection backend. `default`: stock LangChain PIIMiddleware (one-way, parallel). `regex`: token-map scrubber with built-in regex patterns. `presidio`: token-map scrubber with Presidio NLP. `custom`: token-map scrubber with your own regex (requires `regex` field). |
| `rules[].regex` | `string` | `null` | Yes (when `provider: custom`) | Custom regular expression for PII detection. Required when `provider` is `custom`; ignored otherwise. |
| `rules[].label` | `string` | `name.upper()` | No | Token label prefix used in placeholders (e.g., `MAIL` produces `[MAIL_1]`). Defaults to the uppercase `name`. |

### PII Environment Variables

| Env Var | Type | Default | Range | Description |
|---------|------|---------|-------|-------------|
| `PII_HASH_KEY` | `string` | `null` | -- | HMAC key for the `hash` strategy. If unset, a random key is generated per process (not stable across restarts). |
| `PII_TOKEN_MAP_TTL_DAYS` | `int` | `7` | 1--365 | How long PII token mappings are retained for `scrub` restoration. |

---

## `config/agent/runtime/observability.yaml`

OpenTelemetry configuration for operational metrics and distributed tracing. Validated by `OtelFileConfig` in `deep_agent/src/agent/config/otel.py`. Langfuse (LLM trace quality) auto-activates via environment variables and does not need YAML config.

### Langfuse [env-var]

Langfuse auto-activates when the following environment variables are set. No YAML configuration is needed.

| Env Var | Type | Default | Required | Description |
|---------|------|---------|----------|-------------|
| `LANGFUSE_PUBLIC_KEY` | `string` | `null` | Yes (to activate) | Langfuse project public key. |
| `LANGFUSE_SECRET_KEY` | `string` | `null` | Yes (to activate) | Langfuse project secret key. |
| `LANGFUSE_BASE_URL` | `string` | `null` | Yes (to activate) | Langfuse server URL (e.g., `https://cloud.langfuse.com`). |
| `LANGFUSE_TRACING_ENVIRONMENT` | `string` | `"development"` | No | Environment label attached to Langfuse traces. Set via ConfigMap in OpenShift. |

### OpenTelemetry (`otel`) [YAML-loaded]

Disabled by default for local development. Enable by setting `enabled: true` or `ENABLE_OTEL=true`. Environment variable overrides take precedence over YAML values.

| Field | Type | Default | Range | Required | Description |
|-------|------|---------|-------|----------|-------------|
| `otel.enabled` | `bool` | `false` | -- | No | Master switch for OTEL export. Env override: `ENABLE_OTEL`. |
| `otel.exporter.endpoint` | `string` | `"http://localhost:4317"` | -- | No | OTLP gRPC exporter endpoint. Env override: `OTEL_EXPORTER_OTLP_ENDPOINT`. |
| `otel.exporter.insecure` | `bool` | `true` | -- | No | Use insecure (non-TLS) gRPC connection. Set `false` in production. Env override: `OTEL_EXPORTER_OTLP_INSECURE`. |
| `otel.metrics.export_interval_ms` | `int` | `5000` | 1000--60000 | No | Metric export interval in milliseconds. Env override: `OTEL_METRIC_EXPORT_INTERVAL`. |
| `otel.tracing.fastapi_auto_instrument` | `bool` | `true` | -- | No | Automatically instrument FastAPI routes with OTEL tracing spans. |

---

## `config/agent/runtime/ui.yaml`

Frontend BFF (Backend for Frontend) configuration. Mounted as a ConfigMap in OpenShift deployments. Controls server binding, security policies, observability, and feature flags for the UI layer.

### Server

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `server.host` | `string` | `"0.0.0.0"` | No | Bind address for the UI BFF server. |
| `server.port` | `int` | `8080` | No | Port for the UI BFF server. |
| `server.body_limit` | `int` | `1048576` | No | Maximum request body size in bytes (1 MB default). |

### Logging

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `logging.level` | `"debug" \| "info" \| "warn" \| "error" \| "silent"` | `"info"` | No | UI BFF log level. |

### CORS

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `cors.origin` | `string` | `"http://localhost:5173"` | No | Allowed CORS origin. Set to your frontend URL in production. |

### Security

#### `security.helmet`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `helmet.enabled` | `bool` | `true` | No | Enable Helmet.js security headers. |
| `helmet.csp.default_src` | `list[string]` | `["'self'"]` | No | Content Security Policy `default-src` directive. |
| `helmet.csp.script_src` | `list[string]` | `["'self'", "'unsafe-inline'"]` | No | CSP `script-src`. `unsafe-inline` is needed for HTML shell script blocks. |
| `helmet.csp.style_src` | `list[string]` | `["'self'", "'unsafe-inline'"]` | No | CSP `style-src`. `unsafe-inline` is needed because PatternFly injects inline styles. |
| `helmet.csp.img_src` | `list[string]` | `["'self'", "data:", "blob:"]` | No | CSP `img-src`. |
| `helmet.csp.connect_src` | `list[string]` | `["'self'"]` | No | CSP `connect-src`. |
| `helmet.csp.font_src` | `list[string]` | `["'self'", "data:"]` | No | CSP `font-src`. |
| `helmet.csp.object_src` | `list[string]` | `["'none'"]` | No | CSP `object-src`. |
| `helmet.csp.frame_src` | `list[string]` | `["'self'"]` | No | CSP `frame-src`. Required for MCP Apps sandbox proxy iframe. |
| `helmet.csp.frame_ancestors` | `list[string]` | `["'none'"]` | No | CSP `frame-ancestors`. Prevents embedding this page in iframes. |
| `helmet.cross_origin_embedder_policy` | `bool` | `false` | No | Must be `false` -- enabling breaks SSE streaming. |

#### `security.rate_limit`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `rate_limit.enabled` | `bool` | `true` | No | Enable per-IP rate limiting. |
| `rate_limit.max` | `int` | `100` | No | Maximum requests per time window per IP address. |
| `rate_limit.window` | `string` | `"1 minute"` | No | Time window for rate limiting (human-readable duration). |
| `rate_limit.exclude_paths` | `list[string]` | `["/api/health", "/_health"]` | No | URL paths exempt from rate limiting. |

#### `security.session`

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `session.secure_cookie` | `bool` | `false` | No | Set `true` in production (requires HTTPS) to mark session cookies as `Secure`. |
| `session.max_age_days` | `int` | `30` | No | Session cookie lifetime in days. |

### Observability

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `otel.enabled` | `bool` | `true` | No | Enable OTEL tracing for the UI BFF. |
| `otel.service_name` | `string` | `"template-ui"` | No | OTEL service name reported in traces. |

### Features

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `features.mcp_apps_enabled` | `bool` | `true` | No | Serve `/sandbox_proxy.html` for MCP Apps host rendering in an iframe. |

### Announcement Banner

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `announcement.enabled` | `bool` | `true` | No | Show a global announcement banner in the UI. |
| `announcement.message` | `string` | `""` | No | Banner message text. |
| `announcement.type` | `"info" \| "warning" \| "danger" \| "success"` | `"info"` | No | Banner visual style/severity. |

---

## `config/agent/runtime/secrets.example.yaml`

This file is a documentation-only reference. It lists every secret the agent expects but **never stores actual values**. All secrets are injected via OpenShift Secrets mounted as environment variables.

### Google Cloud

| Secret | Default | Description |
|--------|---------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS_CONTENT` | -- | Service account JSON for Vertex AI (Gemini + Claude models). |

### Database (PostgreSQL)

| Secret | Default | Description |
|--------|---------|-------------|
| `POSTGRES_HOST` | `pgvector` | PostgreSQL hostname. |
| `POSTGRES_PORT` | `5432` | PostgreSQL port. |
| `POSTGRES_DB` | `pgvector` | Database name. |
| `POSTGRES_USER` | `pgvector` | Database user. |
| `POSTGRES_PASSWORD` | `pgvector` | Database password. |

### Observability (Langfuse)

| Secret | Description |
|--------|-------------|
| `LANGFUSE_PUBLIC_KEY` | Langfuse project public key. |
| `LANGFUSE_SECRET_KEY` | Langfuse project secret key. |
| `LANGFUSE_BASE_URL` | Langfuse server URL. |

### Cache (Redis)

| Secret | Description |
|--------|-------------|
| `REDIS_HOST` | Redis/Valkey host (e.g., ElastiCache NLB endpoint). |
| `REDIS_PORT` | Redis port (e.g., `6379`). |
| `REDIS_TLS` | Always `true` for AWS ElastiCache Serverless. |

### SSL

| Secret | Description |
|--------|-------------|
| `SSL_KEYFILE` | TLS private key file path (only when TLS termination is at app level). |
| `SSL_CERTFILE` | TLS certificate file path. |

### Async Subagents

| Secret | Description |
|--------|-------------|
| `ASYNC_SUBAGENT_<NAME>_TOKEN` | Bearer token for each async subagent. Convention: uppercase the subagent name (e.g., `ASYNC_SUBAGENT_RESEARCHER_TOKEN`). |

### MCP Servers

| Secret | Description |
|--------|-------------|
| `MCP_AUTH_TOKEN` | Auth token for MCP servers using SSO token passthrough. |

### UI Session

| Secret | Description |
|--------|-------------|
| `COOKIE_SIGN` | Session cookie signing secret (minimum 32 characters). |

### UI Auth (OAuth/SSO)

| Secret | Description |
|--------|-------------|
| `AUTH_ENABLED` | `"true"` to enable SSO for the UI. |
| `AUTH_CLIENT_ID` | OAuth client ID for the UI. |
| `AUTH_CLIENT_SECRET` | OAuth client secret for the UI. |
| `AUTH_DISCOVERY_URL` | OpenID Connect discovery endpoint for the UI. |

### UI Infrastructure

| Secret | Default | Description |
|--------|---------|-------------|
| `AGENT_HOST` | `http://localhost:5002` | Agent backend URL for the UI BFF. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | OTEL collector endpoint (shared with agent). |

### Build Metadata

| Variable | Description |
|----------|-------------|
| `APP_VERSION` | Application version (e.g., `1.2.3`). Injected by CI. |
| `BUILD_HASH` | Git short SHA (e.g., `abc1234`). Injected by CI. |
| `BUILD_TIME` | Build timestamp in ISO 8601 format. Injected by CI. |

---

## `config/agent/PROMPT.md` Frontmatter

The orchestrator prompt file uses YAML frontmatter (delimited by `---`) to declare agent identity, model binding, tool/skill selection, access control, and MCP server filtering. The same schema applies to subagent files in `config/agent/subagents/*.md`.

Frontmatter is parsed by `parse_frontmatter()` in `deep_agent/src/agent/config/parser.py`. The markdown body supports `{{current_date}}` template variables, which are replaced at runtime via `inject_runtime_values()`.

### Frontmatter Fields

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `name` | `string` | filename stem | Yes | Unique identifier for this agent. Used in logs, routing, and as the agent node name in the LangGraph. |
| `description` | `string` | -- | No | Human-readable description of the agent's purpose. Shown to the orchestrator when deciding which subagent to delegate to. |
| `type` | `"default" \| "compiled"` | `"default"` | No | Agent type. `default`: standard ReAct agent. `compiled`: compiled subgraph (used for subagents that need tighter execution control). Only meaningful for subagent files. |
| `model` | `string \| object` | -- | Yes | LLM model to use. Accepts two formats (see below). |
| `tools` | `list[string]` | `[]` | No | Tool names this agent can invoke. Names must match registered tool functions (after MCP `tool_prefix` is applied). Tools not listed here are not available to the agent. |
| `skills` | `list[string]` | `[]` | No | Skill directory names from `config/agent/skills/`. The skill's `SKILL.md` and `references/` are injected into the agent's context via `SkillsMiddleware`. |
| `mcps` | `list[string]` | all (when omitted) | No | MCP server names (keys from `mcp.json`) this agent may connect to. Omitting the field allows access to all enabled MCP servers. An empty list `[]` blocks all MCP servers. |
| `resources` | `list[string]` | all (when omitted) | No | Resource names this agent may access. Omitting the field allows access to all resources. Must be a list of strings if present. |
| `accessibility` | `"private" \| "public"` | `"private"` | No | Access control mode. `private`: users not in any LDAP group are denied (403). `public`: users not in any group get chat access (`users` role) but cannot access eval/dataset APIs. |
| `groups` | `list[object]` | `[]` | No | LDAP group-to-role mappings for access control. See below. |
| `middleware` | `object` | -- | No | Per-agent middleware overrides. Same structure as `middleware` in `agent.yaml`. Merged on top of global defaults and profile settings. |

#### `model` Field Formats

**String format** (legacy -- provider is inferred):
```yaml
model: gemini-2.5-pro
```

Provider inference: known Gemini/Claude models map to `vertex`, `gpt-*` models map to `openai`, everything else maps to `maas` (vLLM).

**Object format** (explicit provider and optional fallback):
```yaml
model:
  provider: vertex    # vertex | openai | maas
  name: gemini-2.5-pro
  fallback:
    provider: maas
    name: granite-3.1-8b-instruct
```

| Sub-field | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `model.provider` | `"vertex" \| "openai" \| "maas"` | inferred from name | No | LLM provider backend. `vertex`: Google Vertex AI (Gemini + Claude). `openai`: OpenAI API. `maas`: Models as a Service via vLLM/OpenAI-compatible endpoint. |
| `model.name` | `string` | -- | Yes | Model identifier (e.g., `gemini-2.5-pro`, `claude-sonnet-4`, `granite-3.1-8b-instruct`). |
| `model.fallback` | `object` | `null` | No | Fallback model specification (same `provider`/`name` structure). Used when the primary model is unavailable. Nested fallback chains are not supported. |

#### `groups[]` Field

| Sub-field | Type | Required | Description |
|-----------|------|----------|-------------|
| `groups[].role` | `"owners" \| "admins" \| "builders" \| "users"` | Yes | Role level. Hierarchy: `owners` > `admins` > `builders` > `users`. The highest matching role wins. |
| `groups[].group` | `string` | Yes | LDAP group name. The SSO user's `preferred_username` is checked against this group's membership. |

---

## `.env.example` -- Environment Variables

Reference for all environment variables consumed by the agent. Grouped by concern. Only secrets and infrastructure endpoints belong here; all operational config lives in `agent.yaml`.

### Environment

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENVIRONMENT` | `string` | `"development"` | Runtime environment. Set to `"production"` to enforce: `ENABLE_AUTH=true`, HTTPS for public URLs, MCP `ssl_verify` enforcement, PII scrubbing in error responses, and security headers. |

### Security

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REQUEST_BODY_MAX_SIZE` | `int` | `10485760` (10 MB) | Maximum HTTP request body size in bytes. DoS protection. |

### SSO / OIDC Authentication

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_AUTH` | `bool` | `true` | Enable SSO/OIDC authentication. Must be `true` in production. |
| `SSO_ISSUER_URL` | `string` | `null` | OIDC issuer URL (e.g., `https://sso.example.com/realms/myrealm`). |
| `SSO_CLIENT_ID` | `string` | `null` | OIDC client ID. |
| `SSO_CLIENT_SECRET` | `string` | `null` | OIDC client secret. |
| `SSO_DEV_USERNAME` | `string` | `"John Doe"` | Fallback display name when `ENABLE_AUTH=false` (local dev). |
| `SSO_DEV_USER_ID` | `string` | `"dev-user"` | Fallback user ID when `ENABLE_AUTH=false` (local dev). |
| `ENABLE_USER_ID_ENCRYPTION` | `bool` | `false` | Encrypt user IDs in observability data for privacy. |

### MCP OAuth / DCR

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MCP_TOKEN_ENCRYPTION_KEY` | `string` | `null` | Fernet key for encrypting OAuth access/refresh tokens in Redis and DCR client secrets in Postgres. Generate with: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS` | `string` | `null` | Previous encryption key during rotation (decrypt only). |
| `AGENT_PUBLIC_BASE_URL` | `string` | `http://localhost:{AGENT_PORT}` | Public base URL for MCP OAuth connect/callback endpoints. Must use `https://` in production (HTTP permitted only for localhost). |
| `UI_ORIGIN` | `string` | `null` | Frontend UI origin for `postMessage` in OAuth callback (e.g., `http://localhost:5173`). Falls back to `AGENT_PUBLIC_BASE_URL`. |

### LDAP Access Control

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LDAP_URL` | `string` | -- | LDAP server URL (e.g., `ldaps://ldap.example.com`). Base DN is derived from the hostname. Required when PROMPT.md has `groups`; if missing, all users are denied (fail-closed). |
| `LDAP_BASE_UID` | `string` | -- | Service account UID for LDAP bind (e.g., `svcacct`). If the value contains a comma, it is used as-is as the full bind DN. |
| `LDAP_PASSWORD` | `string` | -- | LDAP service account password. |
| `LDAP_GROUP_SEARCH_BASE` | `string` | `ou=adhoc,ou=managedGroups,<derived>` | Override the LDAP subtree searched for groups. |
| `LDAP_TLS_VERIFY` | `bool` | `true` | Validate LDAP server TLS certificate. Set `false` only for dev/test with self-signed certs. |
| `LDAP_CACHE_TTL_SECONDS` | `int` | `300` | How long LDAP membership results are cached in Redis (seconds). |

### Infrastructure -- PostgreSQL

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `POSTGRES_HOST` | `string` | `"pgvector"` | PostgreSQL hostname. |
| `POSTGRES_PORT` | `int` | `5432` | PostgreSQL port. |
| `POSTGRES_DB` | `string` | `"template_agent"` | Database name. |
| `POSTGRES_USER` | `string` | `"postgres"` | Database user. |
| `POSTGRES_PASSWORD` | `string` | `"postgres"` | Database password. |

### Infrastructure -- Redis

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS_URL` | `string` | `"redis://redis:6379/0"` | Redis connection URI. Used by the Aegra broker for SSE streaming, job queue, and crash recovery. |
| `REDIS_BROKER_ENABLED` | `bool` | `true` | Enable the Redis-backed Aegra broker. |

### Infrastructure -- MongoDB

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MONGODB_URI` | `string` | `null` | MongoDB connection URI for token usage tracking. May contain credentials -- never commit or log this value. |
| `MONGODB_DB` | `string` | `"tokenusage"` | MongoDB database name. |

### Observability

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LANGFUSE_PUBLIC_KEY` | `string` | `null` | Langfuse public key. |
| `LANGFUSE_SECRET_KEY` | `string` | `null` | Langfuse secret key. |
| `LANGFUSE_BASE_URL` | `string` | `null` | Langfuse server URL. |
| `LANGFUSE_TRACING_ENVIRONMENT` | `string` | `"development"` | Environment label for Langfuse traces. |
| `ENABLE_OTEL` | `bool` | `false` | Enable OTEL export (overrides `otel.enabled` in observability.yaml). |
| `ENABLE_OTEL_METRICS` | `bool` | `false` | Enable OTEL metrics export. |
| `ENABLE_OTEL_TRACES` | `bool` | `false` | Enable OTEL trace export. |
| `OTEL_SERVICE_NAME` | `string` | `"template-agent"` | Service name reported in OTEL spans. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `string` | `null` | OTLP gRPC metrics endpoint. |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | `string` | `null` | OTLP gRPC traces endpoint (e.g., Jaeger at `:4317`). |
| `OTEL_AUTH_TOKEN` | `string` | `null` | Bearer token for authenticated OTEL collectors. |
| `OTEL_METRIC_EXPORT_INTERVAL_MILLIS` | `int` | `10000` | Metric export interval in milliseconds. |

### Model Provider Credentials

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS_CONTENT` | `string` | `null` | Service account JSON for Vertex AI. If unset, Application Default Credentials (ADC) are used. |
| `GOOGLE_CLOUD_PROJECT` | `string` | `null` | GCP project ID override. |
| `VLLM_BASE_URL` | `string` | `null` | vLLM / OpenAI-compatible inference server endpoint (e.g., `http://vllm-server:8000/v1`). |
| `VLLM_API_KEY` | `string` | `"EMPTY"` | API key for vLLM endpoint. Use `"EMPTY"` for unauthenticated endpoints. |

### Granite Guardian Guardrails

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GUARDIAN_API_BASE` | `string` | `null` | Granite Guardian endpoint URL. Guardrails activate when this is set. |
| `GUARDIAN_API_KEY` | `string` | `"EMPTY"` | API key for the guardian endpoint. |
| `GUARDIAN_SSL_VERIFY` | `bool` | `true` | Verify TLS certificates for the guardian endpoint. |

### Gemini Safety Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SAFETY_DANGEROUS_CONTENT` | `SafetyThreshold` | `"BLOCK_MEDIUM_AND_ABOVE"` | Vertex AI safety threshold for dangerous content. |
| `SAFETY_HATE_SPEECH` | `SafetyThreshold` | `"BLOCK_MEDIUM_AND_ABOVE"` | Vertex AI safety threshold for hate speech. |
| `SAFETY_HARASSMENT` | `SafetyThreshold` | `"BLOCK_LOW_AND_ABOVE"` | Vertex AI safety threshold for harassment. |
| `SAFETY_SEXUALLY_EXPLICIT` | `SafetyThreshold` | `"BLOCK_LOW_AND_ABOVE"` | Vertex AI safety threshold for sexually explicit content. |

Valid `SafetyThreshold` values: `BLOCK_NONE`, `BLOCK_LOW_AND_ABOVE`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_ONLY_HIGH`.

### OPA

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPA_URL` | `string` | `null` | OPA policy evaluation endpoint (overrides `opa.url` in agent.yaml). |
| `OPA_TIMEOUT` | `float` | `null` | OPA request timeout in seconds (overrides `opa.timeout`). Must be > 0. |
| `OPA_MAX_RETRIES` | `int` | `null` | OPA retry count (overrides `opa.max_retries`). Must be >= 0. |

### Platform

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEPLOYED_AGENT_NAME` | `string` | `""` | Agent name set by the deployment platform. Used in DCR/token keys. |
| `DEPLOYED_AGENT_ORG` | `string` | `""` | Organisation context for the deployed agent. |
| `PLATFORM_AUDIT_ENABLED` | `bool` | `true` | Enable platform audit event publishing. |
| `PLATFORM_AUDIT_BUFFER_MAX` | `int` | `1000` | Maximum buffered audit events (range: 1--100,000). |

### Runtime

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PYTHON_LOG_LEVEL` | `string` | `"INFO"` | Python logging level. Valid: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `CONFIG_AUTO_RELOAD` | `bool` | `true` | Reload config from disk on every access. Useful for development; disable in production for performance. |
| `CACHE_ENABLED` | `bool` | `true` | Global cache master switch (env var level). |
| `MEMORY_ENABLED` | `bool` | `true` | Global memory master switch (env var level). |
| `MIDDLEWARE_ENABLED` | `bool` | `true` | Global middleware master switch (env var level). |
| `ENABLE_CLI` | `bool` | `true` | Enable CLI interface for the agent. |

### Lifecycle Persistence

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LIFECYCLE_PERSISTENCE_ENABLED` | `bool` | `true` | Enable lifecycle state persistence for crash recovery. |
| `LIFECYCLE_LEASE_SECONDS` | `int` | `300` | Lease duration in seconds for lifecycle state locks. |
| `LIFECYCLE_MAX_RESUME_BATCH` | `int` | `10` | Maximum number of stale lifecycle entries to resume per startup. |
| `LIFECYCLE_RESUME_ON_STARTUP` | `bool` | `true` | Automatically resume interrupted conversations on startup. |

### Eval Runner

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EVAL_RUNNER_URL` | `string` | `null` | URL of the eval-runner service. The agent skips eval calls gracefully when unset. |
| `EVAL_INTERNAL_TOKEN` | `string` | `null` | Shared secret between the agent and eval-runner. Required when running eval-runner. |
| `EVAL_TOKEN_REFRESH_ENABLED` | `bool` | `false` | Enable OIDC token refresh for eval-runner callbacks. |
