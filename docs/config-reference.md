# Config reference

**Generated — do not edit.** Regenerate with `make config-reference`.

Every key the agent reads from `config/agent/`, with its type and
default, derived from the Pydantic models that validate it. If a key
is not here, the agent does not read it.

Files not covered below are not schema-validated: `PROMPT.md` and
`subagents/*.md` are Markdown with YAML frontmatter, `mcp.json` is
JSONC, `skills/` are directories, and `compliance/policies/*.rego`
are OPA policies. `make validate-config` checks all of them.

Run `make validate-config` after editing anything here.

### Middleware

**File:** `runtime/agent.yaml` → `middleware:`

Global middleware defaults from agent.yaml.

| Key | Type | Default | Notes |
|---|---|---|---|
| `summarization_tool` | SummarizationToolConfig | _(section)_ |  |
| `summarization_tool.enabled` | bool | `True` |  |
| `human_approval` | HumanApprovalConfig | _(section)_ |  |
| `human_approval.enabled` | bool | `True` |  |
| `human_approval.mode` | 'all' \| 'none' | `'all'` |  |
| `human_approval.exclude` | list[str] | _(empty)_ |  |
| `human_approval.policies` | list[ApprovalPolicy] | _(empty)_ |  |
| `human_approval.default_decision` | 'allow' \| 'ask' \| 'deny' | `'ask'` |  |
| `memory` | MemoryConfig | _(section)_ |  |
| `memory.enabled` | bool | `True` |  |
| `memory.namespaces` | list[str] | `['memories']` |  |
| `patch_tool_calls` | PatchToolCallsConfig | _(section)_ |  |
| `patch_tool_calls.enabled` | bool | `True` |  |
| `skills` | SkillsConfig | _(section)_ |  |
| `skills.enabled` | bool | `True` |  |
| `model_call_limit` | ModelCallLimitConfig | _(section)_ |  |
| `model_call_limit.enabled` | bool | `True` |  |
| `model_call_limit.run_limit` | int | `50` |  |
| `tool_call_limit` | ToolCallLimitConfig | _(section)_ |  |
| `tool_call_limit.enabled` | bool | `True` |  |
| `tool_call_limit.run_limit` | int | `200` |  |
| `tool_call_limit.per_tool` | map[int] | _(empty)_ |  |
| `tool_exclusion` | ToolExclusionConfig | _(section)_ |  |
| `tool_exclusion.enabled` | bool | `True` |  |
| `tool_exclusion.tools` | list[str] | _(empty)_ |  |
| `model_retry` | ModelRetryConfig | _(section)_ |  |
| `model_retry.enabled` | bool | `True` |  |
| `model_retry.max_retries` | int | `3` |  |
| `model_retry.backoff_factor` | float | `2.0` |  |
| `model_retry.initial_delay` | float | `1.0` |  |
| `model_fallback` | ModelFallbackConfig | _(section)_ |  |
| `model_fallback.enabled` | bool | `False` |  |
| `model_fallback.fallback_model` | str | `''` |  |
| `tool_retry` | ToolRetryConfig | _(section)_ |  |
| `tool_retry.enabled` | bool | `False` |  |
| `tool_retry.max_retries` | int | `2` |  |
| `tool_retry.tools` | list[str] | _(empty)_ |  |
| `extra` | list[str] | _(empty)_ |  |


### Providers

**File:** `runtime/agent.yaml` → `providers:`

Structure of the providers + harness_profiles sections in runtime/agent.yaml.

| Key | Type | Default | Notes |
|---|---|---|---|
| `resolve_strategy` | 'legacy' \| 'deepagents' | `'legacy'` |  |
| `providers` | map[ProviderConfig] | _(empty)_ |  |
| `harness_profiles` | map[HarnessProfileConfig] | _(empty)_ |  |
| `async_tasks` | AsyncTaskConfig | _(section)_ |  |
| `async_tasks.enabled` | bool | `True` |  |
| `async_tasks.system_prompt` | str \| null | `None` |  |


### Cache

**File:** `runtime/agent.yaml` → `cache:`

Top-level cache configuration from agent.yaml ``cache:`` section.

| Key | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `True` |  |
| `model` | ModelCacheConfig | _(section)_ |  |
| `model.enabled` | bool | `True` |  |
| `model.ttl` | int | `600` |  |
| `model.max_size` | int | `50` |  |
| `personalization` | PersonalizationCacheConfig | _(section)_ |  |
| `personalization.enabled` | bool | `True` |  |
| `personalization.ttl` | int | `120` |  |
| `mcp` | McpCacheConfig | _(section)_ |  |
| `mcp.ttl` | int | `300` |  |
| `graph` | GraphCacheConfig | _(section)_ |  |
| `graph.ttl` | int | `300` |  |
| `redis` | ToggleConfig | _(section)_ |  |
| `redis.enabled` | bool | `True` |  |
| `warming` | ToggleConfig | _(section)_ |  |
| `warming.enabled` | bool | `True` |  |
| `metrics` | ToggleConfig | _(section)_ |  |
| `metrics.enabled` | bool | `True` |  |


### Filesystem

**File:** `runtime/agent.yaml` → `filesystem:`

Structure of the ``filesystem:`` section in runtime/agent.yaml.

| Key | Type | Default | Notes |
|---|---|---|---|
| `backend` | BackendConfig | _(section)_ |  |
| `backend.type` | 'state' \| 'composite' \| 'store' \| 'local_shell' | `'state'` |  |
| `backend.local_shell` | LocalShellConfig | _(section)_ |  |
| `backend.local_shell.timeout` | int | `120` |  |
| `backend.local_shell.max_output_bytes` | int | `100000` |  |
| `backend.state` | StateConfig | _(section)_ |  |
| `backend.state.enabled` | bool | `False` |  |
| `backend.store` | StoreConfig | _(section)_ |  |
| `backend.store.enabled` | bool | `False` |  |
| `backend.store.scope` | 'user' \| 'assistant' \| 'org' | `'user'` |  |
| `backend.routes` | map[str] | _(empty)_ |  |
| `backend.roots` | map[str] | _(empty)_ | Optional per-route directory override, relative to the config base dir. Defaults to the route prefix itself (e.g. "/skills/" → skills). |
| `permissions` | list[PermissionRule] | _(empty)_ |  |
| `permission_inheritance` | bool | `False` |  |
| `settings` | FilesystemSettings | _(section)_ |  |
| `settings.tool_token_limit_before_evict` | int | `20000` |  |
| `settings.human_message_token_limit_before_evict` | int | `50000` |  |
| `settings.max_execute_timeout` | int | `3600` |  |


### Policy engine (OPA)

**File:** `runtime/agent.yaml` → `opa:`

OPA authorization config from agent.yaml ``opa:`` section.

| Key | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `False` |  |
| `url` | str | `'http://localhost:8181/v1/data/agent/authz'` |  |
| `timeout` | float | `2.0` |  |
| `max_retries` | int | `0` |  |
| `fail_open` | bool | `False` | When True, OPA errors allow the request (fail open). When False, OPA errors deny the request (fail closed). |


### Execution

**File:** `runtime/agent.yaml` → `execution:`

What a run is allowed to actually do.

| Key | Type | Default | Notes |
|---|---|---|---|
| `effects` | 'live' \| 'dry_run' | `'live'` |  |


### Observability

**File:** `runtime/observability.yaml`

Top-level OTEL configuration from observability.yaml ``otel:`` section.

| Key | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `False` |  |
| `exporter` | OtelExporterConfig | _(section)_ |  |
| `exporter.endpoint` | str | `'http://localhost:4317'` |  |
| `exporter.insecure` | bool | `True` |  |
| `metrics` | OtelMetricsConfig | _(section)_ |  |
| `metrics.export_interval_ms` | int | `5000` |  |
| `tracing` | OtelTracingConfig | _(section)_ |  |
| `tracing.fastapi_auto_instrument` | bool | `True` |  |


### Model routing

**File:** `runtime/router.yaml`

Parsed ``router.yaml``.

| Key | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `False` |  |
| `strategy` | 'first_message' \| 'every_turn' | `'first_message'` |  |
| `tiers` | list[ModelTier] | _(empty)_ |  |
| `fallback_tier` | str \| null | `None` |  |
| `classifier_model` | str | `''` |  |
| `classifier_timeout` | float | `10.0` |  |
