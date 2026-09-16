# Middleware Extension Guide

This guide explains how the template-agent middleware pipeline works, how to write custom middleware, and how to register it. For the YAML configuration options that control built-in middleware, see the [Configuration Reference](./01-configuration-reference.md).

---

## 1. Middleware Architecture Overview

The middleware pipeline sits between the agent and the LLM/tools. Every LLM call and tool execution passes through the middleware stack, giving each middleware a chance to inspect, modify, or short-circuit the operation.

### Request Flow

```
                        ┌───────────────────────────────────┐
                        │         Agent (LangGraph)         │
                        └──────────┬────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   before_agent / abefore_agent│
                    │   (once per agent invocation) │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │          Agent reasoning loop           │
              │                                        │
              │  ┌──────────────────────────────────┐  │
              │  │ before_model / abefore_model      │  │
              │  │ (before each LLM call)            │  │
              │  └──────────────┬───────────────────┘  │
              │                 │                       │
              │  ┌──────────────▼───────────────────┐  │
              │  │ wrap_model_call / awrap_model_call│  │
              │  │ (wraps the actual LLM invocation) │  │
              │  │                                   │  │
              │  │   ┌───────────────────────┐       │  │
              │  │   │    LLM Provider       │       │  │
              │  │   │ (Gemini/Claude/vLLM)  │       │  │
              │  │   └───────────────────────┘       │  │
              │  │                                   │  │
              │  └──────────────┬───────────────────┘  │
              │                 │                       │
              │  ┌──────────────▼───────────────────┐  │
              │  │ after_model / aafter_model        │  │
              │  │ (after each LLM call)             │  │
              │  └──────────────┬───────────────────┘  │
              │                 │                       │
              │     (if LLM returned tool calls)       │
              │                 │                       │
              │  ┌──────────────▼───────────────────┐  │
              │  │ wrap_tool_call / awrap_tool_call  │  │
              │  │ (wraps each tool execution)       │  │
              │  │                                   │  │
              │  │   ┌───────────────────────┐       │  │
              │  │   │   Tool Execution      │       │  │
              │  │   │   (MCP / built-in)    │       │  │
              │  │   └───────────────────────┘       │  │
              │  │                                   │  │
              │  └─────────────────────────────────┘  │
              │                                        │
              │        (loop until agent is done)      │
              └────────────────────────────────────────┘
```

### Resolution Order

The final middleware configuration for any agent is determined by merging three layers:

1. **Global defaults** -- the `middleware:` section in `config/agent/runtime/agent.yaml`
2. **Harness profile** -- matched by model name from the `harness_profiles:` section (e.g., `claude-sonnet-4` can exclude `patch_tool_calls`)
3. **Per-agent frontmatter overrides** -- the `middleware:` block in a subagent's `.md` config file

The merge is performed by `resolve_middleware()` in `deep_agent/src/agent/config/middleware.py`. For each setting, later layers override earlier ones. The `extra` middleware list is additive -- agent-level extras are appended to global extras, not replaced.

---

## 2. The AgentMiddleware Base Class

All middleware extends the `AgentMiddleware` base class from LangChain:

```python
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
```

### Hook Reference

Each hook has a sync and async variant. Implement whichever your middleware needs -- you do not need to implement both unless the agent uses both sync and async execution paths (orchestrator uses async; in-process subagents may use sync).

| Hook | Signature | When It Fires | Return Value |
|------|-----------|---------------|--------------|
| `before_agent` | `(self, state, runtime) -> Any \| None` | Once when the agent is first invoked | Return a dict to modify state, `None` to pass through, or a `Command(goto=END)` to terminate |
| `abefore_agent` | `async (self, state, runtime) -> Any \| None` | Async variant of `before_agent` | Same as `before_agent` |
| `before_model` | `(self, state, runtime) -> Any \| None` | Before each LLM call in the reasoning loop | Return a dict to modify state (e.g., `{"messages": [...]}`) or `None` to pass through |
| `abefore_model` | `async (self, state, runtime) -> Any \| None` | Async variant of `before_model` | Same as `before_model`. Can return `{"jump_to": "end"}` with `@hook_config(can_jump_to=["end"])` to terminate early |
| `after_model` | `(self, state, runtime) -> Any \| None` | After each LLM call returns | Return a dict to modify state (e.g., replace messages) or `None` |
| `aafter_model` | `async (self, state, runtime) -> Any \| None` | Async variant of `after_model` | Same as `after_model` |
| `wrap_model_call` | `(self, request: ModelRequest, handler) -> ModelResponse` | Wraps the entire model invocation | Must call `handler(request)` (or a modified request) and return a `ModelResponse` |
| `awrap_model_call` | `async (self, request: ModelRequest, handler) -> ModelResponse` | Async variant of `wrap_model_call` | Must `await handler(request)` and return a `ModelResponse` |
| `wrap_tool_call` | `(self, request: ToolCallRequest, handler) -> ToolMessage \| Command` | Wraps each tool execution | Must call `handler(request)` and return the result (or a replacement) |
| `awrap_tool_call` | `async (self, request: ToolCallRequest, handler) -> ToolMessage \| Command` | Async variant of `wrap_tool_call` | Must `await handler(request)` and return the result |

### Key Types

- **`ModelRequest`** -- contains `messages` (list of `BaseMessage`), `model` (the LLM instance), and `system_message`. Use `request.override(messages=..., model=...)` to create a modified copy.
- **`ModelResponse`** -- contains `result` (list of `BaseMessage`) and `structured_response`.
- **`ToolCallRequest`** -- contains `tool_call` (dict with `name`, `args`, `id` keys).
- **`hook_config`** -- decorator for hooks that need special capabilities:

```python
@hook_config(can_jump_to=["end"])
async def abefore_model(self, state, runtime):
    """Return {"jump_to": "end"} to terminate the agent."""
    if should_stop(state):
        return {
            "messages": [AIMessage(content="Stopping.")],
            "jump_to": "end",
        }
    return None
```

---

## 3. Writing a Custom Middleware

This section walks through building a request logging middleware from scratch.

### Step 1: Create the Module

Create a Python file in your project that is importable (on `PYTHONPATH` or within the installed package). For example:

```
deep_agent/
  src/
    custom/
      __init__.py
      logging_middleware.py    <-- your middleware
```

### Step 2: Implement the Middleware Class

```python
"""Custom middleware that logs LLM request/response metadata."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)


def _model_name(request: ModelRequest[Any]) -> str:
    """Extract a human-readable model name from the request."""
    model = request.model
    if isinstance(model, str):
        return model
    return (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or "unknown"
    )


class RequestLoggingMiddleware(AgentMiddleware):
    """Log model name and message count before each LLM call,
    and token usage after each LLM call."""

    # ── Sync hooks ───────────────────────────────────────────────

    def before_model(self, state: Any, runtime: Any) -> Any:
        """Log message count before the LLM call."""
        messages = state.get("messages", []) if isinstance(state, dict) else []
        logger.info(
            "before_model: %d message(s) in conversation",
            len(messages),
        )
        return None  # pass through -- do not modify state

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Wrap the model call: log model name, measure latency, log token usage."""
        model = _model_name(request)
        logger.info(
            "wrap_model_call START: model=%s, messages=%d",
            model,
            len(request.messages),
        )

        started = time.monotonic()
        response = handler(request)
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)

        # Extract token usage from response metadata if available
        token_usage = self._extract_token_usage(response)
        logger.info(
            "wrap_model_call DONE: model=%s, latency=%.2fms, tokens=%s",
            model,
            elapsed_ms,
            token_usage,
        )
        return response

    # ── Async hooks ──────────────────────────────────────────────

    async def abefore_model(self, state: Any, runtime: Any) -> Any:
        """Async variant: log message count before the LLM call."""
        messages = state.get("messages", []) if isinstance(state, dict) else []
        logger.info(
            "abefore_model: %d message(s) in conversation",
            len(messages),
        )
        return None

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Async variant: wrap model call with logging."""
        model = _model_name(request)
        logger.info(
            "awrap_model_call START: model=%s, messages=%d",
            model,
            len(request.messages),
        )

        started = time.monotonic()
        response = await handler(request)
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)

        token_usage = self._extract_token_usage(response)
        logger.info(
            "awrap_model_call DONE: model=%s, latency=%.2fms, tokens=%s",
            model,
            elapsed_ms,
            token_usage,
        )
        return response

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _extract_token_usage(response: ModelResponse[Any]) -> dict[str, int]:
        """Pull token usage from response metadata, if the provider includes it."""
        usage: dict[str, int] = {}
        for msg in response.result:
            meta = getattr(msg, "response_metadata", {}) or {}
            token_data = meta.get("usage", meta.get("token_usage", {}))
            if isinstance(token_data, dict):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if key in token_data:
                        usage[key] = token_data[key]
                if usage:
                    return usage
        return usage
```

### Design Notes

- **Return `None` from `before_*` / `after_*` hooks** to indicate no state modification. Return a dict (e.g., `{"messages": [...]}`) to patch the agent state.
- **Always call `handler(request)` in `wrap_*` hooks.** Skipping the handler means the LLM or tool never executes. This is intentional only for blocking middleware (like OPA).
- **Implement both sync and async variants** if your middleware will be used by both the orchestrator (async) and in-process subagents (sync). If you only implement one, the other path will skip your middleware.
- **Keep middleware stateless or thread-safe.** A single middleware instance is shared across all calls in a request.

---

## 4. Registering Custom Middleware

### Via `agent.yaml`

Add your middleware's dotted import path to the `middleware.extra` list in `config/agent/runtime/agent.yaml`:

```yaml
middleware:
  # ... other middleware settings ...
  extra:
    - "deep_agent.src.custom.logging_middleware:RequestLoggingMiddleware"
```

The format is `module.path:ClassName`. The colon separates the Python module path from the class or factory function name.

### How Loading Works

The `_import_middleware()` function in `deep_agent/src/infrastructure/middleware.py` handles loading:

1. It splits the path on `:` to get the module path and attribute name
2. It calls `importlib.import_module()` to load the module
3. It gets the attribute (class or function) from the module
4. If the attribute is callable (a class or factory function), it calls it with no arguments to create the instance

```python
# What _import_middleware does internally:
module_path, attr_name = "deep_agent.src.custom.logging_middleware:RequestLoggingMiddleware".rsplit(":", 1)
module = importlib.import_module(module_path)        # imports the module
factory_or_class = getattr(module, attr_name)         # gets the class
middleware_instance = factory_or_class()              # instantiates it
```

This means your middleware class must have a no-argument constructor, or you must provide a factory function that returns a configured instance:

```yaml
# Using a factory function instead of a class
extra:
  - "deep_agent.src.custom.logging_middleware:create_logging_middleware"
```

```python
def create_logging_middleware():
    """Factory that returns a configured middleware instance."""
    return RequestLoggingMiddleware(log_level="DEBUG")
```

### Module Placement

Your module must be importable at runtime. This means either:

- Place it inside the `deep_agent/` package (recommended)
- Install it as a separate package in the same Python environment
- Ensure the module's parent directory is on `PYTHONPATH`

### Per-Agent Extras

You can also add middleware in a subagent's frontmatter `middleware:` block. The `extra` list from the agent override is appended to the global list:

```yaml
# In a subagent's frontmatter
middleware:
  extra:
    - "deep_agent.src.custom.my_subagent_middleware:MyMiddleware"
```

See the [Configuration Reference](./01-configuration-reference.md) for the full set of middleware YAML options.

---

## 5. Built-in Middleware Reference

The framework provides several middleware classes that are automatically included based on configuration.

### Always-On Middleware (Not Gated by `MIDDLEWARE_ENABLED`)

These middleware are added by `build_middleware_list()` before the `MIDDLEWARE_ENABLED` check, so they run even when the env var kill switch is off:

| Name | Module | Purpose | Hooks Used |
|------|--------|---------|------------|
| `AuditMiddleware` | `deep_agent.src.audit.middleware` | Emits platform audit events for LLM and tool operations. Classifies tool calls as `llm_call`, `mcp_tool_call`, `memory_write`, or `subagent_delegation`. | `wrap_model_call`, `awrap_model_call`, `wrap_tool_call`, `awrap_tool_call` |
| `OPAMiddleware` | `deep_agent.src.opa.middleware` | Enforces OPA authorization policy. Evaluates conversation trajectory before model calls and checks model/tool outputs against policy. Supports retry-on-deny for model calls. | `abefore_model` (with `@hook_config(can_jump_to=["end"])`), `awrap_model_call`, `awrap_tool_call` |
| `GeminiSafetyLogMiddleware` | `deep_agent.src.infrastructure.middleware` (inner class) | Detects Gemini safety filter blocks (both candidate-level and prompt-level) and replaces empty responses with a user-facing refusal message. | `after_model`, `aafter_model` |

### Guardrail Middleware (Gated by `MIDDLEWARE_ENABLED`)

These are built and appended by `_append_guardrails()` when `MIDDLEWARE_ENABLED` is true:

| Name | Config Key | Purpose | Hooks Used |
|------|------------|---------|------------|
| `ModelCallLimitMiddleware` | `middleware.model_call_limit` | Caps LLM calls per run (default: 50) | Internal (wrap) |
| `ToolCallLimitMiddleware` | `middleware.tool_call_limit` | Caps tool calls per run (default: 200) | Internal (wrap) |
| `ModelRetryMiddleware` | `middleware.model_retry` | Retries transient LLM failures with exponential backoff. Skips retrying `ContentSafetyError` exceptions. | Internal (wrap) |
| `ModelFallbackMiddleware` | `middleware.model_fallback` | Switches to a backup model on primary model failure | Internal (wrap) |
| `ToolRetryMiddleware` | `middleware.tool_retry` | Retries specific tools on failure | Internal (wrap) |
| `PIIMiddleware` | `middleware` section + `pii.enabled` | Scrubs PII from LLM inputs using a token map, restores PII in outputs. Also blocks input containing PII when strategy is `block`. | `abefore_agent`, `before_agent`, `awrap_model_call`, `wrap_model_call` |
| `ParallelPIIMiddleware` | `pii.rules` (provider: default) | Runs stock LangChain PII checks concurrently for each rule | `abefore_model`, `before_model` |
| `SummarizationToolMiddleware` | `middleware.summarization_tool` | Gives the agent a tool to trigger conversation summarization proactively | Internal (deepagents) |

### AuditMiddleware Details

`AuditMiddleware` (in `deep_agent/src/audit/middleware.py`) wraps both model and tool calls. For model calls, it emits `llm_call` events with `phase: "start"` and `phase: "complete"`, including latency and status. For tool calls, it classifies each call:

- **`subagent_delegation`** -- when the tool is `task` (orchestrator delegating)
- **`mcp_tool_call`** -- when the tool name is in the MCP tool name set
- **`memory_write`** -- when `edit_file` or `write_file` targets a `/memories/` path
- Unclassified tool calls emit no audit event

### OPAMiddleware Details

`OPAMiddleware` (in `deep_agent/src/opa/middleware.py`) uses three hooks:

1. **`abefore_model`** with `@hook_config(can_jump_to=["end"])` -- evaluates the full conversation trajectory. If OPA denies it, returns `{"messages": [...], "jump_to": "end"}` to terminate the agent.
2. **`awrap_model_call`** -- generates the LLM response silently (with streaming suppressed via `TAG_NOSTREAM`), then evaluates the output against OPA policy. On denial, retries up to `max_retries` times with feedback, then returns a hardcoded refusal.
3. **`awrap_tool_call`** -- executes the tool, then evaluates the output. On denial, returns a `Command` with an error `ToolMessage` asking the agent to retry with compliant arguments.

### PIIMiddleware Details

`PIIMiddleware` (in `deep_agent/src/pii/middleware.py`) operates at two levels:

1. **Input blocking** via `abefore_agent` / `before_agent` -- scans the last human message for PII matching rules with `strategy: block`. If found, returns a `Command(goto=END)` with a refusal message, terminating the agent before it runs.
2. **Scrub/restore** via `awrap_model_call` / `wrap_model_call` -- scrubs PII from all messages and the system prompt before they reach the LLM. After the response, restores PII tokens back to their original values using a snapshot of the token map.

---

## 6. Resolution and Exclusion

### Merge Logic

`resolve_middleware()` in `deep_agent/src/agent/config/middleware.py` follows this order:

```
global defaults (MiddlewareDefaults)
    ↓  override with
profile (ProfileConfig matched by model name)
    ↓  override with
agent frontmatter overrides (dict)
    ↓  produces
ResolvedMiddlewareConfig
```

Each boolean setting is resolved by `_resolve_bool()`:
- If the override is `None`, use the default
- If the override is a `bool`, use it directly
- If the override is a `dict`, look for an `enabled` key

The `extra` middleware list is additive -- agent-level extras are appended, not replaced:

```python
extra = list(defaults.extra)       # start with global extras
if "extra" in overrides:
    extra.extend(overrides["extra"])  # append agent-level extras
```

### Excluding Middleware per Model

Use `harness_profiles[model].excluded_middleware` in `agent.yaml` to disable specific middleware for a given model. For example, Claude models do not need `PatchToolCallsMiddleware`:

```yaml
harness_profiles:
  claude-sonnet-4:
    excluded_middleware:
      - patch_tool_calls
```

In `resolve_middleware()`, profile exclusions are applied after defaults:

```python
if "patch_tool_calls" in profile.excluded_middleware:
    patch_enabled = False
```

The `build_excluded_middleware()` function in `deep_agent/src/infrastructure/middleware.py` converts the resolved config into a list of class names to exclude from deepagents defaults.

### The `MIDDLEWARE_ENABLED` Kill Switch

The `MIDDLEWARE_ENABLED` environment variable (default: `true`) acts as a global kill switch. When set to `false`, `build_middleware_list()` returns early after adding only the always-on middleware (Audit, OPA, GeminiSafetyLog):

```python
if not settings.MIDDLEWARE_ENABLED:
    logger.info("Middleware disabled via MIDDLEWARE_ENABLED=false")
    return middlewares  # only contains audit + OPA + safety
```

This means PII scrubbing, guardrails, custom middleware, and summarization are all skipped. Audit and OPA continue to function because they are added before the check.

---

## 7. Per-Subagent Middleware

### Automatic Middleware Injection

Every subagent automatically receives `AuditMiddleware` and `OPAMiddleware` (when their respective features are enabled). This is handled by `_subagent_middleware()` in `deep_agent/src/infrastructure/subagents.py`:

```python
def _subagent_middleware(
    name: str,
    resolved_tools: list[Any],
    fallback_mw: list[Any],
) -> list[Any] | None:
    """Merge audit + OPA middleware with optional fallback middleware."""
    middleware: list[Any] = []
    audit_mw = build_audit_middleware(
        mcp_tool_names=_mcp_tool_names_from_tools(resolved_tools),
        agent=name,
    )
    if audit_mw is not None:
        middleware.append(audit_mw)
    opa_mw = build_opa_middleware()
    if opa_mw is not None:
        middleware.append(opa_mw)
    middleware.extend(fallback_mw)
    return middleware or None
```

Key points:

- **AuditMiddleware** is created with the subagent's name as the `agent` parameter, so audit events are attributed to the correct subagent.
- **OPAMiddleware** is a single shared instance -- the same policy applies to all agents.
- **ModelFallbackMiddleware** is included when the subagent's model spec has a `fallback` configured (inherited from the orchestrator if not specified).

### Frontmatter Middleware Overrides

Subagent `.md` config files can include a `middleware:` block in their YAML frontmatter to override global defaults. These overrides are passed as the `agent_overrides` parameter to `resolve_middleware()`:

```yaml
---
name: research-subagent
model: gemini-2.5-flash
description: Handles research tasks
middleware:
  memory:
    enabled: false
  extra:
    - "deep_agent.src.custom.research_logger:ResearchLogger"
---

You are a research assistant...
```

The override follows the same merge logic described in Section 6 -- booleans can be set directly or via `{"enabled": ...}` dicts, and `extra` is appended to the global list.

---

## 8. Testing Custom Middleware

### Unit Test Pattern

The recommended approach is to instantiate your middleware directly and call its hooks with mock state and runtime objects. No agent framework setup is required.

```python
"""Tests for RequestLoggingMiddleware."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from deep_agent.src.custom.logging_middleware import RequestLoggingMiddleware


class TestRequestLoggingMiddleware:
    """Unit tests for the logging middleware hooks."""

    def test_before_model_returns_none(self):
        """before_model should not modify state."""
        mw = RequestLoggingMiddleware()
        state = {"messages": [MagicMock(), MagicMock()]}
        runtime = MagicMock()

        result = mw.before_model(state, runtime)

        assert result is None

    def test_wrap_model_call_passes_through(self):
        """wrap_model_call should call handler and return its result."""
        mw = RequestLoggingMiddleware()
        request = MagicMock()
        request.messages = [MagicMock(), MagicMock()]
        request.model = "gemini-2.5-pro"

        expected_response = MagicMock()
        expected_response.result = []
        handler = MagicMock(return_value=expected_response)

        result = mw.wrap_model_call(request, handler)

        handler.assert_called_once_with(request)
        assert result is expected_response

    @pytest.mark.asyncio
    async def test_awrap_model_call_passes_through(self):
        """Async wrap should await handler and return its result."""
        mw = RequestLoggingMiddleware()
        request = MagicMock()
        request.messages = [MagicMock()]
        request.model = "claude-sonnet-4"

        expected_response = MagicMock()
        expected_response.result = []
        handler = AsyncMock(return_value=expected_response)

        result = await mw.awrap_model_call(request, handler)

        handler.assert_awaited_once_with(request)
        assert result is expected_response
```

### Testing Middleware Configuration Resolution

For testing how your middleware integrates with the configuration system, follow the patterns in `tests/unit/config/test_middleware_config.py`:

```python
from deep_agent.src.agent.config.middleware import (
    MiddlewareDefaults,
    MiddlewareFileConfig,
    resolve_middleware,
)


def test_custom_extra_middleware_appears_in_resolved():
    """Verify extra middleware paths survive the resolution process."""
    config = MiddlewareFileConfig(
        defaults=MiddlewareDefaults(
            extra=["deep_agent.src.custom.logging_middleware:RequestLoggingMiddleware"]
        )
    )
    resolved = resolve_middleware(config, "gemini-2.5-pro")

    assert "deep_agent.src.custom.logging_middleware:RequestLoggingMiddleware" in resolved.extra_middleware


def test_agent_override_appends_extra():
    """Agent-level extras should be appended to global extras."""
    config = MiddlewareFileConfig(
        defaults=MiddlewareDefaults(extra=["module_a:ClassA"])
    )
    overrides = {"extra": ["module_b:ClassB"]}
    resolved = resolve_middleware(config, "model", overrides)

    assert resolved.extra_middleware == ["module_a:ClassA", "module_b:ClassB"]
```

### Integration Test Tips

- Use `build_middleware_list()` from `deep_agent.src.infrastructure.middleware` to verify your middleware is instantiated correctly from config.
- Mock `settings.MIDDLEWARE_ENABLED = True` to ensure your middleware is not skipped by the kill switch.
- For middleware that modifies state (e.g., PII scrubbing), assert on the transformed messages rather than the return value of the hook.

---

## Quick Reference

| Task | Where to Look |
|------|---------------|
| Add custom middleware | `middleware.extra` in `config/agent/runtime/agent.yaml` |
| Disable middleware per model | `harness_profiles.<model>.excluded_middleware` in `agent.yaml` |
| Disable all optional middleware | Set `MIDDLEWARE_ENABLED=false` env var |
| Middleware resolution logic | `deep_agent/src/agent/config/middleware.py` |
| Middleware instantiation | `deep_agent/src/infrastructure/middleware.py` |
| Subagent middleware injection | `deep_agent/src/infrastructure/subagents.py` |
| Audit event classification | `deep_agent/src/audit/middleware.py` |
| OPA policy enforcement | `deep_agent/src/opa/middleware.py` |
| PII scrubbing | `deep_agent/src/pii/middleware.py` |
| Working examples | [Working Examples](./06-working-examples.md) |
| YAML config options | [Configuration Reference](./01-configuration-reference.md) |
