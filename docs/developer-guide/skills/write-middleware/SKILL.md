---
name: write-middleware
description: >
  Guides creation of custom agent middleware. Use when a user asks to
  create, write, or add a new middleware to the agent pipeline.
---

# Write Middleware

Create correctly structured, production-ready agent middleware that plugs into the deepagents pipeline.

## When to Use

When a user asks to create, write, or add a custom middleware to the agent. Examples:

- "Create a middleware that logs all tool calls"
- "Add a middleware to filter out certain messages"
- "Write a rate-limiting middleware"

## Step-by-Step Checklist

### 1. Determine Which Hooks Are Needed

Ask the user what the middleware should do, then choose the right hooks:

| Goal | Hook(s) |
|------|---------|
| Inspect or modify messages before the LLM sees them | `before_model` |
| Inspect or modify the LLM response after it returns | `after_model` |
| Wrap the entire LLM call (pre + post + error handling) | `wrap_model_call` |
| Intercept or modify tool execution | `wrap_tool_call` |
| Run logic once at the start of agent invocation | `before_agent` |

See `references/hooks.md` for full signatures and return types.

### 2. Choose Sync vs Async

Prefer async hooks (`awrap_model_call`, `aafter_model`, etc.) because the agent runs asynchronously. Implement sync variants only if the middleware must also support synchronous subagent invocations.

### 3. Subclass `AgentMiddleware`

```python
from langchain.agents.middleware.types import AgentMiddleware

class MyMiddleware(AgentMiddleware):
    ...
```

The constructor **must accept zero arguments** because `_import_middleware()` calls the class with no arguments. If the middleware needs configuration, read it from environment variables or settings inside `__init__`.

### 4. Implement the Hooks

Use `assets/middleware_template.py` as a starting point. Only implement the hooks you need.

### 5. Create the Python File

Place the file in a location importable by the agent runtime. Recommended locations:

- `deep_agent/src/middleware/my_middleware.py` for project-level middleware
- A separate importable package if the middleware is reusable

### 6. Register in `agent.yaml`

Add the dotted path to `config/agent/runtime/agent.yaml` under `middleware.extra`:

```yaml
middleware:
  extra:
    - "deep_agent.src.middleware.my_middleware:MyMiddleware"
```

The format is `module.path:ClassName`. The colon separates the importable module from the class/factory name.

### 7. Test the Middleware

- Verify the middleware loads without import errors
- Test each hook fires at the correct point in the pipeline
- Confirm the middleware does not break existing pipeline behavior

## Resources

- **Hook signatures and return types:** `references/hooks.md`
- **Pipeline execution order:** `references/execution_order.md`
- **Built-in middleware reference:** `references/built_in_middleware.md`
- **Real-world patterns:** `references/patterns.md`
- **Runtime context reference:** `references/runtime_context.md`
- **Basic template:** `assets/middleware_template.py`
- **Production template:** `assets/advanced_middleware_template.py`

## Patterns

Before writing middleware, review real-world patterns from existing implementations:

- **Patterns Reference:** `references/patterns.md` — wrap-and-inspect, classify-and-route, evaluate-retry, token replacement, input blocking, early termination, safety-blocked response detection
- **Runtime Context:** `references/runtime_context.md` — what data is available in each hook (`ModelRequest`, `ToolCallRequest`, state dict, config)
- **Advanced Template:** `assets/advanced_middleware_template.py` — production-ready template with logging, timing, error handling, and both sync/async variants

## Critical Requirements

- **Must subclass `AgentMiddleware`** from `langchain.agents.middleware.types`
- **Must use `module.path:ClassName` format** for registration in `middleware.extra`
- **Constructor must accept zero arguments** -- `_import_middleware()` calls the class/factory with no args
- **For early termination**, use `@hook_config(can_jump_to=["end"])` decorator on the hook method and return `{"messages": [...], "jump_to": "end"}`
- **Prefer async hooks** (`awrap_model_call`, `aafter_model`, etc.) over sync variants
- **Do not duplicate** functionality already handled by built-in middleware (see `references/built_in_middleware.md`)
