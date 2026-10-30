# Middleware Hooks Reference

All hooks are defined on the `AgentMiddleware` base class from `langchain.agents.middleware.types`.

## Import

```python
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import ToolMessage
from langgraph.types import Command
```

## Hook Summary

| Hook | Async Variant | Signature | When It Fires | Return |
|------|---------------|-----------|---------------|--------|
| `before_model` | `abefore_model` | `(self, state, runtime)` | Before each LLM call | `None` (no change), `dict` (state update), or `Command` |
| `after_model` | `aafter_model` | `(self, state, runtime)` | After each LLM call | `None` (no change), `dict` (state update), or `Command` |
| `wrap_model_call` | `awrap_model_call` | `(self, request: ModelRequest, handler)` | Wraps the entire model call | `ModelResponse` |
| `wrap_tool_call` | `awrap_tool_call` | `(self, request: ToolCallRequest, handler)` | Wraps tool execution | `ToolMessage` or `Command` |
| `before_agent` | `abefore_agent` | `(self, state, runtime)` | At agent invocation start | `None` (no change), `dict` (state update), or `Command` |

## Hook Details

### `before_model` / `abefore_model`

Fires before each LLM call. Use it to inspect or modify the conversation state before the model sees it.

```python
async def abefore_model(self, state: dict, runtime: Runtime) -> dict | None:
    messages = state.get("messages", [])
    # Inspect or modify messages
    # Return None to pass through unchanged
    # Return a dict to merge into state (e.g., {"messages": modified_messages})
    # Return Command to jump to another node
    return None
```

**Early termination:** Decorate with `@hook_config(can_jump_to=["end"])` and return `{"messages": [...], "jump_to": "end"}` to stop the agent.

### `after_model` / `aafter_model`

Fires after each LLM call. Use it to inspect or modify the model's response in the state.

```python
async def aafter_model(self, state: dict, runtime: Runtime) -> dict | None:
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    # Inspect the model's response
    # Return None to pass through unchanged
    # Return a dict to merge into state
    return None
```

### `wrap_model_call` / `awrap_model_call`

Wraps the entire model invocation. You get the request and a handler to call (or skip) the model. This is the most powerful hook for model calls.

```python
async def awrap_model_call(
    self, request: ModelRequest, handler
) -> ModelResponse:
    # Pre-processing
    # request.messages — the conversation messages
    # request.model — the LLM instance
    # request.system_message — the system prompt (if any)

    response = await handler(request)  # Call the model

    # Post-processing
    # response.result — list of BaseMessage from the model
    return response
```

**Key `ModelRequest` attributes:**
- `messages` — list of conversation messages
- `model` — the chat model instance
- `system_message` — optional system message
- `override(**kwargs)` — create a copy with modified fields

**Key `ModelResponse` attributes:**
- `result` — list of BaseMessage (the model's output)
- `structured_response` — optional structured output

### `wrap_tool_call` / `awrap_tool_call`

Wraps individual tool executions. You get the tool call request and a handler.

```python
async def awrap_tool_call(
    self, request: ToolCallRequest, handler
) -> ToolMessage | Command:
    tool_call = request.tool_call
    tool_name = tool_call.get("name", "unknown")
    tool_args = tool_call.get("args", {})
    tool_id = tool_call.get("id")

    result = await handler(request)  # Execute the tool

    # Post-processing — result is a ToolMessage or Command
    return result
```

**Key `ToolCallRequest` attributes:**
- `tool_call` — dict with `name`, `args`, `id`

### `before_agent` / `abefore_agent`

Fires once at the start of the agent invocation. Use it for initialization or input validation.

```python
async def abefore_agent(self, state: dict, runtime: Runtime) -> dict | None:
    # One-time setup or input validation
    # Return None to pass through unchanged
    return None
```

## The `@hook_config` Decorator

Use `@hook_config(can_jump_to=["end"])` to declare that a hook can terminate the agent early by returning a jump target.

```python
from langchain.agents.middleware.types import AgentMiddleware, hook_config

class BlockingMiddleware(AgentMiddleware):

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        if self._should_block(state):
            return {
                "messages": [AIMessage(content="Blocked by policy.")],
                "jump_to": "end",
            }
        return None
```
