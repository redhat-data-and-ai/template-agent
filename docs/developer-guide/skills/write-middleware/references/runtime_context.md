# Runtime Context Reference

What data is available inside each middleware hook, verified from the actual codebase implementations.

---

## `wrap_model_call` / `awrap_model_call`

These hooks receive a `ModelRequest` and a `handler` callable.

### `request: ModelRequest`

| Attribute | Type | Description | Used In |
|-----------|------|-------------|---------|
| `request.messages` | `list[AnyMessage]` | Conversation messages (HumanMessage, AIMessage, ToolMessage, etc.) | audit, pii, opa |
| `request.model` | `str` or model instance | The chat model -- may be a string name or an instantiated model object | audit, opa |
| `request.system_message` | `SystemMessage \| None` | The system prompt, if configured | pii |
| `request.config` | `RunnableConfig` | LangGraph runtime config (see below) | -- |
| `request.override(**kwargs)` | method | Create a copy of the request with modified fields | pii, opa |

### Extracting Model Name

The model attribute can be a string or an object. Use this pattern from `AuditMiddleware`:

```python
def _model_name(request: ModelRequest) -> str:
    model = request.model
    if isinstance(model, str):
        return model
    return (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or "unknown"
    )
```

### `handler` Return: `ModelResponse`

| Attribute | Type | Description |
|-----------|------|-------------|
| `response.result` | `list[BaseMessage]` | The model's output messages (typically `AIMessage` instances) |
| `response.structured_response` | `Any \| None` | Optional structured output (when using structured output mode) |

To construct a new response:

```python
from langchain.agents.middleware.types import ModelResponse

return ModelResponse(
    result=[AIMessage(content="replacement text", id=original.id)],
    structured_response=original_response.structured_response,
)
```

---

## `wrap_tool_call` / `awrap_tool_call`

These hooks receive a `ToolCallRequest` and a `handler` callable.

### `request: ToolCallRequest`

| Attribute | Type | Description | Used In |
|-----------|------|-------------|---------|
| `request.tool_call` | `dict` | Tool call dict with `name`, `args`, `id`, `type` | audit, opa |
| `request.config` | `RunnableConfig` | LangGraph runtime config | -- |

### Accessing Tool Call Fields

```python
tool_call = request.tool_call
tool_name = tool_call.get("name", "unknown")   # str: the tool function name
tool_args = tool_call.get("args", {})           # dict: arguments passed to the tool
tool_id = tool_call.get("id")                   # str: unique call ID for correlation
```

### `handler` Return: `ToolMessage | Command`

The handler returns either a `ToolMessage` (normal result) or a `Command` (graph redirect).

**ToolMessage attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `.content` | `str \| list` | The tool's output content |
| `.tool_call_id` | `str` | Correlates back to the original tool call |
| `.name` | `str` | Tool name |
| `.status` | `str` | `"success"` or `"error"` |

**Returning a `Command` instead of `ToolMessage`:**

```python
from langgraph.types import Command

return Command(
    update={
        "messages": [
            ToolMessage(
                content="Tool call blocked by policy.",
                tool_call_id=tool_id,
                name=tool_name,
                status="error",
            ),
        ],
    },
)
```

---

## `before_model` / `abefore_model`

These hooks receive the agent state and runtime, firing before each LLM call.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `dict[str, Any]` | Current agent state containing `messages` list and other state keys |
| `runtime` | `Runtime` | LangGraph runtime info |

### Accessing Messages

```python
messages = state.get("messages") or []
# messages is a list of BaseMessage instances (HumanMessage, AIMessage, ToolMessage, etc.)
```

### Return Values

| Return | Effect |
|--------|--------|
| `None` | No change -- continue normally |
| `dict` | Merge into state (e.g., `{"messages": modified_messages}`) |
| `{"messages": [...], "jump_to": "end"}` | Terminate the graph (requires `@hook_config(can_jump_to=["end"])`) |

---

## `after_model` / `aafter_model`

Same signature as `before_model`, but fires after the model response has been added to state.

### Inspecting the Model Response

```python
def after_model(self, state, runtime):
    msgs = state.get("messages", [])
    last = msgs[-1] if msgs else None

    if isinstance(last, AIMessage):
        # Check content
        if not last.content and not last.tool_calls:
            # Empty response -- might be a safety block
            pass

        # Check response metadata (provider-specific)
        meta = getattr(last, "response_metadata", {}) or {}
        finish_reason = meta.get("finish_reason", "")
```

### Replacing the Last Message

```python
msgs[-1] = AIMessage(content="Replacement text", id=last.id)
return {"messages": msgs}
```

---

## `before_agent` / `abefore_agent`

Fires once at the start of the agent invocation. Same parameters as `before_model`.

### Input Validation Pattern

```python
async def abefore_agent(self, state, runtime):
    messages = state.get("messages", []) if isinstance(state, dict) else []

    # Find the last human message
    human_msg = next(
        (m for m in reversed(messages)
         if getattr(m, "type", None) in ("human", "user")),
        None,
    )
    if not human_msg:
        return None

    # Validate and optionally block
    if self._is_blocked(human_msg.content):
        return Command(
            update={"messages": [AIMessage(content="Request blocked.")]},
            goto=END,
        )
    return None
```

---

## Accessing Thread ID and User ID

Thread and user identifiers are stored in the LangGraph config's `configurable` dict, not directly on the request object.

### From within `wrap_model_call` / `wrap_tool_call`

Use `langgraph.config.get_config()` to access the current config:

```python
from langgraph.config import get_config

config = get_config()
thread_id = (config or {}).get("configurable", {}).get("thread_id")
user_id = (config or {}).get("configurable", {}).get("user_id")
```

**Important:** Wrap `get_config()` in a try/except -- it relies on a ContextVar that may not be set in all execution contexts:

```python
def _get_thread_id(self) -> str | None:
    try:
        from langgraph.config import get_config
        config = get_config()
        return (config or {}).get("configurable", {}).get("thread_id")
    except Exception:
        return None
```
