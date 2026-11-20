# Real-World Middleware Patterns

Production patterns extracted from the codebase middleware implementations. Each pattern shows the technique with a focused snippet -- use these as building blocks for your own middleware.

---

## Pattern 1: Accessing Runtime Context

Every middleware hook receives a `request` object with the conversation state and config. Here is how to extract the data you need.

**Source:** `deep_agent/src/audit/middleware.py`

```python
def _model_name(request: ModelRequest) -> str:
    """Extract the model name from the request."""
    model = request.model
    if isinstance(model, str):
        return model
    return (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or "unknown"
    )

# Inside a wrap_model_call hook:
model = _model_name(request)
msg_count = len(request.messages)
```

**Thread ID** is accessed from LangGraph's config context (not directly from the request):

**Source:** `deep_agent/src/pii/middleware.py`

```python
from langgraph.config import get_config

config = get_config()
thread_id = (config or {}).get("configurable", {}).get("thread_id")
```

**Agent identity** is typically passed via the constructor and stored on `self`:

```python
class AuditMiddleware(AgentMiddleware):
    def __init__(self, *, agent: str | None = None) -> None:
        self._agent = agent or "orchestrator"

    def _base_details(self) -> dict[str, Any]:
        return {"agent": self._agent}
```

---

## Pattern 2: Wrap-and-Inspect (Most Common Pattern)

The most common middleware pattern: call the handler to get the response, then inspect, log, or modify it before returning. Includes timing instrumentation.

**Source:** `deep_agent/src/audit/middleware.py`

```python
async def awrap_model_call(self, request, handler) -> ModelResponse:
    model = _model_name(request)
    started = time.monotonic()

    # Emit a "start" event
    self._emit_llm_phase(phase="start", model=model, message_count=len(request.messages))

    try:
        response = await handler(request)
    except Exception as exc:
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        self._emit_llm_phase(phase="complete", model=model, status="error",
                             latency_ms=elapsed_ms, error=str(exc))
        raise  # Always re-raise -- don't swallow errors

    elapsed_ms = round((time.monotonic() - started) * 1000, 2)
    self._emit_llm_phase(phase="complete", model=model, status="success",
                         latency_ms=elapsed_ms)
    return response
```

**Key takeaways:**
- Always call `handler(request)` (or `await handler(request)` for async) to execute the next middleware/model
- Wrap in `try/except` to log errors but always re-raise
- Use `time.monotonic()` for latency measurement (not `time.time()`)
- Return the response unmodified when you only need to observe

The same pattern applies to tool calls:

```python
async def awrap_tool_call(self, request, handler) -> ToolMessage | Command:
    tool_call = request.tool_call
    tool_name = tool_call.get("name", "unknown")
    tool_args = tool_call.get("args", {})

    started = time.monotonic()
    try:
        result = await handler(request)
        status = "success"
    except Exception as exc:
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        self._emit_event(tool_name=tool_name, status="error", latency_ms=elapsed_ms)
        raise

    elapsed_ms = round((time.monotonic() - started) * 1000, 2)
    self._emit_event(tool_name=tool_name, status=status, latency_ms=elapsed_ms)
    return result
```

---

## Pattern 3: Classify and Route

Inspect the tool call name and arguments to classify the operation, then take different actions per category.

**Source:** `deep_agent/src/audit/middleware.py`

```python
_MEMORY_TOOLS = frozenset({"edit_file", "write_file"})
_SUBAGENT_TOOL = "task"

def classify_tool_call(
    tool_name: str,
    args: dict[str, Any],
    *,
    mcp_tool_names: frozenset[str],
) -> str:
    """Classify a tool call using orchestrator/subagent parity rules."""
    if tool_name == _SUBAGENT_TOOL:
        return "subagent_delegation"
    if tool_name in mcp_tool_names:
        return "mcp_tool_call"
    if tool_name in _MEMORY_TOOLS and "memories" in _tool_path(args):
        return "memory_write"
    return ""  # Unclassified -- skip auditing
```

**Usage inside `awrap_tool_call`:**

```python
tool_call = request.tool_call
tool_name = tool_call.get("name", "unknown")
tool_args = tool_call.get("args", {})

audit_type = classify_tool_call(tool_name, tool_args, mcp_tool_names=self._mcp_tool_names)
if not audit_type:
    return await handler(request)  # Skip -- not a classified operation

# Proceed with audit logging for classified operations
```

**When to use:** Building middleware that only acts on specific tool types (e.g., logging MCP calls differently from built-in tools, or applying special handling to memory writes).

---

## Pattern 4: Evaluate-Retry Loop

Generate a response silently, evaluate it against an external policy, and retry if blocked. After exhausting retries, return a denial.

**Source:** `deep_agent/src/opa/middleware.py`

```python
async def awrap_model_call(self, request, handler) -> ModelResponse:
    max_retries = get_opa_max_retries()
    current_request = request

    for attempt in range(max_retries + 1):
        # 1. Generate response (silently -- no streaming to client)
        result = await handler(
            current_request.override(model=_NoStreamModel(current_request.model))
        )
        text = self._extract_text(result)

        if not text.strip():
            return result  # Empty response (tool-call-only) -- pass through

        # 2. Evaluate against external policy
        opa = await evaluate_message("llm_response", agent_message=text)
        if opa.allowed:
            return result  # Policy allows it -- return normally

        # 3. If blocked and retries remain, inject feedback and retry
        if attempt < max_retries:
            denial_summary = "; ".join(opa.denial_reasons) or "policy violation"
            feedback = HumanMessage(
                content=f"Your response was blocked: {denial_summary}. Please revise.",
                additional_kwargs={"opa_retry": True},
            )
            current_request = current_request.override(
                messages=current_request.messages + [feedback]
            )

    # 4. All retries exhausted -- return a safe denial
    return ModelResponse(
        result=[AIMessage(content="I'm unable to respond to that request.")]
    )
```

**Key takeaways:**
- Use `request.override(model=_NoStreamModel(...))` to suppress streaming during silent evaluation
- Use `request.override(messages=...)` to inject feedback for retry attempts
- Mark retry messages with `additional_kwargs` so downstream middleware can filter them
- Always have a final fallback response after retries are exhausted

---

## Pattern 5: Early Termination with `hook_config`

Use the `@hook_config(can_jump_to=["end"])` decorator on `before_model` hooks to terminate the agent graph before the LLM is invoked.

**Source:** `deep_agent/src/opa/middleware.py`

```python
from langchain.agents.middleware.types import AgentMiddleware, hook_config

class OPAMiddleware(AgentMiddleware):

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        """Evaluate conversation trajectory before each model call."""
        messages = state.get("messages") or []
        if not messages:
            return None  # Nothing to evaluate -- pass through

        # Evaluate the full trajectory
        trajectory = [
            m for m in messages
            if isinstance(m, BaseMessage) and not m.additional_kwargs.get("opa_retry")
        ]
        opa = await evaluate_trajectory(trajectory)

        if opa.allowed:
            return None  # Pass through

        # Block: inject a denial message and jump to end
        return {
            "messages": [AIMessage(content="This thread has been terminated due to policy violation.")],
            "jump_to": "end",
        }
```

**When to use:** Blocking dangerous requests before they reach the LLM (e.g., policy violations, safety checks, rate limiting). This is the only way to prevent the model from being called at all.

**Important:**
- The `@hook_config(can_jump_to=["end"])` decorator is required -- without it, returning `"jump_to": "end"` has no effect
- Returning `None` means "no change, continue normally"
- Returning a dict with `"jump_to": "end"` terminates the graph and sends the messages to the user

---

## Pattern 6: Token Replacement and Restoration (PII Scrub/Restore)

Transform input messages before the LLM sees them, then reverse the transformation in the output. The key challenge is managing the token map across the async boundary.

**Source:** `deep_agent/src/pii/middleware.py`

```python
async def awrap_model_call(self, request, handler) -> ModelResponse:
    from langchain.agents.middleware.types import ModelResponse

    # 1. Set up thread-aware scrubbing state
    thread_id = self._setup_scrub()  # Loads per-thread token map

    # 2. Scrub PII from all input messages and system prompt
    scrubbed_messages = [self._scrub_message(m) for m in request.messages]
    scrubbed_system = self._scrub_system(request.system_message)

    # 3. Snapshot the token map NOW -- before the async handler crosses
    #    any context boundary. The token map is a plain dict (always available),
    #    not a ContextVar (which may be empty after await).
    token_map = self._scrubber.snapshot_token_map()
    self._teardown_scrub(thread_id)  # Persist and snapshot for SSE

    # 4. Call the handler with scrubbed input
    scrubbed_request = request.override(
        messages=scrubbed_messages,
        system_message=scrubbed_system,
    )
    response = await handler(scrubbed_request)

    # 5. Restore PII in the output using the snapshotted token map
    restored_result = [
        self._restore_message_with_map(m, token_map) for m in response.result
    ]
    return ModelResponse(result=restored_result, structured_response=response.structured_response)
```

**Key takeaways:**
- Snapshot state **before** `await handler(...)` -- ContextVars can be lost across async boundaries
- Use `request.override(messages=..., system_message=...)` to pass scrubbed input
- Build a new `ModelResponse` with restored output
- Keep the token map as a plain `dict` local variable, not a ContextVar, for reliability
- Thread-aware persistence: load per-thread maps at start, save after scrubbing

**Restoration helper pattern:**

```python
@staticmethod
def _restore_str(text: str, token_map: dict[str, str]) -> str:
    """Replace all tokens with their original values."""
    for token, value in token_map.items():
        if token in text:
            text = text.replace(token, value)
    return text
```

---

## Pattern 7: Input Blocking

Validate user input at the start of the agent invocation and block if it violates policy. Uses `before_agent` / `abefore_agent` to intercept before the agent loop begins.

**Source:** `deep_agent/src/pii/middleware.py`

```python
from langchain_core.messages import AIMessage
from langgraph.constants import END
from langgraph.types import Command

class PIIMiddleware(AgentMiddleware):

    async def abefore_agent(self, state, runtime):
        """Block requests containing PII with strategy: block."""
        return self._check_input_blocked(state)

    def _check_input_blocked(self, state):
        # 1. Extract the last human message
        messages = state.get("messages", []) if isinstance(state, dict) else []
        human_msg = next(
            (m for m in reversed(messages)
             if getattr(m, "type", None) in ("human", "user")),
            None,
        )
        if not human_msg:
            return None

        # 2. Extract text content (handles str and content-block lists)
        content = getattr(human_msg, "content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )

        # 3. Check for policy violations
        matches = self._scrubber._block_detector.find_all(content)
        if not matches:
            return None

        # 4. Block: inject a refusal and terminate
        labels = ", ".join(sorted({m.label for m in matches}))
        reply = f"I'm unable to process this request as it contains sensitive information ({labels})."
        return Command(
            update={"messages": [AIMessage(content=reply)]},
            goto=END,
        )
```

**Key takeaways:**
- Use `abefore_agent` (not `abefore_model`) to block before the entire agent loop
- Return `Command(update={...}, goto=END)` to inject a message and stop
- Always implement both sync (`before_agent`) and async (`abefore_agent`) variants if the middleware must support both invocation paths
- Handle both string and content-block-list message formats

---

## Pattern 8: Safety-Blocked Response Detection

Post-process the model output in `after_model` to detect edge cases like empty responses caused by safety filters, and replace them with user-facing messages.

**Source:** `deep_agent/src/infrastructure/middleware.py`

```python
_SAFETY_FINISH_REASONS = frozenset({"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"})
_PROMPT_BLOCK_REASONS = frozenset({"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "JAILBREAK"})
_REFUSAL = "I'm unable to respond to that request as it was flagged by the content safety filter."

class GeminiSafetyLogMiddleware(AgentMiddleware):

    def after_model(self, state, runtime):
        return self._check_and_replace(state)

    async def aafter_model(self, state, runtime):
        return self._check_and_replace(state)

    def _check_and_replace(self, state):
        msgs = state.get("messages", []) if isinstance(state, dict) else []
        if not msgs:
            return None

        last = msgs[-1]
        if not isinstance(last, AIMessage):
            return None

        # Only act on empty responses (no content and no tool calls)
        if last.content or last.tool_calls:
            return None

        meta = getattr(last, "response_metadata", {}) or {}

        # Check candidate-level safety block
        reason = meta.get("finish_reason", meta.get("stop_reason", ""))
        if reason in _SAFETY_FINISH_REASONS:
            logger.warning("Gemini safety filter blocked response: finish_reason=%s", reason)
            msgs[-1] = AIMessage(content=_REFUSAL, id=last.id)
            return {"messages": msgs}

        # Check prompt-level safety block
        prompt_feedback = meta.get("prompt_feedback") or {}
        block_reason = prompt_feedback.get("block_reason", "")
        if block_reason in _PROMPT_BLOCK_REASONS:
            logger.warning("Gemini prompt-level block: block_reason=%s", block_reason)
            msgs[-1] = AIMessage(content=_REFUSAL, id=last.id)
            return {"messages": msgs}

        return None
```

**Key takeaways:**
- Use `after_model` / `aafter_model` to inspect state after the model response has been added
- Check `response_metadata` on the last `AIMessage` for provider-specific signals
- Replace the message in the state by mutating `msgs[-1]` and returning `{"messages": msgs}`
- Implement both sync and async variants -- `after_model` and `aafter_model` -- with a shared helper method
- Return `None` when no action is needed (pass through)
