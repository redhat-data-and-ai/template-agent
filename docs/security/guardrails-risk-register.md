# Guardrails Security Risk Register

## What is in place

### Granite Guardian (IBM) — LangChain callback
Registered process-globally via `register_configure_hook`. Fires on every LangChain
LLM call in the process, including in-process subagents.

| Hook | What it checks | Action |
|---|---|---|
| `on_chat_model_start` | User input (last human message) | Blocks if unsafe |
| `on_tool_start` | Tool call arguments outbound | Blocks if unsafe |
| `on_tool_end` | Tool result before it enters LLM context | Blocks if unsafe |
| `on_llm_end` | LLM response | Logs always; blocks if `GUARDIAN_BLOCK_OUTPUT=true` |
| `on_tool_error` | Tool execution errors | Logs only |

### Memory write protection
`PersonalizationRepository.create_memory()` and `upsert_rule()` run a Guardian
check before writing to Postgres. Unsafe content is rejected and never stored.

### MCP tool abuse visibility
Every tool call is audit-logged with tool name, sanitized args (sensitive keys
redacted), and output preview. Circuit breaker trips after 5 MCP server failures.
`server_names` filtering ensures agents only connect to their declared MCP servers.

### CI / supply chain
- Trivy scans container image before push to GHCR; CRITICAL severity fails the build.
- Gitleaks runs in pre-commit on every local commit.
- GitHub Secret Scanning + Push Protection active at platform level.

---

## Remaining risks

### 1. AsyncSubAgent (remote pod) — Medium
**Threat:** Orchestrator delegates to a remote subagent pod via HTTP. Guardian
callback does not cross process boundaries.

**Mitigation in place:** `on_tool_start` Guardian check scans the outgoing
instruction payload before dispatch.

**Residual gap:** The remote pod's own LLM calls and tool calls are only
protected if that pod is deployed with `GUARDIAN_ENABLED=true`.

**Required action:** Enforce `GUARDIAN_ENABLED=true` + Guardian env vars on
every deployed agent pod. Treat this as a deployment standard, not optional.

---

### 2. Per-user tool access control — High
**Threat:** All users of the same agent share identical tool access. A low-privilege
user can invoke any tool the agent has configured.

**Mitigation in place:** OAuth/DCR enforcement at the MCP connection layer.
`server_names` filtering scopes tools to what the agent config declares.

**Residual gap:** No RBAC layer between user identity and tool invocation.

**Required action:** Add a user-role-to-tool mapping in `graph.py` between
`get_current_user()` and `get_mcp_tools()` — filter allowed tools by user role
before passing to `create_deep_agent()`.

---

### 3. Legacy memory records — Low
**Threat:** Memories written to Postgres before the Guardian write-guard was
added are not checked and may contain injection payloads.

**Mitigation in place:** New writes are blocked. Guardian `on_llm_end` flags
any exfiltration in output.

**Required action:** Run a one-off migration job:
```python
for memory in repo.list_all_memories():
    is_safe, _ = await check_safety(memory.content, context="memory")
    if not is_safe:
        await repo.delete_memory(memory.user_id, memory.id)
```

---

### 4. Subagent system prompt not validated at config load — Low
**Threat:** A tampered subagent config file could inject instructions into a
subagent's system prompt at startup.

**Mitigation in place:** Subagents loaded from `config/subagents/*.md` which
are version-controlled. Guardian fires on all in-process subagent LLM calls.

**Residual gap:** No runtime validation of subagent system prompt content at
config load time.

**Required action:** Add a startup Guardian check over all subagent `body` fields
in `load_subagents()` before building subagent instances.

---

### 5. Encoded / obfuscated prompt injection — Low
**Threat:** Base64-encoded, Unicode-homoglyph, or multi-step injection chains
that Guardian's model does not classify as unsafe.

**Mitigation in place:** Guardian checks all text entry points. Multi-layer
checks (input, tool args, tool output, LLM output) reduce the attack surface.

**Residual gap:** Model-level limitation — no classifier is perfect. No
additional decoding / normalization pass before Guardian.

**Required action:** Add a pre-check normalization step (decode base64, strip
unicode overrides) before passing content to `check_safety()`.
