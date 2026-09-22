# Guardrails Security Risk Register

## What is in place

### Granite Guardian (IBM) — LangChain callback
Registered process-globally via `register_configure_hook`. Fires on every LangChain
LLM call in the process, including in-process subagents.

| Hook | What it checks | Action |
|---|---|---|
| `on_chat_model_start` | Last human message — safety + injection | Blocks if unsafe (raises `InputContentSafetyError`) |
| `on_tool_start` | Tool call arguments (sensitive keys redacted) | Blocks if unsafe (raises `ToolContentSafetyError`) |
| `on_tool_end` | Tool result | Audit log only — result safety is handled by `GuardianToolProxy` |
| `on_llm_end` | LLM response | Logs always; always blocks unsafe output |
| `on_tool_error` | Tool execution errors | Logs only |

Content deduplication: the handler SHA-256 hashes each scanned string and skips
re-scanning the same human message on repeat LLM rounds within an agentic loop.

### GuardianToolProxy — per-tool async proxy
`wrap_tools()` in `deep_agent/src/guardrails/tool_proxy.py` replaces every tool
in the agent's tool list with a `GuardianToolProxy` when `GUARDIAN_API_BASE` is set.
The proxy runs three phases on every `ainvoke`:

1. **Pre-check args** — `check_safety(arg_text[:500])` before the inner tool executes.
   Unsafe: returns a `BLOCKED_INPUT` `ToolMessage`; inner tool is never called.
2. **Execute inner tool** — exceptions are caught and returned as `ToolMessage(status="error")`
   so other tools in a parallel batch are unaffected.
3. **Post-check result** — `check_safety` then `check_injection` on the result
   (first 500 chars). Unsafe: replaces the `ToolMessage`/`Command` content with
   `BLOCKED_RESULT` and signals `_safety_ctx["blocked"] = True`.

### SafetyAwareRunnable — agentic-loop circuit breaker
`deep_agent/aegra/safety.py` wraps the compiled graph (`outermost=True`) and every
in-process subagent runnable (`outermost=False`).

- Injects a shared `_safety_ctx` dict into LangGraph config so proxies can signal
  blocks back to the runnable.
- **`astream_events`**: buffers AI output chunks; monitors `on_tool_end` events for
  the `BLOCKED_RESULT` sentinel. When the last in-flight tool in a parallel batch
  completes and one was blocked, it breaks the stream before the orchestrator's
  next LLM call — preventing a retry loop.
- **`ainvoke`**: overrides the final `AIMessage` with `_TOOL_SAFETY_REFUSAL` if any
  tool block was signalled, ensuring a consistent user-facing refusal regardless of
  what the LLM generated.
- Catches `ContentSafetyError` / `InputContentSafetyError` / `ToolContentSafetyError`
  at the outermost boundary and converts them to a clean refusal message rather than
  an unhandled exception.

### AuditMiddleware — structured platform audit trail
`deep_agent/src/audit/middleware.py` (added via `build_middleware_list`) emits
structured JSON audit events to stdout for:

| Event type | Trigger | Key details emitted |
|---|---|---|
| `llm_call` | Every model invocation (sync + async) | model, message_count, latency_ms, status |
| `mcp_tool_call` | Tool in the agent's MCP tool name set | tool, args_keys, latency_ms, status |
| `memory_write` | `edit_file`/`write_file` under `/memories/` | path, latency_ms, status |
| `subagent_delegation` | `task` tool call | delegated_subagent, latency_ms, status |

The emitter scrubs sensitive keys recursively (password, token, api_key, secret,
auth, cookie, credentials, etc.) before serialising. A local ring buffer retries
events that fail to emit due to transient I/O errors.

### Memory write protection
`PersonalizationRepository.create_memory()` and `upsert_rule()` run a Guardian
check before writing to Postgres. Unsafe content is rejected and never stored.

### MCP tool abuse visibility
Every tool call is audit-logged via `GraniteGuardianCallbackHandler.on_tool_start`
(sanitized inputs) and `AuditMiddleware` (args keys, latency). Circuit breaker trips
after 5 MCP server failures. `server_names` filtering ensures agents only connect to
their declared MCP servers.

### Human-in-the-loop (HITL) — tool approval interrupts
`deep_agent/src/agent/config/hitl.py` builds `interrupt_on` predicates for
`create_deep_agent`. When enabled, the LangGraph graph pauses before executing
named tools and waits for human approval before resuming.

### CI / supply chain
- Trivy scans container image before push to GHCR; CRITICAL severity fails the build.
- Gitleaks runs in pre-commit on every local commit.
- GitHub Secret Scanning + Push Protection active at platform level.

---

## Remaining risks

### 1. AsyncSubAgent (remote pod) — Medium
**Threat:** Orchestrator delegates to a remote subagent pod via HTTP. Guardian
callback and `GuardianToolProxy` do not cross process boundaries.

**Mitigation in place:** `on_tool_start` Guardian check scans the outgoing
instruction payload before dispatch. `AuditMiddleware` emits a `subagent_delegation`
event for every `task` tool call.

**Residual gap:** The remote pod's own LLM calls and tool calls are only
protected if that pod is deployed with `GUARDIAN_API_BASE` set.

**Required action:** Enforce `GUARDIAN_API_BASE` + Guardian env vars on
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

### 4. Subagent/skill metadata not validated at config load — Addressed (OFFSEC-379)
**Threat:** A tampered, malicious, or compromised subagent/skill catalogue entry
could inject instructions into a subagent's system prompt, delegation
description, or skill body at startup. Because subagents and skills are
resolved from the shared package catalogue, this content may be authored by a
different user or org than the one deploying the agent — making the catalogue
a cross-tenant prompt-injection distribution channel (DI 06).

**Mitigation in place:** `scan_catalogue_safety()`
(`deep_agent/src/agent/config/catalogue_safety.py`) runs once at startup, after
Guardian is initialised (`run_startup()` → `_validate_catalogue_safety()`).
It runs the existing `check_safety` + `check_injection` Guardian checks over
every subagent's `description` + `body` and every skill's `SKILL.md`
`description` + body. Any subagent or skill flagged unsafe or as an injection
attempt is excluded from the running agent (`AgentConfig.exclude_subagent` /
`exclude_skill`, which also scrubs the skill's path out of any already-resolved
`skill_paths`) — the rest of the agent still starts and runs normally.

**Residual gaps:**
- The Guardian scan itself only runs once, at startup. `CONFIG_AUTO_RELOAD`
  (default `true` — see `settings.py`) reloads subagent/skill configs from
  disk on every `AgentConfig` access; previously-excluded items stay excluded
  across those reloads (`AgentConfig._reapply_exclusions`, keyed by name), but
  if a subagent/skill's *content* changes on disk after startup (e.g. hot
  edited during local dev), the new content is not re-scanned until the
  process restarts.
- No-op when Guardian is disabled (`GUARDIAN_API_BASE` unset or
  `guardrail.enabled: false`) — same fail-open posture as every other Guardian
  check in this register.
- Registry-side (publish-time) scanning is still absent — malicious catalogue
  content is only caught when a consuming agent starts up, not blocked from
  being published in the first place. The registry has no Guardian wiring
  today; adding it is a separate, larger effort tracked outside this fix.

---

### 5. Encoded / obfuscated prompt injection — Low
**Threat:** Base64-encoded, Unicode-homoglyph, or multi-step injection chains
that Guardian's model does not classify as unsafe.

**Mitigation in place:** Guardian `check_safety` + `check_injection` now run at
two layers: user input (`on_chat_model_start`) and every tool result
(`GuardianToolProxy` post-check). Full content is scanned — no truncation.

**Residual gap:** Model-level limitation — no classifier is perfect. No
additional decoding / normalization pass before Guardian.

**Required action:** Add a pre-check normalization step (decode base64, strip
unicode overrides) before passing content to `check_safety()`.

---

### 6. Guardian API fail-open on outage — Medium
**Threat:** If the Guardian endpoint is unreachable (network partition, pod crash,
misconfiguration), every `check_safety` and `check_injection` call catches the
exception, logs a warning, and returns `(is_safe=True, "error")`. All guardrail
checks — user input, tool args, tool results, LLM output, and memory writes — silently
pass through for the duration of the outage.

**Mitigation in place:** Each failed check emits a `guardian_check_failed` warning
log with `exc_info=True`. The agent remains available and functional.

**Residual gap:** This is a deliberate availability-over-security tradeoff. A Guardian
outage is operationally invisible to users and produces only per-check warning logs.
There is no alerting, no circuit breaker, and no `GUARDIAN_FAIL_OPEN=false` mode
for deployments that require guardrails to be enforced even at the cost of availability.

**Required action:** Add a `GUARDIAN_FAIL_OPEN` setting (default `true` to preserve
current behaviour). When `false`, failed Guardian checks return `(is_safe=False)`
so requests are blocked during an outage. Add a monitoring alert on the
`guardian_check_failed` log event to detect outages promptly.

---

### 7. Audit log integrity — Low
**Threat:** Audit events are emitted to stdout only. A compromised container
runtime or log-scraping pipeline could drop or tamper with audit records without
detection.

**Mitigation in place:** `AuditEmitter` uses a local buffer to retry transient
I/O failures. Structured JSON format is compatible with standard log collectors
(Fluentd, Datadog, CloudWatch).

**Residual gap:** No tamper-evident storage, log signing, or SIEM forwarding
configured. Audit completeness relies entirely on the container log driver.

**Required action:** Route stdout audit events (`event=platform.audit`) to a
write-once audit store or forward to a SIEM. Add an alert for gaps in
`trace_id` sequence or missing `llm_call` events in active threads.
