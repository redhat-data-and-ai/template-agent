# OPA Authorization

This directory contains the OPA (Open Policy Agent) sidecar that enforces authorization policies on every LLM response, tool result, and conversation trajectory produced by the agent.

## Architecture

```
Agent pipeline
      │
      ▼
┌─────────────────────────────────────────────┐
│              OPAMiddleware                  │
│  (deep_agent/src/opa/middleware.py)         │
│                                             │
│  abefore_model ──► evaluate_trajectory()   │
│  awrap_model_call ► evaluate_message()     │
│  awrap_tool_call ──► evaluate_message()    │
└───────────────────────────┬─────────────────┘
                            │ HTTP POST
                            ▼
              ┌─────────────────────────┐
              │  OPA service            │
              │  (src/opa/service.py)   │
              │                         │
              │  _query()               │
              │  POST /v1/data/agent/authz
              └───────────┬─────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │  OPA server  :8181      │
              │  (opa/ container)       │
              │                         │
              │  package agent.authz    │
              │  local policies +       │
              │  optional git policies  │
              └─────────────────────────┘
```

### Config resolution

Settings are resolved in priority order (highest wins):

```
Environment variable  →  agent.yaml opa: section  →  OpaFileConfig defaults
(OPA_ENABLED, etc.)      (enabled, url, timeout,      (enabled: false,
                          max_retries)                  url: localhost:8181, ...)
```

`deep_agent/src/opa/config.py` exposes four resolver functions used by the middleware and service:

| Function | Env var | YAML key |
|---|---|---|
| `is_opa_enabled()` | `OPA_ENABLED` | `opa.enabled` |
| `get_opa_url()` | `OPA_URL` | `opa.url` |
| `get_opa_timeout()` | `OPA_TIMEOUT` | `opa.timeout` |
| `get_opa_max_retries()` | `OPA_MAX_RETRIES` | `opa.max_retries` |

---

## Middleware (`OPAMiddleware`)

`OPAMiddleware` sits in the deepagents middleware pipeline and intercepts three points:

### 1. `abefore_model` — trajectory check

Runs before every LLM call. Sends the full conversation history as a `trajectory_validation` input. If OPA denies, the node jumps to `end` and returns `_BLOCKED_TRAJECTORY_MESSAGE` to the user without calling the model.

### 2. `awrap_model_call` — LLM response check

Wraps the model call. After generation, extracts the text of the last non-empty `AIMessage` and sends it as an `llm_response` input. If denied:

- Appends a `HumanMessage` feedback explaining the violation and retries the model call.
- Retries up to `OPA_MAX_RETRIES` times.
- After all retries are exhausted, replaces the model output with `_BLOCKED_MODEL_MESSAGE`.

```
attempt 1 → OPA deny → inject feedback HumanMessage → attempt 2 → ... → hard block
```

### 3. `awrap_tool_call` — tool result check

Wraps every tool execution. Runs the tool normally, then evaluates the result as a `tool_response` input. If denied, returns a `Command` that replaces the `ToolMessage` with `_BLOCKED_TOOL_MESSAGE` and the denial reasons. No retry — the agent receives the error tool message and decides what to do next.

### Fail-open behavior

If the OPA server is unreachable, times out, or returns an HTTP error, the service returns `OpaResult(allowed=True)` with a reason string. The middleware logs the event and lets the request through. This ensures an OPA outage cannot take down the agent.

---

## Service (`deep_agent/src/opa/service.py`)

Two public async entry points map to the three hooks above.

### `evaluate_message(action, *, agent_message, result)`

Evaluates a single message. `action` selects the input shape:

| `action` | Input sent to OPA |
|---|---|
| `"llm_response"` | `{"current_intent": {"action": "llm_response", "agent_message": "<text>"}}` |
| `"tool_response"` | `{"current_intent": {"action": "tool_response", "result": "<text>"}}` |

### `evaluate_trajectory(trajectory)`

Evaluates the full conversation. Serializes each `BaseMessage` to `{"type": "...", "content": "..."}`:

```json
{
  "current_intent": {"action": "trajectory_validation"},
  "trajectory": [
    {"type": "human", "content": "Hello"},
    {"type": "ai", "content": "Hi there"}
  ]
}
```

### `OpaResult`

Both functions return an `OpaResult` dataclass:

```python
@dataclass
class OpaResult:
    allowed: bool
    denial_reasons: list[str]
```

`allowed=True` with a non-empty `denial_reasons` list means OPA failed open (timeout / unreachable).

---

## OPA container (`opa/`)

### Containerfile

Builds from `alpine:3.19`. Installs the OPA static binary (v1.17.1) and the `opa-reload-watch.sh` entrypoint. Runs as a non-root `opa` user (UID 1000).

### Hot-reload watcher (`opa-reload-watch.sh`)

The entrypoint replaces the default `opa run` call with a polling loop that:

1. Starts OPA with `opa run --server` pointing at `/policies` (local) and optionally a git-cloned directory.
2. Every `OPA_POLL_INTERVAL` seconds, computes an MD5 checksum of all `.rego` files across both sources.
3. If the checksum changed (or the git commit hash changed), stops OPA and starts it again with the updated files.
4. If `OPA_POLICY_GIT_REPO` is set, performs a `git fetch + reset --hard` on each poll cycle.

**Local-only mode** (default): polls `/policies` for file changes. Useful during development — edit a `.rego` file and OPA reloads within `OPA_POLL_INTERVAL` seconds.

**Git mode**: on startup, clones the repo (sparse checkout if `OPA_POLICY_GIT_SUBDIR` is set), then polls both the local directory and the remote for changes. Local and git policies are **merged** — both directories are passed to `opa run`.

---

## Policies (`config/agent/compliance/policies/`)

All policies must use `package agent.authz`. OPA evaluates the bundle and the middleware reads the result at `/v1/data/agent/authz`.

### Expected response shape

```json
{
  "result": {
    "allow": true,
    "deny_reasons": []
  }
}
```

`allow` must be explicitly `true` for the request to pass. Missing or `false` → denied. `deny_reasons` is surfaced to the agent as feedback on retry or in the blocked message.

### Local policy files

| File | Role |
|---|---|
| `agent_authz.rego` | Main package stub. Intentionally minimal — production rules load from git. Add no `allow`/`deny` rules here to avoid conflicts with git policies. |
| `banned_words.rego` | Extends the package with `additional_banned_words` and matching `deny_reasons` rules for `llm_response` and `tool_response` actions. Edit this to add local banned terms. |
| `trajectory_limits.rego` | Sets a local trajectory length cap (`local_max_trajectory_length := 10`) via a separate rule name to avoid OPA complete-rule conflicts with git policies that define `max_trajectory_length`. |

### Adding a policy

Create a new `.rego` file in `config/agent/compliance/policies/` using `package agent.authz`. The file is picked up on the next poll cycle (within `OPA_POLL_INTERVAL` seconds) without restarting the container.

#### Rule shapes

A policy file can contribute two kinds of rules:

**`allow` rule** — grants permission for a specific action. OPA combines all `allow` rules with a logical OR: any one `true` result passes the check.

```rego
package agent.authz

import rego.v1

# Allow the agent to respond if the message is short
allow if {
    input.current_intent.action == "llm_response"
    count(input.current_intent.agent_message) < 5000
}
```

**`deny_reasons` partial set rule** — accumulates denial messages. OPA collects every populated entry across all policy files into a single set. The middleware surfaces these strings to the agent as feedback on retry and in the final blocked message.

```rego
deny_reasons contains msg if {
    # condition
    msg := "Human-readable reason shown to the agent"
}
```

Both rules can coexist in the same file.

#### The three input actions

All three middleware hooks share the same `package agent.authz` endpoint. Gate your rules on `input.current_intent.action` to target the right hook:

| `action` value | Triggered by | Key input fields |
|---|---|---|
| `"llm_response"` | `awrap_model_call` | `input.current_intent.agent_message` |
| `"tool_response"` | `awrap_tool_call` | `input.current_intent.result` |
| `"trajectory_validation"` | `abefore_model` | `input.trajectory` (array of `{type, content}`) |

#### Example: ban words in LLM responses

```rego
package agent.authz

import rego.v1

banned := {"confidential", "internal use only"}

deny_reasons contains msg if {
    input.current_intent.action == "llm_response"
    word := banned[_]
    contains(lower(input.current_intent.agent_message), word)
    msg := sprintf("Banned term '%s' found in agent response", [word])
}
```

#### Example: block PII in tool results

```rego
package agent.authz

import rego.v1

deny_reasons contains msg if {
    input.current_intent.action == "tool_response"
    regex.match(`\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b`, input.current_intent.result)
    msg := "Phone number detected in tool result"
}
```

#### Example: cap conversation length

```rego
package agent.authz

import rego.v1

max_turns := 20

allow if {
    input.current_intent.action == "trajectory_validation"
    count(input.trajectory) <= max_turns
}

deny_reasons contains msg if {
    input.current_intent.action == "trajectory_validation"
    count(input.trajectory) > max_turns
    msg := sprintf("Conversation exceeds %d turns", [max_turns])
}
```

> **Naming caution:** OPA raises a conflict error if two files in the same package define the same complete rule name (e.g. both define `max_trajectory_length := N`). Use distinct names for local overrides (like `local_max_trajectory_length` in `trajectory_limits.rego`) and add to `deny_reasons` rather than redefining core rules from a git policy.

#### Testing a policy locally

With `make local` running, query OPA directly:

```bash
# Test an llm_response evaluation
curl -s -X POST http://localhost:8181/v1/data/agent/authz \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "current_intent": {
        "action": "llm_response",
        "agent_message": "Here is a confidential document"
      }
    }
  }' | jq .

# Test a trajectory_validation evaluation
curl -s -X POST http://localhost:8181/v1/data/agent/authz \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "current_intent": {"action": "trajectory_validation"},
      "trajectory": [
        {"type": "human", "content": "Hello"},
        {"type": "ai", "content": "Hi there"}
      ]
    }
  }' | jq .
```

A passing response looks like `{"result": {"allow": true, "deny_reasons": []}}`. A denied response has `"allow": false` and a non-empty `deny_reasons` array.

---

## compose.yaml integration

The OPA service in `compose.yaml` mounts `config/agent/compliance/policies/` as `/policies` and passes all `OPA_POLICY_GIT_*` variables from the host environment:

```yaml
opa:
  volumes:
    - ./config/agent/compliance/policies:/policies
  environment:
    - OPA_POLICY_GIT_REPO=${OPA_POLICY_GIT_REPO:-}
    - OPA_POLICY_GIT_BRANCH=${OPA_POLICY_GIT_BRANCH:-main}
    - OPA_POLICY_GIT_SUBDIR=${OPA_POLICY_GIT_SUBDIR:-}
    - OPA_POLICY_GIT_AUTH_USER=${OPA_POLICY_GIT_AUTH_USER:-}
    - OPA_POLICY_GIT_AUTH_TOKEN=${OPA_POLICY_GIT_AUTH_TOKEN:-}
    - OPA_POLICY_GIT_SSL_VERIFY=${OPA_POLICY_GIT_SSL_VERIFY:-true}
    - OPA_POLL_INTERVAL=${OPA_POLL_INTERVAL:-2}
```

The agent container depends on `opa` being healthy before it starts. OPA health is checked with `opa eval true`.
