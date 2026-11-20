# Frontmatter Schema

Complete field reference for subagent YAML frontmatter.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | Yes | — | Unique identifier for the subagent. |
| `type` | string | No | `default` | `default`, `compiled`, or `async`. |
| `description` | string | Yes | — | Used by the orchestrator for delegation decisions. Be specific. |
| `model` | string or dict | No | Inherits from orchestrator | LLM model. Dict format: `{provider, name, fallback: {provider, name}}`. |
| `tools` | list[string] | No | [] | Tool names to expose (use `tool_prefix` + tool name from `mcp.json`). |
| `skills` | list[string] | No | [] | Skill names to attach. |
| `mcps` | list[string] | No | Inherits | MCP server names. Omit to inherit from orchestrator. |
| `resources` | list[string] | No | Inherits | MCP resource URI allowlists. Omit to inherit from orchestrator. |
| `accessibility` | string | No | — | `public` or `private`. |
| `groups` | list[dict] | No | [] | LDAP role mappings for access control. |
| `middleware` | dict | No | {} | Per-agent middleware overrides. |
| `graph_id` | string | async only | — | Graph ID on the remote server. Required for `async` type. |
| `url` | string | No | — | Remote server URL. Used with `async` type. |

## Examples

**Minimal (defaults + inheritance):**

```yaml
---
name: my-agent
description: >
  Handles X for Y. Use when the user asks to do Z.
---
```

**Explicit model with fallback:**

```yaml
---
name: my-agent
description: >
  Handles X for Y. Use when the user asks to do Z.
model:
  provider: google
  name: gemini-2.5-pro
  fallback:
    provider: google
    name: gemini-2.5-flash
---
```

**Async type:**

```yaml
---
name: remote-processor
type: async
description: >
  Processes long-running tasks on a remote server.
graph_id: my-graph-id
url: https://remote-server.example.com
---
```
