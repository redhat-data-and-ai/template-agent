# Inheritance Rules

Subagents inherit configuration from their parent orchestrator. This document
describes the inheritance behavior implemented in
`deep_agent/src/infrastructure/subagents.py` (`_inherit_from_orchestrator`).

## Model Inheritance

| Subagent Config | Behavior |
|-----------------|----------|
| No `model` in frontmatter | Uses the orchestrator's model (no fallback). If the orchestrator also has no model, falls back to the system default. |
| `model` without `fallback` | Keeps the subagent's model as primary; the orchestrator's model becomes the fallback automatically. |
| `model` with explicit `fallback` | Kept as-is — no automatic injection. |

### Practical Implications

- **Omit `model`** when you want the subagent to always match the orchestrator.
  This is the simplest option and avoids drift.
- **Set `model` without `fallback`** when you want a specific model but want
  automatic resilience — the orchestrator's model is injected as the fallback.
- **Set `model` with `fallback`** when you need full control over both primary
  and fallback models.

## MCP Inheritance

| Subagent Config | Behavior |
|-----------------|----------|
| No `mcps` in frontmatter | Inherits the orchestrator's full MCP server list. |
| `mcps` specified | Uses only the listed MCP servers — no inheritance. |

## Resource Inheritance

| Subagent Config | Behavior |
|-----------------|----------|
| No `resources` in frontmatter | Inherits the orchestrator's resource allowlist. |
| `resources` specified | Uses only the listed resources — no inheritance. |

## Middleware

Every subagent automatically receives the following middleware (no configuration needed):

- **AuditMiddleware** — always active.
- **OPAMiddleware** — active if OPA is enabled in the deployment.
- **ModelFallbackMiddleware** — active if the resolved model config has a fallback.

The `middleware` block in frontmatter can override specific middleware settings
for the subagent without affecting other subagents or the orchestrator.
