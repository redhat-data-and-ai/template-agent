---
name: create-subagent
description: >
  Guides creation of new subagents. Use when a user asks to create,
  add, or define a new subagent for task delegation.
---

# Create Subagent

Step-by-step guide for creating a new subagent definition file with valid
frontmatter and a well-structured system prompt.

## When to Use

When a user asks to create, add, or define a new subagent for task delegation.

## Step-by-Step Checklist

1. **Determine purpose** — what tasks will this subagent handle? Be specific;
   the orchestrator uses the `description` field to decide when to delegate.
2. **Choose the type** — see the decision table below.
3. **Identify tools** — check available MCP tools (use `tool_prefix` + tool
   name from `mcp.json`). Only list tools the subagent actually needs.
4. **Identify skills** — attach any skills the subagent should read at runtime.
5. **Create the file** — `config/agent/subagents/<name>.md`.
6. **Write YAML frontmatter** — see `references/frontmatter_schema.md` for all
   fields and defaults.
7. **Write the system prompt body** — follow the recommended structure in
   `references/prompt_structure.md`. Use `assets/subagent_template.md` as a
   starting point.
8. **Verify** — confirm the subagent loads correctly and the orchestrator can
   delegate to it.

## Type Decision Table

| Use Case | Type |
|----------|------|
| Simple delegation, occasional use | `default` |
| Frequently called, needs performance | `compiled` |
| Runs on a separate server | `async` |

## Resources

- **Frontmatter Schema:** `references/frontmatter_schema.md`
- **Inheritance Rules:** `references/inheritance_rules.md`
- **Prompt Structure:** `references/prompt_structure.md`
- **Starter Template:** `assets/subagent_template.md`

## Critical Requirements

- `name` and `description` are **required** in frontmatter.
- `description` is what the orchestrator uses to decide when to delegate — make
  it specific and action-oriented.
- Model inherits from the orchestrator if not specified in frontmatter.
- MCP servers inherit from the orchestrator if `mcps` is not specified.
- For `async` type, `graph_id` is **required**.
- See `references/inheritance_rules.md` for full inheritance behavior.
