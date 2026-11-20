# System Prompt Structure

Recommended sections for a subagent system prompt, based on the patterns
established by `analyst.md` and `publisher.md`.

## Sections

### 1. Identity

One-line role statement. Opens the system prompt immediately after the
frontmatter closing `---`.

> You are a ROLE for CONTEXT.

### 2. General Behavior

Key behavioral rules and constraints. Reference any attached skills the
subagent should read at runtime.

### 3. Input Requirements

Table of expected inputs. Specify field name, type, whether it is required,
and where it comes from (user input, prior subagent output, etc.).

| Field | Type | Required |
|-------|------|----------|
| field_name | type | Yes/No |

### 4. Workflow

Numbered steps the subagent follows, in order. Reference specific tool names
where applicable.

### 5. Output Format

Formatting rules: Markdown, HTML, tables, lists, JSON, etc. Include any
mandatory elements (disclaimers, footers, headers).

### 6. Out of Scope

Explicit boundaries — what the subagent must NOT do. Helps prevent scope creep
and hallucinated capabilities.

### 7. Error Handling

Failure/action table. Cover tool failures, missing inputs, and edge cases.

| Failure | Action |
|---------|--------|
| failure_case | what_to_do |

### 8. Gotchas

Important reminders as a bulleted list. Cover common mistakes, mandatory
inclusions, and non-obvious constraints.

## Template Variable

`{{current_date}}` is replaced at runtime with the current date. Use it in
the system prompt when the subagent needs date awareness.

## Tips

- Keep the identity line short — one sentence.
- Be specific in Out of Scope — vague boundaries are ignored.
- Error handling should cover every tool the subagent uses.
- Gotchas should be bold-prefixed for scannability.
