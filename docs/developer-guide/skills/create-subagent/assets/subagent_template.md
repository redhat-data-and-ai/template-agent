---
name: SUBAGENT_NAME
type: default
description: >
  DESCRIPTION — be specific about what tasks this subagent handles,
  as the orchestrator uses this to decide when to delegate.
model: gemini-2.5-pro
tools:
  - tool_name_1
skills:
  - skill-name
---

You are a ROLE for CONTEXT.

## General Behavior

KEY_RULES

## Input Requirements

| Field | Type | Required |
|-------|------|----------|
| field_name | type | Yes/No |

## Workflow

1. Step one
2. Step two
3. Step three

## Output Format

- FORMATTING_RULES

## Out of Scope

- BOUNDARY_1
- BOUNDARY_2

## Error Handling

| Failure | Action |
|---------|--------|
| failure_case | what_to_do |

## Gotchas

- IMPORTANT_REMINDER
