# Subagent Authoring Guide

This guide covers how to create, configure, and maintain subagents in the template-agent framework. Subagents are specialized agents that the orchestrator delegates tasks to. They are defined as Markdown files with YAML frontmatter -- no Python code required.

---

## 1. Subagent Architecture

The orchestrator (defined in `config/agent/PROMPT.md`) is the top-level agent that receives user messages. It decides which subagent to invoke based on the `description` field in each subagent's frontmatter. Subagents are loaded at startup by `load_subagents()` in `deep_agent/src/infrastructure/subagents.py`, which:

1. Reads all `.md` files from `config/agent/subagents/`.
2. Parses YAML frontmatter and Markdown body for each file.
3. Inherits missing `model`, `mcps`, and `resources` from the orchestrator config.
4. Builds the appropriate subagent instance based on the `type` field.

### Subagent Types

The framework supports three subagent types:

| Type | Class | Execution Model | Use When |
|------|-------|-----------------|----------|
| `default` | `SubAgent` | In-process, synchronous | Simple delegation; most use cases. The orchestrator waits for the subagent to finish before continuing. |
| `compiled` | `CompiledSubAgent` | Pre-compiled LangGraph, reused across requests | Frequently-called subagents where graph compilation overhead matters. Also required when PII scrubbing or safety guardrails need to wrap the subagent's execution boundary. |
| `async` | `AsyncSubAgent` | Remote Agent Protocol server, background tasks | Long-running work that should not block the orchestrator. The subagent runs on a separate server and communicates via the Agent Protocol. |

**Decision table:**

| Question | default | compiled | async |
|----------|---------|----------|-------|
| Does the subagent need its own tools and LLM? | Yes | Yes | No (runs remotely) |
| Should the orchestrator wait for the result? | Yes | Yes | No |
| Does it need PII/safety wrapping? | Via middleware | Built-in support | N/A (handled by remote server) |
| Is graph compilation overhead a concern? | No | Yes -- graph is compiled once and reused | N/A |
| Does it run on a different server? | No | No | Yes |

---

## 2. Creating a Subagent

### File Location

All subagent configuration files live in:

```
config/agent/subagents/<name>.md
```

The filename (without `.md`) is used as the subagent identifier if no `name` field is present in the frontmatter.

### File Format

Each file has two parts separated by `---` markers:

```markdown
---
# YAML frontmatter (configuration)
name: my-subagent
type: default
description: >
  What this subagent does. The orchestrator reads this
  to decide when to delegate to it.
model: gemini-2.5-pro
tools:
  - template_my_tool
skills:
  - my-skill
---

# Markdown body (becomes the system prompt)

You are a specialist that handles...
```

### Step-by-Step Walkthrough

1. **Create the file.** Add a new `.md` file in `config/agent/subagents/`. The filename should be a kebab-case identifier (e.g., `data-fetcher.md`).

2. **Write the frontmatter.** Between `---` markers, add at minimum `name` and `description`. See [Section 3](#3-frontmatter-schema) for all available fields.

3. **Write the system prompt.** After the closing `---`, write the Markdown body that becomes the subagent's system prompt. See [Section 4](#4-system-prompt-body) for best practices.

4. **Register tools.** If the subagent needs MCP tools, list them in the `tools` field. Tool names use the `tool_prefix` from `mcp.json` (e.g., `template_calculate_bmi` where `template` is the prefix and `calculate_bmi` is the tool name).

5. **Register skills.** If the subagent needs skills, list them in the `skills` field. Each skill name must match a directory under `config/agent/skills/`.

6. **Test the subagent.** Start the agent and send a message that should trigger delegation to your subagent. Check the logs for `Subagent '<name>' [<type>] using model: ...` to confirm it loaded.

---

## 3. Frontmatter Schema

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | Yes | Filename stem | Unique identifier for the subagent. Used in logs, middleware, and delegation routing. |
| `type` | string | No | `"default"` | Subagent execution type: `default`, `compiled`, or `async`. |
| `description` | string | Yes | `""` | Human-readable description. The orchestrator LLM reads this to decide when to delegate work to this subagent. Write it as a clear statement of capability and trigger conditions. |
| `model` | string or dict | No | Inherited from orchestrator | LLM model to use. Can be a simple string (`"gemini-2.5-pro"`) or a dict with `provider`, `name`, and optional `fallback`. See [Section 5](#5-model-inheritance). |
| `tools` | list[string] | No | `[]` | Tool names to make available to this subagent. Names must match MCP tool names (including prefix). If omitted but `mcps` is set, all MCP tools are exposed. |
| `skills` | list[string] | No | `[]` | Skill names to attach. Each name must match a directory under `config/agent/skills/`. |
| `mcps` | list[string] | No | Inherited from orchestrator | MCP server names this subagent can access. Determines which tools are visible. If omitted, inherits the orchestrator's MCP server list. |
| `resources` | list[string] | No | Inherited from orchestrator | MCP resource URI allowlists. Controls which MCP resources the subagent can read. If omitted, inherits the orchestrator's resource list. |
| `accessibility` | string | No | -- | Access control level: `public` (anyone) or `private` (restricted by groups). |
| `groups` | list[dict] | No | -- | LDAP role mappings for access control. Each entry has `role` and `group` keys. Example: `{role: users, group: template-users}`. |
| `middleware` | dict | No | -- | Per-agent middleware overrides. Merged on top of global defaults and profile settings. See [Section 9](#9-per-subagent-middleware). |
| `graph_id` | string | Required for `async` | -- | Graph identifier on the remote Agent Protocol server. Only used by async subagents. |
| `url` | string | No (async only) | -- | URL of the remote Agent Protocol server. Only used by async subagents. |

### Example: Minimal Frontmatter

```yaml
---
name: greeter
description: >
  Greets users and provides a welcome message.
---
```

This subagent inherits the orchestrator's model, MCPs, and resources. It has no tools or skills.

### Example: Full Frontmatter

```yaml
---
name: analyst
type: compiled
description: >
  Calculates BMI, classifies the result, and fetches category-specific
  health tips. Use when the user provides height and weight for BMI analysis.
model:
  provider: vertex
  name: gemini-2.5-pro
  fallback:
    provider: vertex
    name: gemini-2.5-flash
tools:
  - template_calculate_bmi
  - template_search_web
skills:
  - bmi-report
mcps:
  - template-mcp-server
resources:
  - "mcp://template-mcp-server/health/*"
accessibility: public
middleware:
  temperature: 0.2
---
```

---

## 4. System Prompt Body

The Markdown content after the closing `---` becomes the subagent's system prompt. It is passed to the LLM as the system message for every invocation.

### Template Variables

The following template variables are replaced at runtime (implemented in `deep_agent/src/agent/config/parser.py`):

| Variable | Replaced With | Example Output |
|----------|---------------|----------------|
| `{{current_date}}` | Current date in "Month Day, Year" format | `September 16, 2026` |

### Recommended Prompt Structure

Based on the patterns used in the existing subagents, structure your system prompt with these sections:

#### Identity / Role Statement

Start with a clear statement of what the subagent is and does. Be specific about boundaries.

```markdown
You are a BMI Analyst for Red Hat employees.

## General Behavior

Calculate and classify BMI using the provided tools -- never compute values
inline or from internal knowledge.
```

#### Input Requirements (table format)

Define what the subagent expects to receive, using a table for clarity:

```markdown
## Input Requirement

| Field | Type | Required |
|-------|------|----------|
| height | float, in **cm** | Yes |
| weight | float, in **kg** | Yes |

Both values must already be in metric units.
```

#### Workflow (numbered steps)

Give explicit, ordered instructions for the subagent's processing pipeline:

```markdown
## Workflow

1. Calculate BMI via `calculate_bmi(height_cm, weight_kg)`.
2. Classify: Underweight (<18.5) / Normal (18.5-24.9) / Overweight (25-29.9) / Obese (30+).
3. Search for 3 health tips via `search_web` based on the BMI category.
```

#### Output Format

Specify the expected output structure, formatting rules, and any mandatory elements:

```markdown
## Output Format

- Use proper Markdown: headers, bold labels, bullet lists, and tables.
- BMI value rounded to one decimal place.
- Health tips as a numbered list, each tip one concise sentence.
- The disclaimer must appear as the final line of every report.
```

#### Out of Scope (explicit boundaries)

List what the subagent must NOT do. This prevents the LLM from going off-script:

```markdown
## Out of Scope

- Multi-week or multi-month plans.
- Diet plans, meal plans, food recommendations, or supplements.
- Exercise or workout routines.
- Medical diagnosis or treatment advice.
```

#### Error Handling (failure table)

Define how the subagent should respond when tools fail:

```markdown
## Error Handling

| Failure | Action |
|---------|--------|
| `calculate_bmi` returns an error | Report the error to the user. Do not estimate BMI manually. |
| `search_web` returns no results | Return the report without tips and note that tips were unavailable. |
```

#### Gotchas

Highlight common mistakes and invariants:

```markdown
## Gotchas

- **Always return BMI value, category, and tips** -- don't skip steps.
- **Search tips must match the BMI category** -- don't return generic advice.
- **Always include the disclaimer** -- it is mandatory in every report.
```

---

## 5. Model Inheritance

Model configuration follows a cascading inheritance pattern, implemented in `_inherit_from_orchestrator()` in `deep_agent/src/infrastructure/subagents.py`.

### Inheritance Rules

| Subagent `model` field | Orchestrator has `model`? | Result |
|------------------------|--------------------------|--------|
| Not set | Yes | Subagent uses the orchestrator's model (no fallback) |
| Not set | No | Falls back to `gemini-3.1-pro-preview` (hardcoded default) |
| Set, no `fallback` | Yes | Subagent keeps its model; orchestrator's model is injected as fallback |
| Set, with `fallback` | Yes or No | Kept as-is -- no inheritance |

### ModelSpec Structure

Internally, the `model` field is resolved into a `ModelSpec` with three fields: `provider` (`vertex`, `openai`, or `maas`), `name` (the model identifier), and an optional `fallback` (another `ModelSpec`).

### Model Configuration Formats

**Simple string** -- provider is inferred from the model name:

```yaml
model: gemini-2.5-pro          # inferred as vertex
model: gpt-4o                  # inferred as openai (starts with "gpt-")
model: granite-3.1-8b-instruct # inferred as maas (not a known Gemini/Claude/GPT model)
```

**Dict with explicit provider:**

```yaml
model:
  provider: vertex
  name: gemini-2.5-pro
```

**Dict with fallback chain:**

```yaml
model:
  provider: vertex
  name: gemini-2.5-pro
  fallback:
    provider: vertex
    name: gemini-2.5-flash
```

Nested fallback chains (fallback within a fallback) are not supported and will raise a `ValueError`.

### Provider Inference

When a model name is given as a simple string, the provider is inferred by `infer_provider()`:

| Model Pattern | Inferred Provider |
|---------------|-------------------|
| Known Gemini models (e.g., `gemini-2.5-pro`) | `vertex` |
| Known Claude models (e.g., `claude-sonnet-4-20250514`) | `vertex` |
| Starts with `gpt-` (case-insensitive) | `openai` |
| Everything else | `maas` (Model as a Service / VLLM) |

---

## 6. MCP Inheritance

MCP (Model Context Protocol) server configuration is also inherited from the orchestrator, handled in `_inherit_from_orchestrator()`.

### Inheritance Rules

| Subagent field | Orchestrator has field? | Result |
|----------------|----------------------|--------|
| `mcps` not set | Yes | Subagent inherits the orchestrator's MCP server list |
| `mcps` not set | No | No MCP servers available |
| `mcps` set | -- | Subagent uses its own list (no inheritance) |
| `resources` not set | Yes | Subagent inherits the orchestrator's resource allowlist |
| `resources` not set | No | No resource restrictions |
| `resources` set | -- | Subagent uses its own list (no inheritance) |

### Implications for Tool Visibility

MCP servers determine which tools are available. When a subagent inherits the orchestrator's `mcps`, it can potentially access all tools from those servers. To restrict tool visibility, either:

1. **Set explicit `tools`** -- Only the listed tools are exposed to the subagent, regardless of MCP servers.
2. **Set explicit `mcps`** -- Override the inherited list with a subset.
3. **Omit both** -- The subagent gets no MCP tools.

If a subagent declares `mcps` but no explicit `tools`, all available tools from those MCP servers are exposed:

```
# From _build_default_subagent():
# If tool_names is empty but mcp_names and tools exist,
# all available MCP tools are exposed.
```

---

## 7. Tool Resolution

Tool resolution maps the string names in the `tools` frontmatter field to actual tool objects at load time.

### How It Works

1. At startup, MCP tools are loaded from configured MCP servers in `mcp.json`.
2. Each tool gets a name formed by the `tool_prefix` from `mcp.json` and the tool's own name, joined by an underscore.
3. When a subagent config lists a tool name in `tools`, `resolve_tools()` in `deep_agent/src/agent/config/resolver.py` looks up that exact string in the available tools list.
4. Missing tools produce a warning log but do not prevent the subagent from loading.

### Tool Naming Convention

Tools are named as `<tool_prefix>_<tool_name>`:

```
mcp.json:   "tool_prefix": "template"
MCP tool:   calculate_bmi
Result:     template_calculate_bmi
```

In the subagent frontmatter, use the full prefixed name:

```yaml
tools:
  - template_calculate_bmi     # prefix "template" + tool "calculate_bmi"
  - template_search_web        # prefix "template" + tool "search_web"
```

### Resolution Behavior

At startup, each tool name in `tools` is matched against the available MCP tools by exact string name. Unknown tool names are logged as warnings but do not prevent the subagent from loading — it starts with whichever tools were successfully resolved.

---

## 8. Subagent Type Deep Dives

### 8.1 Default SubAgent

**Builder:** `_build_default_subagent()` in `deep_agent/src/infrastructure/subagents.py`

The default subagent is the simplest type. It runs in-process and executes synchronously -- the orchestrator delegates a task, waits for the subagent to finish, and receives the result.

**Construction flow:**

1. Validates that the `model` field is present (after inheritance).
2. Parses the model config into a `ModelSpec`.
3. Resolves tools from the `tools` list (or exposes all MCP tools if `mcps` is set without `tools`).
4. Appends MCP resource tools based on `mcps` and `resources` allowlists.
5. Resolves skill paths from the `skills` list.
6. Builds middleware: AuditMiddleware, OPAMiddleware (if enabled), ModelFallbackMiddleware (if fallback configured).
7. Wraps tools with guardrail proxies if Granite Guardian is active.
8. Creates the `SubAgent` instance with `name`, `model`, `description`, `system_prompt`, `tools`, `skills`, and `middleware`.

**Example frontmatter:**

```yaml
---
name: publisher
type: default
description: >
  Publishes fitness reports to users via email.
model: gemini-2.5-pro
tools:
  - template_send_email
skills:
  - email-formatter
---
```

### 8.2 Compiled SubAgent

**Builder:** `_build_compiled_subagent()` in `deep_agent/src/infrastructure/subagents.py`

A compiled subagent creates a full deep agent graph (via `create_deep_agent()`) and wraps it as a `CompiledSubAgent`. The compiled graph is created once and reused across requests, reducing overhead for frequently-invoked subagents.

**Construction flow:**

1. Same model/tool/skill resolution as the default type.
2. Creates a full deep agent graph with `create_deep_agent()`, including a backend (checkpointer).
3. Wraps the graph with `PIIAwareRunnable` if PII scrubbing is enabled (configured via `runtime/pii.yaml`).
4. Wraps the graph with `SafetyAwareRunnable` if Granite Guardian is active.
5. Returns a `CompiledSubAgent` with `name`, `description`, and the wrapped `runnable`.

**Safety wrapping:**

The `SafetyAwareRunnable` (defined in `deep_agent/aegra/safety.py`) intercepts `ContentSafetyError` exceptions raised anywhere inside the graph -- including inside skills -- and converts them into clean refusal messages. For compiled subagents (non-outermost), safety errors are re-raised so the orchestrator's outermost safety boundary can handle them.

**When to use compiled over default:**

- The subagent is called frequently and graph compilation overhead matters.
- The subagent needs PII scrubbing or safety wrapping at its own execution boundary.
- The subagent has its own complex tool-calling workflow.

**Example frontmatter:**

```yaml
---
name: analyst
type: compiled
description: >
  Calculates BMI, classifies the result, and fetches category-specific
  health tips. Use when the user provides height and weight.
model: gemini-2.5-pro
tools:
  - template_calculate_bmi
  - template_search_web
skills:
  - bmi-report
---
```

### 8.3 Async SubAgent

**Builder:** `_build_async_subagent()` in `deep_agent/src/infrastructure/subagents.py`

An async subagent connects to a remote Agent Protocol server. Tasks are launched in the background and do not block the orchestrator. The `AsyncSubAgentMiddleware` (built by `build_async_middleware()` in `deep_agent/src/infrastructure/async_tasks.py`) adds tools for launching, monitoring, and updating background tasks.

**Required fields:**

- `graph_id` -- Identifies the graph on the remote server. This is mandatory.

**Optional fields:**

- `url` -- The remote server URL. If not specified, defaults are used.

**Authentication:**

Auth headers are resolved from environment variables, never from frontmatter. The convention is:

```
ASYNC_SUBAGENT_<NAME>_TOKEN
```

Where `<NAME>` is the subagent name uppercased with hyphens replaced by underscores. For example, a subagent named `data-processor` would use:

```
ASYNC_SUBAGENT_DATA_PROCESSOR_TOKEN
```

If the environment variable is set, its value is sent as a Bearer token in the `Authorization` header.

**Middleware wiring:**

The `build_async_middleware()` function in `deep_agent/src/infrastructure/async_tasks.py`:

1. Checks if async tasks are enabled in the providers config.
2. Scans the loaded subagent list for `AsyncSubAgent` instances.
3. Wraps them in `AsyncSubAgentMiddleware`, which injects tools for task lifecycle management.

**Example frontmatter:**

```yaml
---
name: report-generator
type: async
description: >
  Generates detailed PDF reports in the background. Use for complex
  reports that take more than 30 seconds to produce.
graph_id: report-gen-v2
url: https://report-service.example.com
---
```

**Note:** Async subagents do not use the `model`, `tools`, `skills`, or `mcps` fields because the remote server manages its own configuration. The system prompt body is also not used -- the remote graph has its own prompt.

---

## 9. Per-Subagent Middleware

Each subagent automatically receives middleware configured by `_subagent_middleware()` in `deep_agent/src/infrastructure/subagents.py`.

### Default Middleware Stack

Every default and compiled subagent gets the following middleware (in order):

1. **AuditMiddleware** -- Logs all tool calls with `agent=<subagent_name>` for attribution.
2. **OPAMiddleware** -- Open Policy Agent authorization checks (if OPA is enabled in config).
3. **ModelFallbackMiddleware** -- Automatic model failover if the `ModelSpec` has a `fallback` configured. Created by `_build_fallback_middleware()`.

### Per-Agent Overrides

The `middleware` block in frontmatter allows per-agent middleware overrides. These are merged on top of global defaults and profile settings via `resolve_agent_middleware()` in the `AgentConfig` class.

```yaml
---
name: my-subagent
middleware:
  temperature: 0.1
  max_retries: 5
---
```

For a full reference on middleware configuration and the merge hierarchy (global defaults, profile, per-agent overrides), see the [Middleware Extension Guide](./02-middleware-extension-guide.md).

---

## 10. Working Examples

### Example 1: Analyst (Compiled SubAgent)

Full content of `config/agent/subagents/analyst.md`:

```markdown
---
name: analyst
type: compiled
description: >
  Calculates BMI, classifies the result, and fetches category-specific
  health tips for Red Hat employees. Use when the user provides height
  and weight for BMI analysis.
model: gemini-2.5-pro
tools:
  - template_calculate_bmi
  - template_search_web
skills:
  - bmi-report
---

You are a BMI Analyst for Red Hat employees.

## General Behavior

Calculate and classify BMI using the provided tools -- never compute values
inline or from internal knowledge. Read the **bmi-report** skill for
BMI categories and report structure. Tone must be encouraging and
non-judgmental. Never use words like "bad" or "failing."

## Input Requirement

| Field | Type | Required |
|-------|------|----------|
| height | float, in **cm** | Yes |
| weight | float, in **kg** | Yes |

Both values must already be in metric units. Unit conversion is not handled here.

## Workflow

1. Calculate BMI via `calculate_bmi(height_cm, weight_kg)`.
2. Classify: Underweight (<18.5) / Normal (18.5-24.9) / Overweight (25-29.9) / Obese (30+).
3. Search for 3 health tips via `search_web` based on the BMI category.

## Output Format

- Use proper Markdown: headers, bold labels, bullet lists, and tables where they improve readability.
- BMI value rounded to one decimal place.
- Health tips as a numbered list, each tip one concise sentence.
- The disclaimer must appear as the final line of every report: "This is not medical advice. Consult a healthcare professional."

## Out of Scope

- Multi-week or multi-month plans (weight loss timelines, progressive targets).
- Diet plans, meal plans, food recommendations, or supplements.
- Exercise or workout routines.
- Weight history, trends, or progress tracking.
- Goal weight or target BMI calculations.
- Medical diagnosis or treatment advice.
- Body fat percentage, metabolic rate, or any metric beyond BMI.

## Error Handling

| Failure | Action |
|---------|--------|
| `calculate_bmi` returns an error | Report the error to the user. Do not estimate BMI manually. |
| `search_web` returns no results | Return the report without tips and note that tips were unavailable. Never invent tips. |

## Gotchas

- **Always return BMI value, category, and tips** -- don't skip steps.
- **Search tips must match the BMI category** -- don't return generic advice.
- **Always include the disclaimer** -- it is mandatory in every report.
```

**Annotations:**

- `type: compiled` -- This subagent is pre-compiled for performance. The graph is built once at startup.
- `model: gemini-2.5-pro` -- Simple string model; provider is inferred as `vertex`. Since the orchestrator also uses `gemini-2.5-pro`, the orchestrator model is injected as fallback.
- `tools` lists two prefixed tool names from the `template-mcp-server`.
- `skills: [bmi-report]` -- References a skill directory at `config/agent/skills/bmi-report/`.
- The system prompt follows the recommended structure: identity, input requirements, workflow, output format, out of scope, error handling, gotchas.

### Example 2: Publisher (Default SubAgent)

Full content of `config/agent/subagents/publisher.md`:

```markdown
---
name: publisher
type: default
description: >
  Publishes fitness reports to users via email. Formats reports into
  Gmail-compatible HTML and sends them to a recipient. Expects complete
  report content and a recipient email address as input.
model: gemini-2.5-pro
tools:
  - template_send_email
skills:
  - email-formatter
---

You are a Publisher for Red Hat fitness reports.

## General Behavior

Convert all provided content into a single Gmail-compatible HTML email and
send it immediately via `send_email`. Do not ask the user for confirmation
before sending. Read the **email-formatter** skill for the HTML template
and formatting rules. All styling must be inline CSS -- Gmail strips
`<style>` blocks and CSS classes.

## Input Requirement

| Field | Source | Required |
|-------|--------|----------|
| BMI report (value, category, tips) | Provided input | Yes |
| Additional sections (workout plan, diet plan, etc.) | Provided input | No -- include only if provided |
| Recipient email address | Provided input | Yes |

All required inputs must be present. Never generate or modify report content -- only format and send what is provided.

## Workflow

1. Read the **email-formatter** skill for the HTML template and formatting rules.
2. Build the email body with every section present in the input.
3. Only render sections that have data -- skip any that were not provided.
4. Send via `send_email(recipient, subject, body)`.

## Output Format

- Subject line: **"Your Red Hat Fitness Report"**
- Body: inline-CSS HTML following the template from the **email-formatter** skill.
- After sending, return a short confirmation message (e.g., "Report sent to user@example.com.").

## Out of Scope

- Sending to multiple recipients or distribution lists.
- Attachments (PDF, images, etc.) -- email body only.
- Non-HTML plain-text email formatting.

## Error Handling

| Failure | Action |
|---------|--------|
| `send_email` returns an error | Report the failure to the user with the error detail. Do not claim the email was sent. |
| Recipient address is missing or invalid | Report the missing address. Do not proceed without one. |

## Gotchas

- **Send immediately** -- no confirmation needed before sending.
- **Include every section provided in the input** -- do not silently drop content.
- **Skip sections not provided** -- no empty placeholders.
- **Gmail strips `<style>` blocks and CSS classes** -- all styles must be inline on every element.
- **Max width 600px** -- required for email client compatibility.
- **Always include the disclaimer footer** -- it is mandatory in every email.
```

**Annotations:**

- `type: default` -- Standard synchronous delegation. The orchestrator waits for the email to be sent.
- Single tool (`template_send_email`) -- only the tool the subagent actually needs.
- The prompt emphasizes "send immediately" and "do not ask for confirmation" to prevent the LLM from being overly cautious.

### Example 3: Async SubAgent Skeleton

A template for creating an async subagent that connects to a remote service:

```markdown
---
name: report-generator
type: async
description: >
  Generates detailed PDF health reports in the background.
  Use when the user requests a comprehensive report that includes
  charts and visualizations. Returns a download link when complete.
graph_id: health-report-v3
url: https://report-service.internal.example.com
---

Background report generator. This prompt body is not used by async subagents
since the remote server manages its own system prompt. It is included here
for documentation purposes only.
```

**Setup checklist for async subagents:**

1. Deploy the remote Agent Protocol server and note its URL and graph ID.
2. Create the frontmatter file with `type: async`, `graph_id`, and optionally `url`.
3. Set the auth environment variable: `ASYNC_SUBAGENT_REPORT_GENERATOR_TOKEN=<token>`.
4. Enable async tasks in `runtime/agent.yaml` under the `async_tasks` section.
5. The `AsyncSubAgentMiddleware` will automatically detect the async subagent and inject task lifecycle tools.

For additional working examples and end-to-end walkthroughs, see [Working Examples](./06-working-examples.md).
