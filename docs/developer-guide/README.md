# Developer Guide

This guide covers everything you need to configure, extend, and operate a template-agent instance. No Python code changes are required for standard configuration — everything is driven by YAML, JSON, and Markdown files in `config/agent/`.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      HTTP / WebSocket                       │
│                    (FastAPI + Starlette)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────┐    ┌──────────────────────────────────────┐  │
│  │           │    │       Middleware Pipeline             │  │
│  │  Auth     │    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌───────┐  │  │
│  │  (SSO /   │───▶│  │Audit│→│ OPA │→│ PII │→│Custom │  │  │
│  │   OIDC)   │    │  └─────┘ └─────┘ └─────┘ └───────┘  │  │
│  │           │    │  ┌───────┐ ┌────────┐ ┌──────────┐   │  │
│  └───────────┘    │  │Limits │→│Retry / │→│Fallback  │   │  │
│                   │  │       │ │Backoff │ │          │   │  │
│                   │  └───────┘ └────────┘ └──────────┘   │  │
│                   └──────────────────────────────────────┘  │
│                              │                              │
│                              ▼                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Orchestrator (PROMPT.md)                  │  │
│  │         LangGraph Agent + LLM Provider                │  │
│  └──────────┬────────────────┬───────────────────────────┘  │
│             │                │                              │
│     ┌───────▼──────┐  ┌─────▼──────────────────────────┐   │
│     │  Subagents   │  │      MCP Servers (External)    │   │
│     │              │  │                                 │   │
│     │ ┌──────────┐ │  │  ┌────────┐  ┌────────┐        │   │
│     │ │ default  │ │  │  │ SSO    │  │ OAuth  │        │   │
│     │ │ compiled │ │  │  │ DCR    │  │API Key │        │   │
│     │ │  async   │ │  │  └────────┘  └────────┘        │   │
│     │ └──────────┘ │  └─────────────────────────────────┘   │
│     └──────────────┘                                        │
│                                                             │
│  ┌──────────────────────┐  ┌────────────────────────────┐   │
│  │   Observability      │  │     Data Stores            │   │
│  │                      │  │                            │   │
│  │  Langfuse (LLM)      │  │  PostgreSQL (checkpoints)  │   │
│  │  OpenTelemetry (ops) │  │  Redis (cache, tokens)     │   │
│  │                      │  │  MongoDB (token budget)    │   │
│  └──────────────────────┘  └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Configuration File Map

| File | Path | Format | Purpose |
|------|------|--------|---------|
| **agent.yaml** | `config/agent/runtime/agent.yaml` | YAML | Unified runtime config — identity, providers, middleware, filesystem, cache, memory, guardrails, OPA, logging, server |
| **mcp.json** | `config/agent/mcp.json` | JSONC | MCP server registry — endpoints, auth modes, SSL, timeouts |
| **pii.yaml** | `config/agent/runtime/pii.yaml` | YAML | PII detection rules — strategies, providers, custom regex |
| **observability.yaml** | `config/agent/runtime/observability.yaml` | YAML | OpenTelemetry config — exporter, metrics, tracing |
| **ui.yaml** | `config/agent/runtime/ui.yaml` | YAML | Frontend BFF config — CORS, security, features |
| **secrets.example.yaml** | `config/agent/runtime/secrets.example.yaml` | YAML | Secrets reference (documentation only) |
| **PROMPT.md** | `config/agent/PROMPT.md` | MD + YAML frontmatter | Orchestrator system prompt and agent metadata |
| **Subagent .md files** | `config/agent/subagents/*.md` | MD + YAML frontmatter | Subagent definitions — one file per subagent |
| **.env.example** | `.env.example` | dotenv | Environment variable reference |

> **Note:** There are no separate `middleware.yaml` or `providers.yaml` files. Middleware and provider configuration are sections within the unified `agent.yaml`.

## Guides

### Which guide do I need?

| I want to... | Guide |
|--------------|-------|
| Understand every config field and its default | [Configuration Reference](./01-configuration-reference.md) |
| Write a custom middleware to intercept LLM/tool calls | [Middleware Extension Guide](./02-middleware-extension-guide.md) |
| Create a new subagent for task delegation | [Subagent Authoring Guide](./03-subagent-authoring-guide.md) |
| Connect an external MCP server for tools | [MCP Server Integration](./04-mcp-server-integration.md) |
| Set up Langfuse for LLM observability | [Langfuse Setup Guide](./05-langfuse-setup-guide.md) |
| Get copy-paste-ready example configs | [Working Examples](./06-working-examples.md) |

### Guide Index

1. **[Configuration Reference](./01-configuration-reference.md)** — Every field in agent.yaml, mcp.json, pii.yaml, observability.yaml, ui.yaml, secrets.example.yaml, PROMPT.md frontmatter, and .env.example. Includes types, defaults, and validation constraints.

2. **[Middleware Extension Guide](./02-middleware-extension-guide.md)** — How the middleware pipeline works, the `AgentMiddleware` base class and its hooks, writing and registering custom middleware, built-in middleware reference, resolution and exclusion rules.

3. **[Subagent Authoring Guide](./03-subagent-authoring-guide.md)** — Creating subagents as Markdown files, frontmatter schema, three subagent types (default, compiled, async), model and MCP inheritance, system prompt best practices.

4. **[MCP Server Integration](./04-mcp-server-integration.md)** — Connecting external MCP servers, authentication modes (SSO, OAuth, DCR, API key), token management, circuit breaker, resource tools, health checks.

5. **[Langfuse Setup Guide](./05-langfuse-setup-guide.md)** — Setting up Langfuse observability with env vars, what gets traced, HITL trace stitching, PII scrubbing of traces, feedback integration, OpenTelemetry.

6. **[Working Examples](./06-working-examples.md)** — Complete, copy-paste-ready example configs for every pattern: minimal and full agent.yaml, mcp.json per auth mode, subagent skeletons, pii.yaml rules, Langfuse .env, custom middleware.

## Developer Skills

Structured skill definitions with checklists, reference docs, and templates for common development tasks. Each skill includes a `SKILL.md` (instructions), `references/` (field schemas, validation rules), and `assets/` (templates).

| Skill | Path | Purpose |
|-------|------|---------|
| **write-middleware** | [`skills/write-middleware/`](./skills/write-middleware/SKILL.md) | Create custom agent middleware with correct hooks, registration, and testing |
| **create-subagent** | [`skills/create-subagent/`](./skills/create-subagent/SKILL.md) | Author new subagents with valid frontmatter, inheritance, and prompt structure |
| **add-mcp-server** | [`skills/add-mcp-server/`](./skills/add-mcp-server/SKILL.md) | Connect MCP servers with correct auth mode config and validation |
| **setup-langfuse** | [`skills/setup-langfuse/`](./skills/setup-langfuse/SKILL.md) | Configure Langfuse observability with env vars, PII scrubbing, and verification |

## Getting Started

1. Start with the [Working Examples](./06-working-examples.md) — copy the minimal `agent.yaml` to get running quickly.
2. Use the [Configuration Reference](./01-configuration-reference.md) to customize fields as needed.
3. Follow the specific guide for whatever you're integrating (middleware, subagents, MCP, or Langfuse).
4. Use the [Developer Skills](#developer-skills) for step-by-step checklists when extending the agent.

## Related Documentation

- [Main README](../../README.md) — Project overview, quick start, API reference
- [Contributing Guide](../../CONTRIBUTING.md) — Branch strategy, conventional commits, code style
- [OPA Authorization](../../opa/README.md) — Policy authoring and hot-reload
- [Evaluation Guide](../../config/agent/evals/README.md) — Running evals with Promptfoo and Lightspeed
- [Security Risk Register](../security/guardrails-risk-register.md) — Security controls and risk mitigations
