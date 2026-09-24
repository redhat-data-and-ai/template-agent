"""Capability model — enforced tool-invocation manifest independent of the agent prompt.

Addresses OFFSEC-384 (DI-05): "No capability model, the system prompt is treated
as a security control." Before this package existed, the platform enforced
which MCP *servers* an agent declared, but not which *tools* it could actually
invoke at runtime -- an agent that declared an MCP server without an explicit
``tools:`` allow-list silently got every tool that server happened to expose,
and nothing re-checked a tool call against a reviewed manifest at dispatch
time. Any instruction that reached the model's context (including one smuggled
in through a tool result) was therefore transitively a grant of the agent's
full runtime tool surface.

This package provides:

- ``CapabilityManifest`` / ``resolve_capability_manifest`` — build a frozen
  allow-list of tool names for one agent or subagent, once per request, from
  the deploy-time frontmatter agent-engine materializes into the pod. Nothing
  in the running conversation can alter it.
- ``CapabilityToolProxy`` / ``enforce_capability`` — wrap every tool with a
  dispatch-time gate that checks manifest membership immediately before the
  inner tool runs, no matter which code path assembled the tool list. This
  runs in plain Python outside the model's context window, so a prompt
  injection has no channel to reach or bypass it.
"""

from deep_agent.src.capability.manifest import (
    CapabilityManifest,
    resolve_capability_manifest,
)
from deep_agent.src.capability.tool_proxy import CapabilityToolProxy, enforce_capability

__all__ = [
    "CapabilityManifest",
    "resolve_capability_manifest",
    "CapabilityToolProxy",
    "enforce_capability",
]
