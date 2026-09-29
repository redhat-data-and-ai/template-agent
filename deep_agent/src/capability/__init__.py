"""Capability model — enforced tool-invocation manifest independent of the agent prompt.

Provides a frozen, per-request allow-list of tools an agent may invoke,
resolved once from deploy-time frontmatter. The dispatch-time gate runs
in plain Python outside the model's context window so prompt injections
cannot bypass it.

- ``CapabilityManifest`` / ``resolve_capability_manifest`` — build the
  frozen tool allow-list from frontmatter config.
- ``CapabilityToolProxy`` / ``enforce_capability`` — wrap every tool with
  a dispatch-time manifest membership check.
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
