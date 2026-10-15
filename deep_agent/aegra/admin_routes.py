"""Read-only introspection of the agent's resolved configuration.

Operators need to see what the process actually resolved, not what the
YAML says. Approval policies are glob patterns resolved in order, so the
effective decision for a given tool is not readable from the config file
without applying the patterns by hand — which is how a tool silently ends
up auto-allowed.

Gated on the same privileged LDAP role as the eval endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from deep_agent.aegra.eval_routes import _require_developer
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(prefix="/admin", tags=["admin"])

# deepagents' built-ins, which never appear in the caller-supplied tool
# list but are still governed by approval policy.
_BUILTIN_TOOL_NAMES = frozenset(
    {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "execute",
        "write_todos",
        "task",
        "compact_conversation",
    }
)


@router.get("/config")
async def admin_config(_token: str = Depends(_require_developer)) -> dict[str, Any]:
    """Report the agent's effective configuration, secrets omitted."""
    from deep_agent.src.agent.config import agent_config
    from deep_agent.src.agent.config.hitl import describe_policies

    orchestrator = agent_config.get_orchestrator_config()

    # Resolved, not raw: this must reflect the defaults/profile/override
    # merge the process actually applied.
    approval = agent_config.resolve_agent_middleware(
        orchestrator.get("model", "")
    ).human_approval

    skills = agent_config.get_available_skills()
    servers = agent_config.get_mcp_servers()
    effects = _effects_mode()

    return {
        "agent": {
            "name": orchestrator.get("name"),
            "model": orchestrator.get("model"),
        },
        "approval": {
            "enabled": approval.enabled,
            "mode": approval.mode,
            "default_decision": approval.default_decision,
            "policy_count": len(approval.policies),
            # Per tool, so the effect of the globs is visible rather than
            # inferred.
            "tools": describe_policies(
                set(skills) | set(servers) | set(_BUILTIN_TOOL_NAMES),
                approval,
                effects=effects,
            ),
        },
        "execution": {"effects": effects},
        "skills": {name: str(path) for name, path in sorted(skills.items())},
        "mcp_servers": {
            name: {
                "enabled": cfg.get("enabled", False),
                "transport": cfg.get("transport"),
                "auth_mode": cfg.get("auth_mode"),
                # url and auth_env_var deliberately omitted.
            }
            for name, cfg in sorted(servers.items())
        },
        "policy_engine": _opa_summary(),
    }


def _opa_summary() -> dict[str, Any]:
    """Whether external policy evaluation is active, and how it fails.

    `fail_open` matters more than `enabled`: a policy engine that is
    enabled but failing open is not enforcing anything.
    """
    try:
        from deep_agent.src.opa.config import get_opa_fail_open, is_opa_enabled

        return {
            "opa_enabled": is_opa_enabled(),
            "fail_open": get_opa_fail_open(),
        }
    except Exception as exc:
        logger.debug("Could not read OPA config: %s", exc)
        return {"opa_enabled": None, "fail_open": None}


def _effects_mode() -> str:
    """The configured effects axis, defaulting to live."""
    from deep_agent.src.agent.config import agent_config
    from deep_agent.src.agent.config.middleware import ExecutionConfig

    raw = agent_config.get_orchestrator_config().get("execution") or {}
    try:
        return ExecutionConfig(**raw).effects
    except Exception:
        logger.warning("Invalid execution config; assuming effects=live")
        return "live"
