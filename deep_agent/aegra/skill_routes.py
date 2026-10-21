"""What this agent is made of — skills, subagents, and its prompt.

Deliberately ungated, unlike `/admin/config`. A skill's name and
description are the clearest statement of the agent's capabilities,
and that is the first thing a user needs — "can it review a merge
request?" should be answerable without asking it and hoping.

The list exposes frontmatter only; `/skills/{name}` serves the full
source. Showing the operating instructions is a deliberate choice:
these are internal agents whose skills live in git, and someone
deciding whether to trust an answer about a merge request needs to
read the rules the agent was following. The alternative — a
capability list you cannot inspect — asks for trust without
evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from deep_agent.src.agent.config.parser import parse_frontmatter
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(tags=["skills"])


def _describe(path: Any) -> str:
    """The skill's own description, from `SKILL.md` frontmatter."""
    skill_md = path / "SKILL.md"
    if not skill_md.is_file():
        return ""
    try:
        meta = parse_frontmatter(skill_md)
    except Exception:
        # A malformed skill should cost its own description, not the
        # whole list — one bad frontmatter must not blank the page.
        logger.debug("Could not parse frontmatter for %s", skill_md, exc_info=True)
        return ""
    description = meta.get("description") or ""
    return " ".join(str(description).split())


@router.get("/skills")
async def list_skills() -> dict[str, Any]:
    """Skills this agent has, and which of its agents use each one."""
    from deep_agent.src.agent.config import agent_config

    available = agent_config.get_available_skills()

    # Who uses what. A skill present on disk but attached to nothing is
    # not a capability — it is a file — so the distinction is reported
    # rather than smoothed over.
    users: dict[str, list[str]] = {}
    orchestrator = agent_config.get_orchestrator_config()
    for name in orchestrator.get("skills") or []:
        users.setdefault(str(name), []).append(
            str(orchestrator.get("name") or "orchestrator")
        )
    for agent_name, cfg in (agent_config.get_all_subagent_configs() or {}).items():
        for name in cfg.get("skills") or []:
            users.setdefault(str(name), []).append(str(agent_name))

    skills = [
        {
            "name": name,
            "description": _describe(path),
            "agents": sorted(users.get(name, [])),
            "attached": bool(users.get(name)),
        }
        for name, path in available.items()
    ]

    # Attached first. A skill whose MCP server is not enabled is
    # shipped-but-dormant, and interleaving those alphabetically with
    # the live ones makes the reader check `attached` on every row to
    # learn what the agent can currently do.
    skills.sort(key=lambda s: (not s["attached"], s["name"]))
    return {"skills": skills}


# Text that belongs in an editor pane. Binary assets and caches do not.
_READABLE_SUFFIXES = {".md", ".json", ".yaml", ".yml", ".txt", ".py", ".sh"}

# One skill's reference file is prose, not a dataset. A cap keeps a
# stray large file from becoming a multi-megabyte JSON response.
_MAX_FILE_BYTES = settings.SKILL_MAX_FILE_BYTES


@router.get("/skills/{name}")
async def read_skill(name: str) -> dict[str, Any]:
    """The full source of one skill, for read-only display."""
    from fastapi import HTTPException

    from deep_agent.src.agent.config import agent_config

    # Looked up by exact key in the scanned index rather than joined
    # onto a path. `name` arrives from the URL, and `../` reaching the
    # filesystem is the whole class of bug this avoids — there is no
    # path construction from user input anywhere below.
    available = agent_config.get_available_skills()
    root = available.get(name)
    if root is None:
        raise HTTPException(status_code=404, detail=f"No such skill: {name}")

    root = root.resolve()
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _READABLE_SUFFIXES:
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        resolved = path.resolve()
        # Belt and braces: a symlink inside the skill directory could
        # still point outside it.
        if not resolved.is_relative_to(root):
            logger.warning("Skipping %s: escapes its skill directory", path)
            continue
        try:
            raw = resolved.read_bytes()
        except OSError:
            logger.debug("Could not read %s", resolved, exc_info=True)
            continue
        truncated = len(raw) > _MAX_FILE_BYTES
        text = raw[:_MAX_FILE_BYTES].decode("utf-8", errors="replace")
        entry: dict[str, Any] = {
            "path": str(resolved.relative_to(root)),
            "content": text,
            "truncated": truncated,
        }

        # Split frontmatter from prose server-side. The metadata and
        # the instructions are different kinds of thing — one is a
        # contract the loader reads, the other is text for a model —
        # and rendering the YAML as part of the document makes the
        # first line of every skill a wall of keys.
        #
        # Done here rather than in the browser because the parser
        # already exists and is the same one the loader uses, so the
        # split shown is the split that actually happens. A second
        # implementation in TypeScript could disagree with it.
        if resolved.suffix.lower() == ".md" and not truncated:
            try:
                meta = parse_frontmatter(resolved)
            except Exception:
                logger.debug("No usable frontmatter in %s", resolved, exc_info=True)
            else:
                body = meta.pop("body", None)
                if meta:
                    entry["frontmatter"] = {k: meta[k] for k in sorted(meta)}
                if isinstance(body, str):
                    entry["body"] = body

        files.append(entry)

    # SKILL.md first — it is the one everybody opens the dialog to read.
    files.sort(key=lambda f: (f["path"] != "SKILL.md", f["path"]))
    return {"name": name, "files": files}


def _split_markdown(path: Path) -> dict[str, Any]:
    """A markdown file as frontmatter plus body.

    Shared by subagents and the orchestrator prompt for the same
    reason the skill viewer does it: the frontmatter is a contract
    the loader reads and the body is text for a model, and rendering
    them as one document opens every page with a wall of YAML.

    Parsed with the loader's own parser, so the split shown is the
    split that actually happens.
    """
    raw = path.read_bytes()
    truncated = len(raw) > _MAX_FILE_BYTES
    text = raw[:_MAX_FILE_BYTES].decode("utf-8", errors="replace")
    entry: dict[str, Any] = {
        "path": path.name,
        "content": text,
        "truncated": truncated,
    }
    if not truncated:
        try:
            meta = parse_frontmatter(path)
        except Exception:
            logger.debug("No usable frontmatter in %s", path, exc_info=True)
        else:
            body = meta.pop("body", None)
            if meta:
                entry["frontmatter"] = {k: meta[k] for k in sorted(meta)}
            if isinstance(body, str):
                entry["body"] = body
    return entry


@router.get("/subagents")
async def list_subagents() -> dict[str, Any]:
    """The specialists this agent delegates to."""
    from deep_agent.src.agent.config import agent_config

    out = []
    for name, cfg in sorted((agent_config.get_all_subagent_configs() or {}).items()):
        out.append(
            {
                "name": name,
                "description": " ".join(str(cfg.get("description") or "").split()),
                "model": str(cfg.get("model") or ""),
                # What it is allowed to reach. The useful thing to
                # know about a subagent is its blast radius, not its
                # prose.
                "mcps": sorted(str(m) for m in (cfg.get("mcps") or [])),
                "skills": sorted(str(s) for s in (cfg.get("skills") or [])),
                # The enforced allow-list, not a hint. An explicit
                # `tools:` is authoritative and is never widened with
                # the rest of the declared servers' tools, so a
                # subagent naming `mcps: [gitlab]` can still be
                # confined to two of its tools — which the mcps list
                # alone reads as the opposite of.
                "tools": sorted(str(t) for t in (cfg.get("tools") or [])),
                "tool_scope": (
                    "explicit"
                    if cfg.get("tools")
                    else ("inherited" if cfg.get("mcps") else "none")
                ),
            }
        )
    return {"subagents": out}


def _subagent_file(name: str) -> Path:
    """Resolve a subagent's markdown, refusing anything outside the dir."""
    from fastapi import HTTPException

    from deep_agent.src.agent.config import agent_config

    # Matched against the loaded set rather than joined onto a path:
    # `name` arrives from the URL, so there must be nothing for `../`
    # to traverse.
    if name not in (agent_config.get_all_subagent_configs() or {}):
        raise HTTPException(status_code=404, detail=f"No such subagent: {name}")
    path = (agent_config.base_dir / "subagents" / f"{name}.md").resolve()
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No such subagent: {name}")
    return path


@router.get("/subagents/{name}")
async def read_subagent(name: str) -> dict[str, Any]:
    """One subagent's definition, for read-only display."""
    return {"name": name, "files": [_split_markdown(_subagent_file(name))]}


@router.get("/prompt")
async def read_prompt() -> dict[str, Any]:
    """The orchestrator's prompt — its identity, routing and rules."""
    from fastapi import HTTPException

    from deep_agent.src.agent.config.paths import config_file

    path = config_file("prompt")
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="No prompt configured")
    return {"name": path.name, "files": [_split_markdown(path)]}


def _provider_of(model: str) -> str:
    """Which provider serves this name.

    By prefix, like `llm.create_model` decides it — an allowlist here
    would be a second list to keep in step, and the one that goes
    stale is always the copy.
    """
    from deep_agent.src.agent.llm import (
        is_claude_model,
        is_openai_model,
        is_vertex_model,
    )

    name = (model or "").strip()
    if not name:
        return "unknown"
    if is_claude_model(name):
        return "anthropic-vertex"
    if is_vertex_model(name):
        return "google-vertex"
    if is_openai_model(name):
        return "openai"
    # Everything else goes to the configured OpenAI-compatible
    # endpoint, which is how a local stub or a self-hosted model is
    # reached.
    return "openai-compatible"


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """Which model answers, and which model each subagent uses.

    Ungated for the same reason the skill list is: "which model wrote
    this" is a question a reader of an answer is entitled to ask, and
    it is the first thing anyone checks when an answer looks wrong.

    Names and roles only. Endpoints, project ids and credentials are
    deployment facts, not capability facts, and belong in /admin.
    """
    from deep_agent.src.agent.config import agent_config

    def _name_of(raw: Any) -> str:
        # `model:` is either a bare name or a mapping with one.
        if isinstance(raw, dict):
            return str(raw.get("name") or "").strip()
        return str(raw or "").strip()

    entries: list[dict[str, Any]] = []

    orchestrator = ""
    try:
        orchestrator = _name_of(
            (agent_config.get_orchestrator_config() or {}).get("model")
        )
    except Exception:
        logger.debug("Could not read the orchestrator model", exc_info=True)
    if orchestrator:
        entries.append(
            {
                "role": "orchestrator",
                "name": orchestrator,
                "provider": _provider_of(orchestrator),
                "used_by": "The agent itself — every turn that is not delegated.",
            }
        )

    try:
        subagents = agent_config.get_all_subagent_configs() or {}
    except Exception:
        logger.debug("Could not read subagent configs", exc_info=True)
        subagents = {}

    for name in sorted(subagents):
        cfg = subagents.get(name) or {}
        model = _name_of(cfg.get("model")) or orchestrator
        entries.append(
            {
                "role": "subagent",
                "name": model,
                "provider": _provider_of(model),
                "used_by": name,
                # True when the subagent names no model of its own, so
                # a reader can tell "chose this" from "inherited it".
                "inherited": not _name_of(cfg.get("model")),
            }
        )

    # A profile can strip tools or middleware for a given model, so
    # a capability Skills still advertises can be quietly absent.
    # Reported against the models above rather than as a list of
    # profiles: the question is "what can *this* agent not do", not
    # "what profiles exist".
    profiles: dict[str, Any] = {}
    try:
        raw = agent_config.get_providers_config().harness_profiles
        if isinstance(raw, dict):
            profiles = raw
    except Exception:
        logger.debug("Could not read harness profiles", exc_info=True)

    for entry in entries:
        profile = profiles.get(entry["name"])
        if profile is None:
            continue
        # Parsed models, not dicts — `get_raw_section` does not exist
        # and the first version read nothing at all, which an
        # `except Exception` turned into "no exclusions" rather than
        # into an error.
        excluded_tools = [
            str(t) for t in (getattr(profile, "excluded_tools", None) or [])
        ]
        excluded_middleware = [
            str(m) for m in (getattr(profile, "excluded_middleware", None) or [])
        ]
        if excluded_tools:
            entry["excluded_tools"] = sorted(excluded_tools)
        if excluded_middleware:
            entry["excluded_middleware"] = sorted(excluded_middleware)

    routing: dict[str, Any] = {"enabled": False, "tiers": []}
    try:
        from deep_agent.src.routing.config import load_router_config

        router_cfg = load_router_config()
        routing = {
            "enabled": bool(router_cfg.enabled),
            "strategy": router_cfg.strategy,
            "tiers": [
                {
                    "tier": tier.name,
                    "name": tier.model,
                    "provider": _provider_of(tier.model),
                    "criteria": tier.criteria,
                }
                for tier in router_cfg.tiers
            ],
        }
    except Exception:
        logger.debug("Could not read the router config", exc_info=True)

    return {"models": entries, "routing": routing}


@router.get("/safeguards")
async def list_safeguards() -> dict[str, Any]:
    """What is applied to a message besides the model.

    PII scrubbing, guardrails and the token budget all change what
    the model sees or what comes back, and all of them are silent:
    a redacted input looks like the text you typed, a filtered reply
    looks like an ordinary one, and a run stopped by the budget looks
    like a model with little to say.

    Ungated, like the rest of Components. "Is what I type being
    redacted before it reaches the model" is a question the person
    typing is entitled to have answered, and answering it only for
    operators is the same as not answering it.

    State and scope, never values: which entity classes are matched,
    not the patterns; that a guardrail model is configured, not where
    it lives.
    """
    from deep_agent.src.agent.config import agent_config

    # The typed accessors, not get_raw_section: pii lives in its own
    # file and guardrails and the budget are parsed into models, so
    # reading raw sections returned empty for all three and reported
    # "nothing is on" for a config where things were.
    def _read(fn: str, default: Any) -> Any:
        try:
            return getattr(agent_config, fn)()
        except Exception:
            logger.debug("Could not read config via %s", fn, exc_info=True)
            return default

    pii = _read("get_custom_pii_config", None)
    guardrail = _read("get_guardrails_config", None)
    budget = _read("get_token_budget_config", None)

    return {
        "pii": {
            "enabled": bool(getattr(pii, "enabled", False)),
            # What is matched, not how. A redaction pattern is a map
            # of what evades it.
            "entities": sorted(
                str(rule.name)
                for rule in (getattr(pii, "rules", None) or [])
                if rule.name
            ),
            # Applies to traces rather than to the model's input, and
            # the difference matters to anyone reading this.
            "trace_strategy": str(getattr(pii, "trace_strategy", "") or ""),
        },
        # Just the switch. A "model configured" flag looked useful
        # and would have been wrong: the loader drops `model` when
        # guardrails are disabled, so it reported "none configured"
        # for a config that names one.
        "guardrail": {"enabled": bool(getattr(guardrail, "enabled", False))},
        "token_budget": {"enabled": bool(getattr(budget, "enabled", False))},
    }


@router.get("/permissions")
async def list_permissions() -> dict[str, Any]:
    """What this agent is allowed to do.

    Three rules decide whether an action happens, and none of them
    were visible: whether a tool runs unattended, pauses for a
    decision or is refused; whether the agent is in dry run and
    therefore cannot act at all; and which paths it may read or
    write.

    Dry run is the one that most needs saying. Every tool that would
    have asked is refused instead, so the agent investigates and
    proposes and changes nothing — and from the outside that is
    indistinguishable from an agent whose tools are broken.

    Patterns and decisions only. No arguments, no paths outside the
    configured roots, nothing about what has actually been run.
    """
    from deep_agent.src.agent.config import agent_config

    def _read(fn: str) -> Any:
        try:
            return getattr(agent_config, fn)()
        except Exception:
            logger.debug("Could not read config via %s", fn, exc_info=True)
            return None

    middleware = _read("get_middleware_config")
    approval = getattr(getattr(middleware, "defaults", None), "human_approval", None)
    filesystem = _read("get_filesystem_config")

    # admin_routes already resolves this, including the fallback for
    # an invalid block. A second copy here is how the two come to
    # disagree about whether the agent can act.
    from deep_agent.aegra.admin_routes import _effects_mode

    try:
        effects = _effects_mode()
    except Exception:
        logger.debug("Could not read the effects mode", exc_info=True)
        effects = "live"

    policies = [
        {
            "tools": sorted(str(t) for t in (getattr(policy, "tools", None) or [])),
            "decision": str(getattr(policy, "decision", "ask")),
            "risk": str(getattr(policy, "risk", "medium")),
            "reason": str(getattr(policy, "reason", "") or ""),
        }
        for policy in (getattr(approval, "policies", None) or [])
    ]

    # `exclude` predates policies and still means "never prompt".
    # Presented as what it is rather than as a separate concept
    # nobody reading the page would know to ask about.
    never_asks = sorted(str(p) for p in (getattr(approval, "exclude", None) or []))

    rules = []
    for rule in getattr(filesystem, "permissions", None) or []:
        rules.append(
            {
                "operations": sorted(
                    str(o) for o in (getattr(rule, "operations", None) or [])
                ),
                "paths": sorted(str(p) for p in (getattr(rule, "paths", None) or [])),
                "mode": str(getattr(rule, "mode", "") or ""),
            }
        )

    return {
        "approval": {
            "enabled": bool(getattr(approval, "enabled", True)),
            "default_decision": str(getattr(approval, "default_decision", "ask")),
            "policies": policies,
            "never_asks": never_asks,
        },
        # Flat rather than nested under approval: dry run is enforced
        # through the approval policy but it is not an approval
        # setting, and burying it reads as a detail of one.
        "effects": effects,
        "dry_run": effects == "dry_run",
        "filesystem": {"rules": rules},
    }


@router.get("/auth/whoami")
async def whoami(request: Request) -> dict[str, Any]:
    """Who this credential belongs to, and whether it is privileged.

    Exists so a service that cannot validate a JWT itself can still
    authorise one. The UI's BFF is the case: it authenticates people
    by session cookie and has no JWKS client, so a CI script holding
    a perfectly good token had no way to reach its admin routes.
    Rather than put a second JWT verifier in the BFF — two
    implementations of the same check, drifting — it asks here.

    Authenticated but not developer-gated: the whole point is to
    *report* the role, so refusing non-developers would make it
    useless for deciding that they are not one.
    """
    from deep_agent.aegra.auth import ENABLE_AUTH
    from deep_agent.aegra.auth_helpers import authenticated_user_id
    from deep_agent.src.ldap import PRIVILEGED_ROLES, resolve_user_role

    user_id = await authenticated_user_id(request, reject_anonymous=ENABLE_AUTH)

    # With auth disabled there is no identity provider to resolve a
    # role against, and the dev user is the operator by definition —
    # `_build_dev_user` already grants it admin. Falling through to
    # the LDAP lookup instead returns "users", which is how the agent
    # came to disagree with the UI about who the single local user
    # is.
    if not ENABLE_AUTH:
        return {"user_id": user_id, "role": "owners", "is_developer": True}

    # Resolved against `preferred_username`, not the `sub` returned
    # above: LDAP has never heard of the UUID, so looking it up
    # reported every developer as an ordinary user.
    from deep_agent.aegra.auth_helpers import role_identity

    try:
        role = await resolve_user_role(await role_identity(request) or user_id)
    except Exception:
        logger.debug("Could not resolve role for %s", user_id, exc_info=True)
        role = None

    return {
        "user_id": user_id,
        "role": role,
        # None means the lookup failed. Callers treat anything other
        # than True as a refusal, so an outage denies rather than
        # grants.
        "is_developer": role in PRIVILEGED_ROLES if role is not None else None,
    }
