"""What the agent should know about itself.

An agent that does not know its own limits spends turns discovering
them in front of a user: calling a tool its policy denies, trying to
write to a read-only volume, or promising an action it is in dry run
and cannot take. Each of those is a failed turn that reads as the
agent being unreliable rather than as it being constrained.

Derived from the same config the introspection routes serve, not
written by hand. A hand-maintained list of limits is wrong the first
time a policy changes, and wrong in the direction that matters: it
says the agent can do something it cannot.

One section, not several. The MCP status notice used to be its own
middleware appending its own block, which left the model two
descriptions of itself to reconcile; it is folded in here.

Deliberately *not* a second routing table. `PROMPT.md` already says
which subagent handles what, in an author's voice, and duplicating
that would be a third description. This carries only what config
knows and prose cannot keep up with: what exists right now, what is
reachable right now, and what is refused.
"""

from __future__ import annotations

from typing import Any

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _denied_patterns(approval: Any) -> list[str]:
    """Tool patterns the policy refuses outright."""
    out: list[str] = []
    for policy in getattr(approval, "policies", None) or []:
        if str(getattr(policy, "decision", "")) == "deny":
            out.extend(str(t) for t in (getattr(policy, "tools", None) or []))
    return sorted(set(out))


def _ask_patterns(approval: Any) -> list[str]:
    """Tool patterns that pause for a human."""
    out: list[str] = []
    for policy in getattr(approval, "policies", None) or []:
        if str(getattr(policy, "decision", "")) == "ask":
            out.extend(str(t) for t in (getattr(policy, "tools", None) or []))
    return sorted(set(out))


def _write_denied_paths(filesystem: Any) -> list[str]:
    out: list[str] = []
    for rule in getattr(filesystem, "permissions", None) or []:
        operations = {str(o) for o in (getattr(rule, "operations", None) or [])}
        if str(getattr(rule, "mode", "")) == "deny" and operations & {"write", "edit"}:
            out.extend(str(p) for p in (getattr(rule, "paths", None) or []))
    return sorted(set(out))


def _capabilities(agent_config: Any) -> list[str]:
    """Skills and subagents, by name and purpose."""
    lines: list[str] = []

    try:
        skills = agent_config.get_available_skills() or []
    except Exception:
        logger.debug("self-knowledge: could not read skills", exc_info=True)
        skills = []
    names = sorted(str(s) for s in skills)
    if names:
        lines.append(f"- Skills you can follow: {', '.join(names)}.")

    try:
        subagents = agent_config.get_all_subagent_configs() or {}
    except Exception:
        logger.debug("self-knowledge: could not read subagents", exc_info=True)
        subagents = {}
    for name in sorted(subagents):
        cfg = subagents.get(name) or {}
        reach = sorted(str(m) for m in (cfg.get("mcps") or []))
        # What it can reach, not what it is for: PROMPT.md says what
        # it is for, and this is the part that changes with config.
        detail = f" (reaches {', '.join(reach)})" if reach else ""
        lines.append(f"- You can delegate to `{name}`{detail}.")

    return lines


def build_self_description(
    model_name: str = "",
    connected: list[str] | None = None,
    unconnected: list[str] | None = None,
) -> str:
    """Who the agent is, what it can reach, and what it is refused.

    Empty when there is nothing to say — a heading followed by
    nothing trains the model to skip the section.
    """
    from deep_agent.src.agent.config import agent_config

    def _read(fn: str) -> Any:
        try:
            return getattr(agent_config, fn)()
        except Exception:
            logger.debug("self-knowledge: could not read %s", fn, exc_info=True)
            return None

    middleware = _read("get_middleware_config")
    approval = getattr(getattr(middleware, "defaults", None), "human_approval", None)
    filesystem = _read("get_filesystem_config")

    try:
        from deep_agent.aegra.admin_routes import _effects_mode

        dry_run = _effects_mode() == "dry_run"
    except Exception:
        logger.debug("self-knowledge: could not read the effects mode", exc_info=True)
        dry_run = False

    excluded_tools: list[str] = []
    if model_name:
        try:
            profiles = agent_config.get_providers_config().harness_profiles or {}
            profile = profiles.get(model_name)
            excluded_tools = sorted(
                str(t) for t in (getattr(profile, "excluded_tools", None) or [])
            )
        except Exception:
            logger.debug(
                "self-knowledge: could not read harness profiles", exc_info=True
            )

    can: list[str] = _capabilities(agent_config)

    # Tell it which model it is.
    #
    # The name was read here only to look up excluded tools, and
    # mentioned in the prompt only when there were any — so most of
    # the time the model was never told what it was running on.
    # Asked directly it answered from training data and said
    # "Gemini 3.7 Flash" while every call in the trace went to
    # gemini-3.8-flash. A model does not reliably know its own
    # version; the deployment does, and this is the file whose job
    # is telling it who it is.
    if model_name:
        can.insert(0, f"- You are running on the model `{model_name}`.")

    if connected:
        can.append(f"- Integrations you can use now: {', '.join(sorted(connected))}.")
        if any("(as " in entry for entry in connected):
            # The identity that matters per integration.
            #
            # The agent's own login is the account the user signed in
            # with here; an integration acts as whichever account was
            # connected to it, and the two are routinely different
            # people's identifiers for the same human. "Who is my
            # manager" resolved against the agent login answers for
            # the wrong directory entry without ever looking wrong.
            can.append(
                "- When you answer from an integration, 'the user' is the "
                "account shown in brackets for that integration, not the "
                "identity you were given for this session. Resolve 'me', "
                "'my' and 'mine' against that account, and say which "
                "account an answer is about when more than one could apply."
            )

    lines: list[str] = []

    if unconnected:
        # Ahead of the policy limits: an unconnected server is the
        # one the model is most likely to reach for and the only one
        # the user can fix themselves.
        #
        # This used to say "tell the user to connect it rather than
        # calling it". That was right when calling one only produced
        # an error. Calling one now pauses the run and puts a Connect
        # button in front of the user, so the old wording stopped the
        # model one step short of the thing that actually gets the
        # integration connected — it answered "please connect Atlan"
        # without ever touching the tool, and the pause never
        # happened.
        lines.append(
            "- Not connected yet: "
            f"{', '.join(sorted(unconnected))}. Do not refuse these and do "
            "not ask the user to go and connect them. If one is needed, call "
            "its tool as you normally would: that pauses the run, offers the "
            "user a Connect button, and resumes where it left off once they "
            f"connect. Only point at {settings.MCP_SETTINGS_PATH} if calling "
            "the tool tells you the user has already declined."
        )

    if dry_run:
        # First, and stated as a fact about this run rather than a
        # caution. In dry run every acting tool is refused, so an
        # agent that plans to act will fail every time it tries.
        lines.append(
            "- You are running in **dry run**. Any tool that would change "
            "something is refused. Investigate, explain and propose a plan; "
            "do not promise to carry it out."
        )

    denied = _denied_patterns(approval)
    if denied:
        lines.append(
            "- These tools are refused and will never run, so do not offer "
            f"them as an option: {', '.join(denied)}."
        )

    if excluded_tools:
        lines.append(
            f"- Unavailable with the model you are running on ({model_name}): "
            f"{', '.join(excluded_tools)}."
        )

    asks = _ask_patterns(approval)
    if asks:
        lines.append(
            "- These pause for a person to approve before running, so say "
            f"what you are about to do and why: {', '.join(asks)}."
        )

    write_denied = _write_denied_paths(filesystem)
    if write_denied:
        lines.append(
            f"- You can read but not write: {', '.join(write_denied)}. "
            "Do not offer to edit or delete anything there."
        )

    sections: list[str] = []
    if can:
        sections.append("## What you can do\n\n" + "\n".join(can))
    if lines:
        sections.append("## What you cannot do\n\n" + "\n".join(lines))

    return "\n\n".join(sections)
