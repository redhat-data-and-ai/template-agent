"""Startup-time safety scan over catalogue-sourced subagent and skill content.

Subagent descriptions/bodies and skill ``SKILL.md`` content originate in the
shared package catalogue and may be authored by a different user or org than
the one deploying this agent (see OFFSEC-379: "The catalogue is a cross
tenant prompt injection distribution channel"). Unlike user input, tool
results, LLM output, and memory writes, this content was never passed
through Granite Guardian before entering the agent's context — it is
trusted purely because it passed structural validation at publish time in
the registry.

This module runs the existing Guardian ``check_safety`` / ``check_injection``
checks (see ``deep_agent.src.guardrails.client``) over that content once, at
startup, and excludes — rather than blocks the whole agent on — any subagent
or skill whose metadata is flagged unsafe or as a prompt-injection attempt.
This preserves availability: a single compromised or malicious catalogue
package cannot take down an agent that otherwise depends on it.

Functions:
    scan_catalogue_safety: Scan all loaded subagents/skills and exclude unsafe ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deep_agent.src.guardrails import get_guardrails_config
from deep_agent.utils.pylogger import get_python_logger

from .parser import parse_frontmatter

if TYPE_CHECKING:
    from .loader import AgentConfig

logger = get_python_logger()


async def _check_content_safety(content: str, context: str) -> tuple[bool, str, str]:
    """Run the harm and injection checks for one piece of catalogue content.

    Args:
        content: The text to scan (e.g. a subagent's description + body).
        context: Label passed through to Guardian logging for traceability.

    Returns:
        A tuple of ``(is_safe, failed_check, verdict)``. ``failed_check`` and
        ``verdict`` are empty strings when ``is_safe`` is True.
    """
    from deep_agent.src.guardrails.client import check_injection, check_safety

    if not content.strip():
        return True, "", ""

    is_safe, verdict = await check_safety(content, context=context)
    if not is_safe:
        return False, "safety", verdict

    is_safe, verdict = await check_injection(content, context=context)
    if not is_safe:
        return False, "injection", verdict

    return True, "", ""


async def _scan_subagents(config: "AgentConfig") -> list[str]:
    """Scan every loaded subagent's description + body. Returns excluded names."""
    excluded: list[str] = []

    for name, cfg in list(config.get_all_subagent_configs().items()):
        text = "\n\n".join(
            part for part in (cfg.get("description", ""), cfg.get("body", "")) if part
        )
        is_safe, failed_check, verdict = await _check_content_safety(
            text, context=f"subagent_config:{name}"
        )
        if is_safe:
            continue

        logger.warning(
            "catalogue_content_unsafe",
            kind="subagent",
            name=name,
            check=failed_check,
            verdict=verdict,
        )
        config.exclude_subagent(name, reason=f"{failed_check} check flagged: {verdict}")
        excluded.append(name)

    return excluded


async def _scan_skills(config: "AgentConfig") -> list[str]:
    """Scan every loaded skill's SKILL.md description + body. Returns excluded names."""
    excluded: list[str] = []

    for name, path in list(config.get_available_skills().items()):
        skill_md = path / "SKILL.md"
        if not skill_md.is_file():
            # Nothing to scan (e.g. malformed/empty skill dir) — leave as-is,
            # the existing skill loading path already tolerates this.
            continue

        try:
            skill_cfg: dict[str, Any] = parse_frontmatter(skill_md)
        except Exception as exc:
            # Fail closed: a SKILL.md that can't be parsed can't be scanned,
            # but the directory-based skill index (_scan_available_skills)
            # loads it regardless of parse success — unlike subagents, which
            # are dropped entirely on a parse failure (_load_all_subagents).
            # Leaving it available here would let a deliberately malformed
            # frontmatter (with a malicious body) bypass this scan entirely.
            logger.warning(
                "catalogue_content_unscannable",
                kind="skill",
                name=name,
                error=str(exc),
            )
            config.exclude_skill(name, reason=f"SKILL.md failed to parse: {exc}")
            excluded.append(name)
            continue

        text = "\n\n".join(
            part
            for part in (skill_cfg.get("description", ""), skill_cfg.get("body", ""))
            if part
        )
        is_safe, failed_check, verdict = await _check_content_safety(
            text, context=f"skill_config:{name}"
        )
        if is_safe:
            continue

        logger.warning(
            "catalogue_content_unsafe",
            kind="skill",
            name=name,
            check=failed_check,
            verdict=verdict,
        )
        config.exclude_skill(name, reason=f"{failed_check} check flagged: {verdict}")
        excluded.append(name)

    return excluded


async def scan_catalogue_safety(config: "AgentConfig") -> dict[str, list[str]]:
    """Scan all loaded subagent and skill metadata for unsafe/injected content.

    No-op when guardrails are disabled or not yet initialised — mirrors the
    short-circuit in ``deep_agent.src.guardrails.client._call_guardian``, so
    this scan never fails startup or blocks when Guardian is not configured.

    Args:
        config: The loaded ``AgentConfig`` singleton to scan and mutate.

    Returns:
        Summary dict with ``subagents_excluded`` and ``skills_excluded`` name lists.
    """
    if get_guardrails_config() is None:
        logger.info("catalogue_safety_scan_skipped", reason="guardrails_disabled")
        return {"subagents_excluded": [], "skills_excluded": []}

    subagents_excluded = await _scan_subagents(config)
    skills_excluded = await _scan_skills(config)

    logger.info(
        "catalogue_safety_scan_complete",
        subagents_excluded=len(subagents_excluded),
        skills_excluded=len(skills_excluded),
    )

    return {
        "subagents_excluded": subagents_excluded,
        "skills_excluded": skills_excluded,
    }
