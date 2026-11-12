"""Jinja2 rendering for system prompts.

Agent prompts (the orchestrator's ``PROMPT.md`` body, and any custom
template) may contain Jinja2 placeholders such as ``{{ current_date }}``
that are rendered with runtime values when the config loads.

Template resolution, in order:
    1. An explicit ``template_path`` argument.
    2. The ``SYSTEM_PROMPT_TEMPLATE`` environment variable (path to a
       custom ``.j2`` template that replaces the prompt body).
    3. The prompt body itself, rendered as a template.
    4. The bundled default template (``templates/system_prompt.j2``),
       used when the prompt body is empty.

Context variables provided by default (callers may add more, e.g. user
context, via ``extra_context``):
    current_date: Today's date in ISO format (YYYY-MM-DD).
    date: Alias for ``current_date``.
    agent_name: The agent's name from its config ("" if unknown).

Rendering never fails config loading: unknown variables render as
empty strings, and a template that cannot be parsed is returned
unrendered with a warning logged.
"""

import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

import jinja2

logger = logging.getLogger(__name__)

TEMPLATE_ENV_VAR = "SYSTEM_PROMPT_TEMPLATE"
DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "templates" / "system_prompt.j2"


def build_prompt_context(
    agent_name: str = "", extra_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the default Jinja2 context for system-prompt rendering."""
    today = date.today().isoformat()
    context: dict[str, Any] = {
        "current_date": today,
        "date": today,
        "agent_name": agent_name,
    }
    if extra_context:
        context.update(extra_context)
    return context


def _template_source(body: str, template_path: str | Path | None) -> str:
    """Resolve the template text to render (see module docstring)."""
    custom = template_path or os.environ.get(TEMPLATE_ENV_VAR)
    if custom:
        try:
            return Path(custom).read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(
                "Could not read system prompt template %s (%s); "
                "falling back to the prompt body/default template",
                custom,
                e,
            )
    if body.strip():
        return body
    return DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8")


def render_system_prompt(
    body: str,
    agent_name: str = "",
    extra_context: dict[str, Any] | None = None,
    template_path: str | Path | None = None,
) -> str:
    """Render a system prompt with Jinja2.

    Args:
        body: The prompt text (e.g. the ``body`` of PROMPT.md). May
            itself contain Jinja2 placeholders.
        agent_name: Value for the ``agent_name`` context variable.
        extra_context: Additional context variables (e.g. user
            context); these override the defaults on key conflicts.
        template_path: Path to a custom template file that replaces
            ``body``. Overrides the ``SYSTEM_PROMPT_TEMPLATE``
            environment variable when given.

    Returns:
        The rendered prompt. On any template error the unrendered
        source is returned instead, so a bad template can never break
        agent startup.
    """
    source = _template_source(body, template_path)
    context = build_prompt_context(agent_name, extra_context)
    try:
        template = jinja2.Environment(undefined=jinja2.Undefined).from_string(source)
        return str(template.render(context))
    except Exception as e:
        # Render-time failures (e.g. ZeroDivisionError from {{ 1 / 0 }})
        # are not TemplateError subclasses, but a bad template must
        # never break agent startup: fall back to the verbatim source.
        logger.warning(
            "System prompt template failed to render (%s); using it verbatim", e
        )
        return source
