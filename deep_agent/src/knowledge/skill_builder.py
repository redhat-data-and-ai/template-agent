"""Distil a document corpus into an agent skill.

Dumping raw documents into a skill does not work: it flattens the
hierarchy, blows the context budget, and the agent has to re-read
everything on every turn to find the one rule that mattered. Condensing
first is what makes a corpus usable as a skill.

The output follows the agentskills.io layout:

    <skill-name>/
      SKILL.md              frontmatter + when to use + the rules
      references/*.md       detail, read on demand

The split is the point. `SKILL.md` is loaded whenever the skill is
active, so it holds only what the agent needs to *decide* and route;
references hold the detail and cost nothing until something reads them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# SKILL.md is resident context whenever the skill is active. Past this
# it stops being an index and becomes the thing it was meant to index.
MAX_SKILL_CHARS = 6000

# A reference is read on demand, so it can be longer — but not so long
# that reading one swamps the conversation.
MAX_REFERENCE_CHARS = 12000


@dataclass
class SourceDoc:
    """One input document."""

    name: str
    text: str


@dataclass
class BuiltSkill:
    """What to write to disk."""

    name: str
    skill_md: str
    references: dict[str, str]


def slugify(value: str) -> str:
    """A directory- and frontmatter-safe skill name."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", (value or "").lower()).strip("-")
    return slug[:48] or "corpus"


_REFERENCE_PROMPT = """\
Condense the document below into a reference sheet for an AI agent.

Keep: rules, thresholds, numbers, names, dates, exceptions, and
procedures. Drop: preamble, motivation, history, anything that does not
change what an agent would do or say.

Write it as markdown with short sections and tables where the source
has tabular facts. Do not add anything the document does not say. Do
not summarise in prose — an agent reads this to look something up, not
to understand the narrative.

Document: {name}

---
{text}
"""

_SKILL_PROMPT = """\
Write the SKILL.md for an agent skill built from the sources below.

Output exactly this structure and nothing else:

---
name: {slug}
description: >
  ONE paragraph. State what the skill covers and when to use it, then
  an explicit "Do not use for ..." clause. This text is how an agent
  decides whether to load the skill, so it must be about *routing*,
  not about the subject matter.
---

# {title}

## When to Use

Two or three bullets, each a concrete user intent.

## Key Facts

The handful of rules, thresholds or figures an agent needs most often,
inline so they cost nothing to reach. Tables where appropriate.

## References

A bullet per reference file, each saying what question it answers:
{reference_list}

## Boundaries

What this corpus does NOT cover, so the agent declines instead of
inventing.

Rules: under {max_chars} characters. No invented facts. If the sources
disagree, say so rather than picking one. Plain markdown only — no
LaTeX, no MathJax: write "<=" rather than a math expression, because
this renders in a terminal and a chat bubble, not a paper.

The reference sheets:

{summaries}
"""


def build_prompt_for_reference(doc: SourceDoc) -> str:
    """Prompt asking the model to turn one source document into a reference."""
    return _REFERENCE_PROMPT.format(name=doc.name, text=doc.text)


def build_prompt_for_skill(slug: str, title: str, references: dict[str, str]) -> str:
    """Prompt asking the model to write SKILL.md over a set of references."""
    reference_list = "\n".join(f"- references/{n}" for n in sorted(references))
    # Reference bodies are truncated here only to bound the prompt; the
    # files written to disk keep their full text.
    summaries = "\n\n".join(
        f"## references/{name}\n{body[:3000]}"
        for name, body in sorted(references.items())
    )
    return _SKILL_PROMPT.format(
        slug=slug,
        title=title,
        reference_list=reference_list,
        max_chars=MAX_SKILL_CHARS,
        summaries=summaries,
    )


def reference_filename(doc_name: str) -> str:
    """Slugified `.md` filename for a source document."""
    stem = slugify(doc_name.rsplit(".", 1)[0])
    return f"{stem}.md"


def strip_code_fence(text: str) -> str:
    """Remove a wrapping ```markdown fence if the model added one."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3:
        return stripped
    body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
    return "\n".join(body).strip()


def ensure_frontmatter(text: str, slug: str, description: str) -> str:
    """Guarantee valid frontmatter with the right name.

    The model is asked for it, and usually complies — but a skill whose
    frontmatter is missing or whose name does not match its directory
    is silently ignored by the loader, which is the worst failure mode
    available: no error, no skill.
    """
    text = strip_code_fence(text)

    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if match:
        front, body = match.group(1), match.group(2)
        if re.search(rf"^name:\s*{re.escape(slug)}\s*$", front, re.M):
            return text
        front = re.sub(r"^name:.*$", f"name: {slug}", front, count=1, flags=re.M)
        if "name:" not in front:
            front = f"name: {slug}\n{front}"
        return f"---\n{front}\n---\n{body}"

    return f"---\nname: {slug}\ndescription: >\n  {description}\n---\n\n{text}"


# ── Datasets ─────────────────────────────────────────────────────────

_DATASET_SKILL_PROMPT = """\
Write the SKILL.md for an agent skill describing a dataset.

The profiles below are **measured facts**. Copy figures from them
exactly. Do not estimate, round, extrapolate, or state anything about
the data that is not in the profiles — a data dictionary that misstates
a column is wrong in a way nobody checks.

Output exactly this structure and nothing else:

---
name: {slug}
description: >
  ONE paragraph: what this dataset contains and when an agent should
  consult it, then an explicit "Do not use for ..." clause. This is how
  an agent decides whether to load the skill, so write about routing,
  not about the data.
---

# {title}

## When to Use

Two or three concrete question types this dataset can answer.

## Datasets

A table: file, rows, and what one row represents. Infer "what a row
represents" from the column names only, and say "unclear" if it is.

## Key Columns

The columns an agent needs to know about without opening a reference:
identifiers, the main measures, and any column whose allowed values
are small enough to matter. Give the allowed values verbatim.

{join_section}

## Caveats

Null-heavy columns, sampled files, and anything in the profiles that
would mislead someone writing a query. Only what the profiles show.

## References

One bullet per reference file:
{reference_list}

## Boundaries

What these datasets do NOT contain, so the agent declines instead of
guessing. Derive this from what the columns are missing.

Rules: under {max_chars} characters. Plain markdown only — no LaTeX.

The measured profiles:

{profiles}
"""


def build_prompt_for_dataset_skill(
    slug: str,
    title: str,
    references: dict[str, str],
    join_keys: list[tuple[str, list[str]]] | None = None,
) -> str:
    """Prompt for a data-dictionary skill, grounded in measured profiles."""
    reference_list = "\n".join(f"- references/{n}" for n in sorted(references))

    if join_keys:
        shared = "\n".join(
            f"- `{name}` appears in: {', '.join(where)}" for name, where in join_keys
        )
        # The instruction and the data must be separated. Inlining
        # "state that these are candidates…" into the section put the
        # instruction itself into the generated skill, verbatim.
        join_section = (
            "## Joining\n\nList these shared columns, and say in one line "
            "that they are matched by name only and not verified "
            "relationships. Do not repeat this instruction.\n\n"
            "Shared columns:\n" + shared + "\n"
        )
    else:
        join_section = ""

    profiles = "\n\n".join(
        f"### {name}\n{body}" for name, body in sorted(references.items())
    )
    return _DATASET_SKILL_PROMPT.format(
        slug=slug,
        title=title,
        join_section=join_section,
        reference_list=reference_list,
        max_chars=MAX_SKILL_CHARS,
        profiles=profiles,
    )
