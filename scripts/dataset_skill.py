"""Turn a dataset into an agent skill — a data dictionary it can use.

    make dataset-skill NAME="sales" SRC=./data/sales

Profiles every tabular file in a directory (CSV, TSV, JSON, JSONL,
Parquet, XLSX) and writes a skill under `config/agent/skills/`.

Why a skill rather than retrieval. Schema is not something to look up
per question — an agent needs the column names, units and allowed
values *before* it can form a query at all. Retrieval answers
questions you already know how to ask; a data dictionary is what lets
you ask them.

Every figure in the output is measured, not inferred. The profiler
computes ranges, null rates and vocabularies; the model is only
allowed to narrate them. For prose corpora use book-to-skill instead —
see docs/skills.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / "config" / "agent" / "skills"

# Past this the prompt stops fitting and the skill gets vague.
MAX_DATASETS = 20


def _model_name() -> str:
    try:
        from deep_agent.src.agent.config import agent_config

        configured = (agent_config.get_orchestrator_config() or {}).get("model")
        if isinstance(configured, dict):
            return str(configured.get("name") or "gemini-3.8-flash")
        return str(configured or "gemini-3.8-flash")
    except Exception:
        return "gemini-3.8-flash"


def main() -> int:
    """Build a skill from a dataset. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Skill name, e.g. sales-data")
    parser.add_argument(
        "--src", type=Path, required=True, help="Directory of data files"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing skill"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the skill. Defaults to config/agent/skills/.",
    )
    args = parser.parse_args()

    if not args.src.is_dir():
        parser.error(f"{args.src} is not a directory")

    from deep_agent.src.knowledge.dataset_profile import (
        find_join_keys,
        profile_bytes,
        profile_to_markdown,
    )
    from deep_agent.src.knowledge.skill_builder import (
        build_prompt_for_dataset_skill,
        ensure_frontmatter,
        reference_filename,
        slugify,
    )

    profiles = []
    for path in sorted(args.src.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        profile = profile_bytes(path.name, path.read_bytes())
        if profile:
            print(
                f"  profiled {path.name}: {profile.row_count:,} rows, "
                f"{len(profile.columns)} columns"
            )
            profiles.append(profile)
        else:
            print(f"  skipped {path.name} (not tabular)")

    if not profiles:
        print("No tabular files found.")
        return 1

    profiles = profiles[:MAX_DATASETS]
    # Deduplicated, because `reference_filename` drops the extension:
    # `sales.csv` and `sales.parquet` both map to `sales.md`, the dict
    # kept whichever came last, and the other dataset's reference
    # vanished with no warning.
    references: dict[str, str] = {}
    for profile in profiles:
        name = reference_filename(profile.name)
        if name in references:
            stem, _, ext = name.rpartition(".")
            suffix = slugify(profile.name.rsplit(".", 1)[-1])
            name = f"{stem}-{suffix}.{ext}"
            counter = 2
            while name in references:
                name = f"{stem}-{suffix}-{counter}.{ext}"
                counter += 1
        references[name] = profile_to_markdown(profile)
    joins = find_join_keys(profiles)
    if joins:
        print(f"  shared column names: {', '.join(n for n, _ in joins)}")

    slug = slugify(args.name)
    out = (args.out or SKILLS_DIR) / slug
    if out.exists() and not args.force:
        print(f"{out} already exists. Use --force to overwrite.")
        return 1

    from deep_agent.src.agent.llm import create_model

    print("  writing SKILL.md")
    model = create_model(_model_name(), temperature=0.0, max_output_tokens=8192)
    result = model.invoke(
        build_prompt_for_dataset_skill(slug, args.name, references, joins)
    )
    content = result.content
    if isinstance(content, list):
        content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))

    skill_md = ensure_frontmatter(
        str(content or "").strip(),
        slug,
        f"Data dictionary for {len(profiles)} dataset(s).",
    )

    (out / "references").mkdir(parents=True, exist_ok=True)
    (out / "SKILL.md").write_text(skill_md)
    for filename, body in references.items():
        (out / "references" / filename).write_text(body)

    try:
        shown = out.relative_to(REPO_ROOT)
    except ValueError:
        # Outside the repo, which is the normal case for a PVC.
        shown = out
    print(f"\nWrote {shown}")
    print(f"List `{slug}` under `skills:` in PROMPT.md, then `make validate-config`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
