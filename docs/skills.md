# Skills

A skill is a folder under `config/agent/skills/` holding a `SKILL.md`
and, usually, a `references/` directory. `SKILL.md` is loaded whenever
the skill is active; references cost nothing until something reads one.
That split is the whole design — `SKILL.md` is for *deciding and
routing*, references are for detail.

```
config/agent/skills/<skill-name>/
  SKILL.md              frontmatter + when to use + the rules
  references/*.md       detail, read on demand
```

The `name:` in frontmatter must equal the directory name. A mismatch is
not an error — the loader silently ignores the skill, which is the worst
failure mode available: no skill, no message.

Skills are mounted read-only (`"/skills/": filesystem_readonly` in
`config/agent/runtime/agent.yaml`) and enabled by the `skills`
middleware.

## Which builder

| Corpus | Use | Runs |
|---|---|---|
| Prose — books, docs, ADRs, runbooks, policies | `book-to-skill` | Developer laptop |
| Tabular — CSV, TSV, JSON/JSONL, Parquet, XLSX | `make dataset-skill` | This repo |

### Prose corpora — book-to-skill

Use [`book-to-skill`](../../../misc/planning/templates/extras/book-to-skill)
(MIT). It is itself a skill, executed by a host agent, and it produces a
richer structure than per-source references:

```
SKILL.md + chapters/ + glossary.md + patterns.md + cheatsheet.md
```

It needs a real shell for `extract.py` and a real filesystem for
`~/.agents/skills/`, so it runs on a laptop and the result is committed
here — not inside the deployed pod.

Don't rebuild this. A prose distiller already exists; a second one in
this repo would be a worse copy.

### Datasets — make dataset-skill

```sh
make dataset-skill NAME="sales-data" SRC=./data/sales
```

Profiles every tabular file in `SRC` and writes
`config/agent/skills/sales-data/`. `FORCE=1` overwrites an existing one.

Schema belongs in a skill rather than in retrieval: an agent needs
column names, units and allowed values *before* it can form a query at
all. Retrieval answers questions you already know how to ask.

**Every figure is measured.** `deep_agent/src/knowledge/dataset_profile.py`
computes row counts, null rates, ranges and vocabularies with pyarrow;
the model is only allowed to narrate them. A model asked to read rows
invents a plausible distribution, and a data dictionary that misstates a
column is wrong in a way nobody checks.

Consequences worth knowing:

- Vocabularies are listed only for textual columns with ≤ 25 distinct
  values. Three distinct numbers in a sample is not a constraint.
- Ranges appear only for ordered types. Min/max on free text is
  alphabetical noise.
- Files over 50,000 rows are sampled, and the skill says so.
- XLSX: first sheet only. Several unrelated sheets are several datasets.
- Join keys are **matched by column name only** and labelled as
  candidates. Confirming a real relationship needs the values, and a
  guessed join is a query that silently returns the wrong rows.

Check the generated `SKILL.md` before committing — the figures are
trustworthy, the narration around them is not reviewed by anything.

## Why the agent does not author skills at runtime

There was a `create_skill` tool. It was removed.

Skills load from a read-only backend rooted at committed config. An
agent that could write one would be granting itself always-loaded
instructions from content a user supplied — a prompt-injection
amplifier, and persistent across every later conversation. Document
content reaches the agent through attachments and `search_knowledge`,
where it stays data.

Authoring a skill is a build-time act with a diff and a reviewer.
