---
name: data-discovery
description: >
  Find a dataset, table or data product and establish whether it can be
  trusted and used — ownership, lineage, freshness and classification.
  Use for "where is the data for X", "who owns this table", "what feeds
  this dashboard", or before writing a query against an unfamiliar source.
---

# Data Discovery

**Purpose:** Answer "can I use this data, and should I" — not just
"does a table with this name exist".

## When to Use

- "Where do I find <metric / entity / report> data?"
- "Who owns this table?" "Is it still maintained?"
- "What breaks if I change this column?"
- Before querying a source the user has not used before

## Workflow

1. **Search by meaning, not by name.** `semantic_search_tool` in
   Atlan. People ask for "customer churn" and the asset is called
   `dim_acct_lifecycle_v3`; a name match finds nothing and the honest
   conclusion "it does not exist" would be wrong.
   `fluent_search` when you already know exact attributes to filter on.
2. **Resolve the asset.** `get_assets_tool` for the detail,
   `describe_asset_type_tool` when the asset kind itself is unfamiliar.
3. **Establish trust before reporting it as usable:**
   - **Owner** — a dataset with no owner is a risk, and saying so is
     part of the answer.
   - **Classification** — if it carries PII or restricted data, lead
     with that, not as a footnote.
   - **Freshness** — last updated. Stale data that looks current is
     the most expensive kind.
4. **Trace lineage** with `traverse_lineage_tool` when the question is
   impact ("what breaks if…") or provenance ("where does this come
   from"). Upstream for provenance, downstream for blast radius.
5. **Cross-check the data product** in Dataverse when the asset maps
   to one — it carries the domain, the owning team and the contract,
   which Atlan may not.

## Guardrails

- **Never present a search hit as the answer without checking
  ownership and freshness.** The failure mode is handing someone a
  plausible table that nobody maintains, which is worse than finding
  nothing: they will build on it.
- **PII and restricted classifications go first**, before the schema,
  before the sample. Someone skimming must not miss it.
- **Do not query the data to answer a question about the data.** Use
  the catalogue. `create_sql_insight` builds a query — it is for when
  the user explicitly wants one, not for satisfying a metadata
  question by sampling rows.
- **Report ambiguity rather than picking.** Three candidate tables
  means three, with what distinguishes them. Choosing silently makes
  the user's decision for them with less information than they have.
- **Say when lineage is incomplete.** Atlan knows what it was told;
  an unconnected pipeline leaves a gap, and "no downstream consumers"
  must never be reported when it means "none recorded".

## Output

```
**Asset.** <qualified name> (<type>)
**Classification.** <PII / restricted / public — state it even if public>
**Owner.** <team or person, or "unowned — treat with caution">
**Freshness.** <last updated>
**What it holds.** <one or two lines>
**Lineage.** <upstream → this → downstream, as far as recorded>
**Caveats.** <gaps, candidates not ruled out>
```
