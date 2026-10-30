---
name: catalog
description: >-
  Explore what data products exist and what they contain. Uses Atlan
  fluent_search and bundled config data files as reference.
metadata:
  author: imrankha
  version: "3.0.0"
---
# catalog

Do not call `task`. Do not call `execute_sql`.

## 1. Get the catalog

Use `read_config_data` with filename `product_catalog.yaml` to load the full
catalog of data products bundled with this agent.

If the user asks about specific table columns or schemas, also call
`read_config_data` with filename `product_tables.json` for detailed table
metadata.

For live Atlan metadata (freshest lineage, owners, tags), use `fluent_search`.

## 2. Answer the question

**"What data products exist?"** or **"What can I query?"**
- List products grouped by category. Include name, maturity, one-line description.
- Highlight `ai_ready` vs `ai_experimental` vs `not_specified`.

**"Tell me about [product]"** or **"What tables does [product] have?"**
- Find the product in the catalog via `read_config_data("product_catalog.yaml")`.
- For live table details, call `fluent_search` with `limit=20`:

```text
FluentSearch().where(CompoundQuery.active_assets()).where(CompoundQuery.asset_types([Table, View])).where(Asset.QUALIFIED_NAME.startswith("default/snowflake/1749560559/{PRODUCT}_DB/")).include_on_results(Asset.NAME).include_on_results(Table.DESCRIPTION).include_on_results(View.DESCRIPTION).include_on_results(Table.SCHEMA_NAME).include_on_results(View.SCHEMA_NAME)
```

If `fluent_search` fails, fall back to `read_config_data("product_tables.json")`
and show the tables for that product.

**"Which product has [topic] data?"**
- Search routing rules and use cases across all products.

## 3. Format

- Markdown tables or bullet lists.
- Group by category when listing many products.
- Always mention maturity level.
- If the user wants actual data values, tell them to ask a data query.
