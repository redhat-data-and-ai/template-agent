---
name: data-query
description: >-
  2-call pipeline. Router pre-classifies product AND provides tables.
  Agent calls get_sql + execute_sql. Nothing else needed.
metadata:
  author: imrankha
  version: "6.0.0"
---
# data-query

The DiffusionGemma router has already classified the query and attached
the product name AND its tables. Everything is in the classification context above.

Do not call `identify_dataproducts`, `shortlist_tables`, or `read_file`.
Never write SQL yourself. Always use `get_sql`.

## Pipeline: 2 calls

1. `get_sql(data_product, tables_list, user_query)` - generate SQL.
2. `execute_sql(sql)` - run the SQL.

## Step 1: Call get_sql

Use the product and tables_list exactly as provided in the classification:

- `data_product`: product name from classification (e.g., "roverpeople")
- `tables_list`: the tables array from classification, as `list[dict]`:
  ```json
  [{"name": "ROVER_PEOPLE_CURR", "schema": "RHAI_MARTS", "description": "..."}]
  ```
- `user_query`: the user's question. For "my" questions, append `(User ID = 'imrankha')`.

`get_sql` handles Atlan metadata, GitLab guidance, SQL generation, and validation internally.
Do NOT modify the returned SQL.

## Step 2: Execute

Call `execute_sql(sql="<exact SQL from get_sql>")`.

## Empty results

If `row_count: 0`:
1. `execute_sql` with `EXPLAIN SELECT * FROM <table> LIMIT 1` to check RLS.
2. If no RLS, call `get_sql` again with corrective guidance. Max 1 retry.
3. If RLS + COUNT = 0, tell user about access restrictions.

## Follow-ups

Reuse product and tables from prior turn. Call `get_sql` directly.
