---
name: data-products
description: >
  Work with Dataverse data products — what exists, which domain owns
  one, what it promises. Use for "what data products are there",
  ownership and domain questions, or checking a product's contract
  before depending on it.
---

# Data Products

**Purpose:** Answer questions about Dataverse data products as
*products* — owned, versioned, with a contract — rather than as
tables.

Pairs with **data-discovery**, which searches the Atlan catalogue for
assets. Use that to find a table; use this when the unit is a data
product and the question is about ownership or the contract.

## Tool discovery

Tool names are not listed here. This server's surface was not
reachable when the skill was written, and naming a tool that does not
exist is worse than naming none — the model calls it, gets nothing,
and reports a result. Use what you find under the `dataverse`
prefix.

## Workflow

1. **List before assuming.** Product names in conversation rarely
   match the registered name.
2. **Ownership is the first thing to report.** A data product
   without a named owning domain is a risk to depend on, and saying
   so is part of the answer rather than a caveat at the end.
3. **Read the contract, not just the description.** Schema,
   guarantees, refresh cadence. "It has customer data" does not tell
   someone whether they can build on it.
4. **Check freshness.** A product that has not refreshed on schedule
   is a different answer from one that has.

## Guardrails

- **Read-only.** Do not create, register, modify or deprecate a data
  product. These are published contracts with consumers attached.
- **Do not describe a product you did not read.** Dataverse,
  Snowflake and Atlan all hold overlapping metadata; answer from the
  one you actually queried and say which.
- **Report ambiguity rather than choosing.** Two products with
  similar names means two, with what distinguishes them.
- **"No consumers listed" is not "no consumers".** The registry knows
  what was registered.
