---
name: unstructured-query
description: >-
  Deep document search via Pathfinder RAG. Produces detailed answers
  with inline citations, page references, and structured source tables.
metadata:
  author: imrankha
  version: "4.0.0"
---
# unstructured-query (Pathfinder)

Do not call `task`, `execute_sql`, or `get_sql`.

## Pipeline

### 1. Search (pick the right tool)

| Tool | Best for | When |
|---|---|---|
| `search(query, top_k=12)` | Default, general | Most questions |
| `search_semantic(query, top_k=12)` | Concepts, themes, strategy | "What is the strategy for...", "How does..." |
| `search_keyword(query, top_k=12)` | Exact values, names, numbers | "$1.2B", "MCT4316", "Jennifer Meyers" |

Reformulate the query to match document language. For earnings calls, include
the quarter and year. For policies, include the policy domain.

### 2. Get chunk details (PARALLEL calls)

Call `get_chunk(chunk_id)` for the TOP 3-5 relevant chunks.
Call ALL get_chunk calls in ONE turn, not sequentially.

Each chunk returns:
- `body`: full text content
- `crop_path`: image crop of the source page region
- `page_number`: exact page
- `doc_title`: document name

If a chunk needs more context, call `get_page(doc_id, page_number)`.

### 3. Write a DETAILED answer

DO NOT summarize. Write a thorough, multi-paragraph answer that:
- Extracts specific facts, numbers, percentages, quotes
- Draws connections across multiple documents
- Provides context and analysis
- Cites every claim inline: (Document Title, p.N)

Example:
```
Red Hat's hybrid cloud strategy positions OpenShift as the core platform for
enterprise Kubernetes, with a focus on multi-cloud portability and AI-native
workloads (Strategy on a slide, p.2). The company reported $1.2B in ARR from
OpenShift in Q3 FY25, representing 23% year-over-year growth (Q3 2024 Earnings
Call, p.5). This growth is driven by hyperscaler partnerships, which now account
for 40% of the Consulting book of business (Market Insights Compendium, p.8).
```

### 4. Source table (MANDATORY)

ALWAYS end with this format. NEVER use bullet points for sources.

```markdown
## Sources

| # | Document | Pages | Key Content |
|---|----------|-------|-------------|
| 1 | Q3 2024 Earnings Call | p. 5, 11 | OpenShift ARR figures, AI revenue commentary |
| 2 | Strategy on a slide_internal_2026 | p. 2 | Platform positioning, multi-cloud strategy |
| 3 | Market Insights Compendium | p. 8-9 | Hyperscaler revenue share, partnership growth |
```

### 5. Available documents (when asked)

Call `list_documents()` and present as a table:

```markdown
| # | Document | Pages | Category |
|---|----------|-------|----------|
| 1 | Q3 2024 Earnings Call | 14 | Earnings |
| 2 | AI Governance Policy | 4 | Governance |
```

## Quality bar

- Minimum 3 sources cited per answer
- Every factual claim has an inline citation
- Source table has document name + page + what it contributes
- Answer is 3+ paragraphs for substantive questions
- Never say "the document mentions..." - state the facts directly
