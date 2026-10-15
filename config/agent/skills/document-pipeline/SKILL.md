---
name: document-pipeline
description: >
  Work with the Dataverse unstructured-data pipelines that turn documents
  into queryable data — list pipelines, fetch a processed document, and
  retrieve its chunks. Use for questions about pipeline status or about
  content already ingested by one.
---

# Document Pipeline

**Purpose:** Reach documents that have already been processed into the
platform, and report honestly on pipeline state.

## When to Use

- "What pipelines do I have?" "Did my document get processed?"
- "What's in <document> that the pipeline ingested?"
- Questions about content the user says was uploaded to Dataverse

**Not** for a file the user attached to this conversation. That is
already under `/uploads/` — read it with `read_file`. Routing an
attachment through here finds nothing and reports the document is
missing, which is both wrong and alarming.

## Workflow

1. `list_unstructured_data_pipelines_for_user` — establishes what
   exists and who owns it. Do this first even when the user names a
   pipeline, so a name that does not exist is caught immediately
   rather than inferred around.
2. `get_processed_document` — the processed form of a specific
   document.
3. `get_chunks_for_embeddings` — the chunked text, when the question
   needs the content rather than metadata about it.

## Guardrails

- **A missing document is not an empty document.** If the pipeline has
  not processed it, say it has not been processed. Returning "I found
  nothing in it" about a document that was never ingested is a wrong
  answer dressed as a complete one.
- **Chunks are fragments.** They are split for embedding, not for
  reading. Summarising from chunks loses whatever fell across a
  boundary — say so when the answer depends on completeness.
- **Report pipeline state plainly.** Running, failed and finished are
  different, and "the document isn't there yet" is a useful answer.
