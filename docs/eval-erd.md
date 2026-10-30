# Evaluation Tables - ERD

## Overview

The evaluation system uses four Postgres tables to store test datasets, eval run metadata, and per-turn evaluation results.

## Entity Relationship Diagram

```
+----------------------------------+
|       eval_dataset_items         |
|       (one row per case)         |
+----------------------------------+
| PK  id            SERIAL         |
| UQ  case_id       TEXT NOT NULL  |
|     case_data     JSONB NOT NULL |
|     judge_model   TEXT           |
|     created_by    TEXT           |
|     updated_by    TEXT           |
|     created_at    TIMESTAMPTZ    |
|     updated_at    TIMESTAMPTZ    |
+----------------------------------+
        |
        |  dataset defines test cases
        |  that produce eval runs
        v
+----------------------------------+
|             evals                |
|      (one row per eval run)      |
+----------------------------------+
| PK  id              SERIAL      |
| IX  config_hash     TEXT NOT NULL|
| IX  eval_status     TEXT NOT NULL|
|     eval_score      FLOAT        |
|     ls_run_ids      TEXT[]       |-------+
|     pass            INTEGER      |       |
|     fail            INTEGER      |       |
|     error           INTEGER      |       |
|     judge_model     TEXT         |       |
|     results_detail  JSONB        |       |
|     force_reeval    BOOLEAN      |       |
|     created_at      TIMESTAMPTZ  |       |
|     updated_at      TIMESTAMPTZ  |       |
|     completed_at    TIMESTAMPTZ  |       |
+----------------------------------+       |
                                           |
                              ls_run_ids[] --> run_id
                                           |
+----------------------------------+       |
|      evaluation_results          |       |
|  (one row per metric per turn)   |       |
+----------------------------------+       |
| PK  id                     SERIAL|       |
| IX  run_id              VARCHAR  |<------+
|     timestamp           TIMESTAMP|
| IX  conversation_group_id VARCHAR|
|     tag                 VARCHAR  |
|     turn_id             VARCHAR  |
| IX  metric_identifier   VARCHAR  |
|     metric_metadata     TEXT     |
|     result              VARCHAR  |
|     score               FLOAT   |
|     threshold            FLOAT  |
|     reason               TEXT   |
|     query                TEXT   |
|     response             TEXT   |
|     execution_time       FLOAT  |
|     evaluation_latency   FLOAT  |
|     api_input_tokens    INTEGER |
|     api_output_tokens   INTEGER |
|     judge_llm_input_tokens  INT |
|     judge_llm_output_tokens INT |
|     embedding_tokens    INTEGER |
|     judge_scores         TEXT   |
|     time_to_first_token  FLOAT  |
|     streaming_duration   FLOAT  |
|     agent_latency        FLOAT  |
|     tokens_per_second    FLOAT  |
|     tool_calls           TEXT   |
|     contexts             TEXT   |
|     expected_response    TEXT   |
|     expected_intent      TEXT   |
|     expected_keywords    TEXT   |
|     expected_tool_calls  TEXT   |
+----------------------------------+


+----------------------------------+
|     eval_datasets (LEGACY)       |
|    kept for rollback only        |
+----------------------------------+
| PK  id            SERIAL         |
|     dataset       JSONB NOT NULL |
|     judge_model   TEXT           |
|     created_at    TIMESTAMPTZ    |
+----------------------------------+
```

## Relationships

| From | To | Type | Description |
|------|-----|------|-------------|
| `eval_dataset_items` | `evals` | Logical | Dataset defines what gets evaluated. Cache invalidation compares `MAX(eval_dataset_items.updated_at)` vs `evals.completed_at`. |
| `evals.ls_run_ids[]` | `evaluation_results.run_id` | Array FK | Links an eval run to its per-turn metric results. Used to fetch turn-level detail on demand. |

## Tables

### eval_dataset_items

Stores one row per eval test case (datapoint). Replaced the legacy `eval_datasets` single-blob design.

- **case_id**: unique identifier from the UI (the case's `id` field)
- **case_data**: full case JSON including name, tag, description, and turns
- **judge_model**: denormalized from dataset-level setting
- **created_by / updated_by**: user identity from JWT for audit tracking

### evals

Tracks eval run lifecycle and aggregated results.

- **config_hash**: 16-char SHA256 prefix of agent config files (prompts, skills, tools)
- **eval_status**: `not_started` | `in_progress` | `completed` | `error` | `no_dataset`
- **results_detail**: JSONB with summary stats only (turns stripped to avoid bloat at scale)
- **ls_run_ids**: array of LangSmith run IDs produced during the eval

### evaluation_results

Per-turn, per-metric evaluation results written by the eval-runner.

- **run_id**: groups all results from a single eval run
- **conversation_group_id**: maps back to the test case name
- **metric_identifier**: e.g. `custom:answer_correctness`, `geval:tone_safety`
- **result**: `PASS` | `FAIL` | `ERROR` | `UNKNOWN`

### eval_datasets (LEGACY)

Old single-blob table. No longer read or written by active code paths. Kept so the one-time startup migration can copy existing data into `eval_dataset_items`. Safe to drop after validation in production.

## Data Flow

```
1. UI saves cases ---------> eval_dataset_items (one row per case)

2. Trigger eval -----------> evals row created (status: in_progress)

3. Eval-runner runs cases -> evaluation_results (one row per metric per turn)

4. Eval-runner completes --> evals row updated (status: completed,
                             results_detail: summary only, turns stripped)

5. UI fetches results -----> evals row (summary)
                             + evaluation_results (turns fetched on demand
                               via ls_run_ids, lean column set only)
```
