# Phase 1 input guardrails

Host-agnostic input gate for tool-using agents. The package does not import an
application, a vector store, or a chat UI. Copy this directory out of
template-agent and publish it on its own (`agent-guardrails`).

The design follows `docs/Agent_Guardrails.pdf`, Phase 1 only. Tool-argument
checks and output checks are later phases and are not in this package.

## What the gate does

`InputGate.evaluate` runs, in order:

1. **Rules.** Maximum length (8000 characters) and an optional character pattern.
2. **Schema.** When allowed or required fields are configured, a JSON object is
   checked for unknown and missing fields. Free text is not a structured query.
3. **Prompt injection.** User message only. Score at or above 0.5 blocks.
4. **Content filter.** Hate, abuse, and profanity. Score at or above 0.5 blocks.
5. **Safety vote.** Violence, sexual content, self-harm, and crime. Either this
   vote or the content filter can block.
6. **PII.** Mask and continue. The agent sees the masked text.
7. **Secrets.** Mask API keys, tokens, and private-key blocks, then continue.
8. **Intent.** A fitted logistic-regression head. Confidence below 0.6, or a
   label with no tool map, restricts the turn to `read_only_tools`. An unfitted
   head does not narrow tools.

A block sets `retryable` to false. The gate does not pause for a person.

The handoff is `decision.text`, `decision.intent`, and `decision.allowed_tools`.
`decision.tool_allowed(name)` is how the host rejects an intent-incompatible tool
before that tool runs. Passing the allowlist into a later tool-argument phase
is left to the host.

## Install

```bash
pip install -e .
```

From template-agent the package is on the test path via `guardrails/src`.
The runtime image copies it onto `PYTHONPATH`.

## Use

```python
from agent_guardrails import InputGate, InputGateConfig
from agent_guardrails.checks.intent import LogisticIntentClassifier

classifier = LogisticIntentClassifier()
classifier.fit(["show revenue by region"], ["data_query"])
gate = InputGate(
    InputGateConfig(
        intent={
            "tools_for_label": {"data_query": ["search", "get_sql"]},
            "read_only_tools": ["search"],
        }
    ),
    intent=classifier,
)
decision = gate.evaluate("What was revenue last quarter?")
if decision.action == "block":
    raise RuntimeError(decision.error)
```

## Models

The default backend is `builtin`: deterministic scorers with the same
thresholds, so tests run without downloading weights. Point a check at
`huggingface` to load the checkpoint named in the spec.

| Check | Checkpoint | Block when |
| --- | --- | --- |
| Content filter | `ibm-granite/granite-guardian-hap-125m` | score ≥ 0.5 |
| CPU fallback name | `classification_transformer_en_slate.38m.hap` | same contract |
| Prompt injection | `meta-llama/Llama-Prompt-Guard-2-86M` | user message, score ≥ 0.5 |
| Safety vote | `hivetrace/gliner-guard-uniencoder` | score ≥ 0.5 |
| Intent encoder | `BAAI/bge-small-en-v1.5` plus logistic regression | confidence < 0.6 restricts tools |

Only the intent head is trained. If a configured model cannot be loaded,
`on_unavailable` is `builtin` or `deny`. A model that exceeds the input budget
denies the turn.

PII defaults to `auto`: Microsoft Presidio with `en_core_web_sm` when that
stack imports, otherwise checksum-aware patterns (email, phone, Luhn cards,
SSN structure, IBAN mod-97, Spanish NIF). Set `pii.backend` to `pattern` to
skip Presidio, or `presidio` to fail closed when Presidio cannot run.

`embedding: hashing` is the portable encoder. `embedding: bge` loads
`BAAI/bge-small-en-v1.5` when sentence-transformers is installed.

## Overhead

| Phase | Target | On miss |
| --- | --- | --- |
| Input gate | 100 ms, p95 | Deny the turn when a model check exceeds the remaining budget |

`python -m agent_guardrails.eval` prints a JSON report and exits 0 only when
every case passed and the p95 stayed inside 100 ms. The packaged eval uses the
builtin backend, so the measured p95 is the code path without downloaded
weights. That path meets the target in CI.

Recorded exception, not a waiver: the published classifiers and Presidio are
heavier than 100 ms on a cold load, and this repository does not download
their weights in CI. Hosts that set `backend: huggingface` still have the
timeout. A score that overruns 100 ms is a deny, which is the architecture's
timeout action. Builtin scorers are not a claim of the same recall as
Granite Guardian, Prompt Guard 2, or GLiNER. Using them in production instead
of the published checkpoints needs security product sign-off.

## template-agent

Leave `input_guardrail.enabled` false until you want the gate on. It does not
replace Granite Guardian. When both are enabled, both screen the user message.

```yaml
input_guardrail:
  enabled: true
  gate: {}
```

Startup registers `InputGuardrailHandler` on LangChain chat-model calls. A
block raises `InputContentSafetyError`. Redacted text replaces the user
message. A later tool call in the same task raises `ToolContentSafetyError`
when the Phase 1 allowlist does not include that tool.

Fit the intent head and set `tools_for_label` plus `read_only_tools` before
relying on tool restriction. Until the head is fitted, intent does not narrow
the tool list.
