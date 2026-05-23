# Agent Evaluations

This directory contains evaluation suites for testing the deep-agent BMI fitness assistant.

## Overview

We use multiple evaluation frameworks:

1. **Skills Evals** - Test individual skills (client-intake, bmi-report, email-formatter)
2. **Promptfoo** - Fast iteration testing with LLM-rubric assertions
3. **Lightspeed** - Formal benchmark evaluation (optional)

## Running Evaluations Locally

### Prerequisites

- Agent must be running at `http://localhost:5002`
- Environment variables set:
  - `GOOGLE_GENAI_API_KEY` - for LLM calls and judging
  - `GOOGLE_APPLICATION_CREDENTIALS_CONTENT` - for Google service account

### 1. Skills Evals (Pytest-based)

Tests individual skills using LLM-as-judge:

```bash
# Run all skills evals
make test-skills

# Or use pytest directly
pytest tests/skills -m skills -v
```

Results are saved to `tests/workspaces/<skill-name>/eval-<id>/`

### 2. Promptfoo Agent Evals

Fast iteration testing with assertion-based evaluation:

```bash
# Start the agent first
make local

# In another terminal, run evals
make eval-promptfoo

# Or run directly
cd config/agent/evals/promptfoo
npx promptfoo@latest eval

# View results in browser
npx promptfoo@latest view
```

**What it tests:**
- BMI calculation delegation to analyst subagent
- Imperial unit conversion
- Health tips by BMI category
- Email delivery validation
- Out-of-scope request handling
- Edge cases (missing data, invalid input)

### 3. Lightspeed Evals (Optional)

Formal benchmark evaluation (requires additional setup):

```bash
# Install lightspeed-evaluation
pip install lightspeed-evaluation

# Run evals
lightspeed-eval \
  --system-config config/agent/evals/lightspeed/system.yaml \
  --eval-data config/agent/evals/lightspeed/eval_data.yaml \
  --output-dir eval_output
```

## CI/CD

Evals run automatically in GitHub Actions:

- **Unit Tests** - Run on every PR/push
- **Skills Evals** - Run on every PR/push
- **Promptfoo Agent Evals** - Run on every PR/push

See `.github/workflows/test.yml` for details.

## Eval Structure

### Promptfoo Config

```yaml
providers:
  - http endpoint to agent
tests:
  - description: Test case name
    vars:
      prompt: User input
    assert:
      - type: llm-rubric | contains | not-contains
        value: Expected behavior
```

### Lightspeed Config

```yaml
conversation_group_id: test_scenario
turns:
  - turn_id: step_1
    query: User input
    expected_response: Expected behavior
    turn_metrics:
      - custom:answer_correctness
      - geval:delegation_compliance
```

## Adding New Tests

### For Promptfoo:

1. Edit `config/agent/evals/promptfoo/config.yaml`
2. Add new test case under `tests:`
3. Run `npx promptfoo eval` to verify

### For Skills:

1. Edit skill's `config/agent/skills/<skill-name>/evals/evals.json`
2. Add new eval case with assertions
3. Run `pytest tests/skills -m skills` to verify

## Troubleshooting

**Agent not responding:**
```bash
# Check if agent is running
curl http://localhost:5002/health

# Check logs
tail -f logs/agent.log
```

**Promptfoo timeout:**
- Increase timeout in `config/agent/evals/promptfoo/config.yaml`:
  ```yaml
  defaultTest:
    options:
      timeout: 180000  # 3 minutes
  ```

**Skills eval failures:**
- Check LLM judge is using correct model (gemini-3.1-pro-preview)
- Ensure pass rate threshold is reasonable (70% default)
- Review `tests/workspaces/<skill>/eval-<id>/grading.json`

## Metrics

- **Skills Evals**: 70% pass rate required per eval
- **Promptfoo**: All assertions must pass
- **Lightspeed**: Configurable thresholds per metric
