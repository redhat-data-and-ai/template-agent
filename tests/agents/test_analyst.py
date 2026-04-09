"""Tests for analyst subagent (bmi-report skill)."""

import asyncio
import json
from pathlib import Path

import pytest

from conftest import (
    create_analyst_agent,
    extract_output,
    extract_tokens,
    load_skill_evals,
)


# ============================================================================
# Helpers
# ============================================================================


def save_output(output_dir: Path, output: str):
    """Save agent output to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(output)


def save_grading(output_dir: Path, results: list, summary: dict):
    """Save grading results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grading = {"assertion_results": results, "summary": summary}
    (output_dir / "grading.json").write_text(json.dumps(grading, indent=2))


def calculate_summary(results: list) -> dict:
    """Calculate pass/fail summary."""
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    return {
        "passed": passed,
        "failed": total - passed,
        "total": total,
        "pass_rate": passed / total if total > 0 else 0,
    }


def build_context(skill_name: str, eval_id: int, eval_case: dict) -> dict:
    """Build evaluation context."""
    return {
        "skill_name": skill_name,
        "eval_id": eval_id,
        "prompt": eval_case["prompt"],
        "expected_output": eval_case.get("expected_output"),
    }


async def run_agent_async(agent, prompt: str, tracer) -> str:
    """Run agent asynchronously."""
    tracer.start()

    config = {"configurable": {"thread_id": f"analyst-test"}}

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config,
        )

        output = extract_output(result)
        tokens = extract_tokens(result)
        tracer.end(total_tokens=tokens)
        return output

    except Exception:
        tracer.end()
        raise


def run_agent_sync(agent, prompt: str, tracer) -> str:
    """Synchronous wrapper for async agent execution."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(run_agent_async(agent, prompt, tracer))


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_generate_tests(metafunc):
    """Generate test cases from evals.json."""
    if "eval_case" in metafunc.fixturenames:
        skill_config = load_skill_evals("bmi-report")
        evals = skill_config["evals"]

        test_cases = []
        ids = []

        for eval_case in evals:
            eval_id = eval_case["id"]
            description = eval_case.get("description", f"eval-{eval_id}")

            test_cases.append(
                {
                    "eval_id": eval_id,
                    "description": description,
                    "eval_case": eval_case,
                    "skill_name": skill_config["skill_name"],
                }
            )
            ids.append(f"eval-{eval_id}")

        metafunc.parametrize(
            "eval_case,eval_id,description,skill_name",
            [
                (tc["eval_case"], tc["eval_id"], tc["description"], tc["skill_name"])
                for tc in test_cases
            ],
            ids=ids,
        )


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.analyst
def test_analyst_evaluation(
    eval_case,
    eval_id,
    description,
    skill_name,
    workspace_dir,
    tracer,
    evaluator,
    model,
):
    """Test analyst subagent with bmi-report skill."""
    # Setup workspace
    workspace = workspace_dir / "analyst-workspace" / f"eval-{eval_id}"
    output_dir = workspace / "outputs"

    # Create agent
    agent = create_analyst_agent(model)

    # Run agent
    prompt = eval_case["prompt"]
    output = run_agent_sync(agent, prompt, tracer)

    # Save output
    save_output(output_dir, output)

    # Grade assertions
    context = build_context(skill_name, eval_id, eval_case)
    results = []

    for assertion in eval_case["assertions"]:
        result = evaluator.evaluate(
            assertion=assertion,
            output=output,
            context=context,
        )
        results.append(result)

    # Calculate summary
    summary = calculate_summary(results)

    # Save grading
    save_grading(output_dir, results, summary)

    # Assert pass rate
    pass_rate = summary["pass_rate"]
    assert pass_rate >= 0.7, (
        f"Analyst failed eval-{eval_id}: "
        f"{summary['passed']}/{summary['total']} assertions passed"
    )
