"""Generic skill tests with auto-discovery.

This single test file automatically discovers and tests all skills in
agent_config/skills/ by loading their evals.json files.
"""

import asyncio
import json
from pathlib import Path

import pytest
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

from template_agent.src.infrastructure.backend import get_backend


# ============================================================================
# Skills are self-contained - no external tools needed
# ============================================================================
#
# All skills use only local scripts and reference documents:
# - client-intake: uses scripts/convert_units.py and reference docs
# - bmi-report: uses reference docs (bmi_categories.md, health_tips, etc.)
# - email-formatter: uses reference docs (template.html, css rules, etc.)
#
# No mock tools required for skill testing!


# ============================================================================
# Helpers
# ============================================================================


def save_output(output_dir: Path, output: str):
    """Save agent output to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "response.md").write_text(output)


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


async def run_agent_async(agent, prompt: str, thread_id: str, tracer) -> str:
    """Run agent asynchronously."""
    from conftest import extract_output, extract_tokens

    tracer.start()

    config = {"configurable": {"thread_id": thread_id}}

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


def run_agent_sync(agent, prompt: str, thread_id: str, tracer) -> str:
    """Synchronous wrapper for async agent execution."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(run_agent_async(agent, prompt, thread_id, tracer))


def create_skill_agent(skill_dir: str, skill_name: str, model):
    """Create agent for a specific skill.

    Skills are self-contained and use only local scripts/reference docs.
    No external tools are needed.
    """
    agent = create_deep_agent(
        model=model,
        skills=[skill_dir],
        tools=[],  # Skills don't need external tools
        backend=get_backend(),
        checkpointer=MemorySaver(),
    )

    return agent


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.skills
def test_skill_evaluation(
    skill_eval,
    workspace_dir,
    tracer,
    evaluator,
    model,
):
    """Test skill with eval case using LLM judge.

    Auto-discovers all skills from agent_config/skills/ and runs their evals.
    """
    skill_name = skill_eval["skill_name"]
    skill_dir = skill_eval["skill_dir"]
    eval_id = skill_eval["eval_id"]
    eval_case = skill_eval["eval_case"]

    # Setup workspace
    workspace = workspace_dir / skill_name / f"eval-{eval_id}"
    output_dir = workspace / "outputs"

    # Create agent with skill
    agent = create_skill_agent(skill_dir, skill_name, model)

    # Run agent
    prompt = eval_case["prompt"]
    thread_id = f"{skill_name}-test-{eval_id}"
    output = run_agent_sync(agent, prompt, thread_id, tracer)

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

    # Assert pass rate (70% threshold)
    pass_rate = summary["pass_rate"]
    assert pass_rate >= 0.7, (
        f"{skill_name} failed eval-{eval_id}: "
        f"{summary['passed']}/{summary['total']} assertions passed"
    )
