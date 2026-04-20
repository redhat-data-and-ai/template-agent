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
    """Calculate pass/fail summary.

    Assertions with passed=null are counted as 'aborted' and excluded
    from pass rate calculation (LLM judge couldn't determine verdict).
    """
    passed = sum(1 for r in results if r["passed"] is True)
    failed = sum(1 for r in results if r["passed"] is False)
    aborted = sum(1 for r in results if r["passed"] is None)
    total = len(results)

    # Pass rate excludes aborted tests
    evaluated = passed + failed
    pass_rate = passed / evaluated if evaluated > 0 else 0

    return {
        "passed": passed,
        "failed": failed,
        "aborted": aborted,
        "total": total,
        "pass_rate": pass_rate,
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
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(run_agent_async(agent, prompt, thread_id, tracer))


def create_skill_agent(skill_dir: str, skill_name: str, model):
    """Create agent for a specific skill.

    Skills are self-contained and use only local scripts/reference docs.
    No external tools are needed.
    """
    system_prompt = """\
    You are a helpful assistant with specialized skills.

    CRITICAL: You have been provided with skill instructions as part of SKILL.md

    You MUST strictly follow the skill instructions.

    When using a skill, follow its instructions EXACTLY as written.
    """

    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
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

    Each eval must pass 70% of its assertions to be considered successful.

    IMPORTANT: These tests use real LLM calls and LLM-as-judge evaluation,
    which means results can vary between runs. The system prompt helps guide
    the model to follow skill instructions more strictly.
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

    # Calculate summary for this eval
    summary = calculate_summary(results)

    # Save grading
    save_grading(output_dir, results, summary)

    # Assert pass rate (70% threshold per eval)
    pass_rate = summary["pass_rate"]

    # Build failure message
    msg_parts = [
        f"{skill_name} failed eval-{eval_id}:",
        f"{summary['passed']}/{summary['total']} assertions passed",
    ]
    if summary["aborted"] > 0:
        msg_parts.append(f"({summary['aborted']} aborted, excluded from rate)")
    msg_parts.append(f"pass_rate: {pass_rate:.1%}, threshold: 70%")

    assert pass_rate >= 0.7, " ".join(msg_parts)
