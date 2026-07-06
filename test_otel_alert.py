#!/usr/bin/env python3
"""Test script to emit sample OTEL token usage events."""

import asyncio
import os


async def test_otel_usage() -> None:
    os.environ["ENABLE_OTEL_METRICS"] = "true"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "localhost:4327",
    )
    os.environ["OTEL_SERVICE_NAME"] = "template-agent-test"

    from deep_agent.src.observability.otel_setup import setup_otel_metrics
    from deep_agent.src.settings import settings
    from deep_agent.src.token_budget.otel_emit import emit_token_usage
    from deep_agent.utils.pylogger import get_python_logger

    log = get_python_logger()
    setup_otel_metrics(settings, log)

    samples = [
        ("test-thread-1", "user-1", 1200, 340, 1200, 1200, 340),
        ("test-thread-1", "user-1", 800, 210, 2000, 2000, 550),
        ("test-thread-2", "user-2", 500, 90, 500, 500, 90),
    ]

    for i, (
        thread_id,
        user_id,
        input_tokens,
        output_tokens,
        cumulative_total,
        cumulative_input,
        cumulative_output,
    ) in enumerate(samples, 1):
        print(
            f"\n[{i}] Emitting usage for {thread_id}: "
            f"+{input_tokens} in / +{output_tokens} out "
            f"(cumulative {cumulative_total})"
        )
        emit_token_usage(
            thread_id=thread_id,
            user_id=user_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cumulative_total=cumulative_total,
            cumulative_input=cumulative_input,
            cumulative_output=cumulative_output,
        )
        await asyncio.sleep(1)

    print("\nDone. Check your collector for token_budget.tokens metrics.")


if __name__ == "__main__":
    asyncio.run(test_otel_usage())
