"""Run ``python -m deep_agent.headless``."""

from __future__ import annotations

import argparse
import sys

from deep_agent.headless.auth import access_token
from deep_agent.headless.invoke import HeadlessInvoker
from deep_agent.headless.settings import HeadlessSettings
from deep_agent.headless.workflow import WORKFLOWS, WorkflowError, apply_workflow


def main(argv: list[str] | None = None) -> int:
    """Authenticate and invoke the agent once."""
    parser = argparse.ArgumentParser(
        description=(
            "CLI client for the agent's headless POST /invoke API "
            "(same route exposed when this agent is deployed). "
            "Tools are auto-approved on that path."
        )
    )
    parser.add_argument(
        "workflow",
        choices=WORKFLOWS,
        help=(
            "local: browser SSO → this agent's /invoke (AGENT_URL or AGENT_PORT). "
            "preprod/prod: SA password → deployed AGENT_URL /invoke (direct; no AIF "
            "registry). Optional AIF engine via AGENT_BASE_URL+ORG+NAME."
        ),
    )

    parser.add_argument(
        "message", nargs="?", help="Prompt to send. Defaults to AGENT_MESSAGE."
    )
    args = parser.parse_args(argv)

    settings = HeadlessSettings()
    try:
        settings = apply_workflow(settings, args.workflow)
    except WorkflowError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.message:
        settings = settings.model_copy(update={"agent_message": args.message})

    print(f"Workflow: {settings.workflow}")
    print(f"Auth: {settings.auth_mode}")
    print(f"URL: {settings.invoke_url()}")
    token = access_token(settings)
    if not token:
        return 1
    print("Authenticated.")
    return HeadlessInvoker(settings, token).run()


if __name__ == "__main__":
    sys.exit(main())
