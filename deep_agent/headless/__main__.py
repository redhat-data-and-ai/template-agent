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
        description="Invoke this agent without the chat UI."
    )
    parser.add_argument(
        "workflow",
        choices=WORKFLOWS,
        help=(
            "local signs in with your SSO and skips JIT. "
            "preprod and prod use the service account, grant JIT, and call the deployed agent."
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
    token = access_token(settings)
    if not token:
        return 1
    print("Authenticated.")
    return HeadlessInvoker(settings, token).run()


if __name__ == "__main__":
    sys.exit(main())
