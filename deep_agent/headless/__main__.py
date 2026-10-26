"""Run ``python -m deep_agent.headless``."""

from __future__ import annotations

import argparse
import sys

from deep_agent.headless.auth import access_token
from deep_agent.headless.invoke import HeadlessInvoker
from deep_agent.headless.settings import HeadlessSettings


def main(argv: list[str] | None = None) -> int:
    """Authenticate and invoke the agent once."""
    parser = argparse.ArgumentParser(
        description="Invoke this agent without the chat UI."
    )
    parser.add_argument(
        "message", nargs="?", help="Prompt to send. Defaults to AGENT_MESSAGE."
    )
    parser.add_argument(
        "--target",
        choices=("local", "aif"),
        help="local talks to this agent API. aif talks to AI Factory. Default: HEADLESS_TARGET or local.",
    )
    parser.add_argument(
        "--auth",
        choices=("sso", "password", "client-creds"),
        help="sso opens a browser. password uses SSO_SA_USERNAME. client-creds uses the client secret only.",
    )
    args = parser.parse_args(argv)

    settings = HeadlessSettings()
    updates: dict[str, str] = {}
    if args.target:
        updates["headless_target"] = args.target
    if args.auth:
        updates["auth_mode"] = args.auth
    if args.message:
        updates["agent_message"] = args.message
    if updates:
        settings = settings.model_copy(update=updates)

    print(f"Target: {settings.headless_target}")
    print(f"Auth: {settings.auth_mode}")
    token = access_token(settings)
    if not token:
        return 1
    print("Authenticated.")
    return HeadlessInvoker(settings, token).run()


if __name__ == "__main__":
    sys.exit(main())
