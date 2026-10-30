"""Headless access to this agent.

Server (ships with every deploy)
    ``POST /invoke`` on the agent HTTP app auto-approves tool pauses and
    streams the run. Callers: CI, other services, eval, or the CLI below.
    Auth is the caller's bearer token (SSO or service account).

CLI (optional client of that API)
    ``python -m deep_agent.headless local|preprod|prod "…"``

    * ``local`` — browser SSO → ``http://127.0.0.1:<port>/invoke``
    * ``preprod`` / ``prod`` — SA password → remote ``AGENT_URL/invoke``
      (direct; no AIF registry). Optional AIF engine via
      ``AGENT_BASE_URL`` + ``AGENT_ORG`` + ``AGENT_NAME``.
"""
