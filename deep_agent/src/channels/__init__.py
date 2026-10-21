"""Inbound channels: ways for something other than the UI to start a run.

The agent has always been a thing you visit. A channel makes it a thing
that reaches you — Slack posts an event, a CI job posts a webhook, and a
run happens without a browser or an SSO session.

That inverts the trust model, so the rules here are deliberately rigid:

* Every channel verifies a signature. There is no unauthenticated path.
* Every channel maps its sender to a platform user. Runs are owned,
  authorised and audited exactly as a UI run is.
* Channel endpoints live under ``/channels`` and are the *only* thing the
  external Route exposes. The rest of the agent API stays internal.
"""
