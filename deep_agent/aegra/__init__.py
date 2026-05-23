"""Aegra integration for agent deployment.

This package bridges template-agent with Aegra,
enabling the agent to be served via `aegra dev` or `aegra serve`
and used with deep-agents-ui.

Modules:
    graph: Graph builder and exported agent for aegra.json
    auth: OIDC/SSO authentication with token propagation
    state: Extended LangGraph state schema
    converters: State conversion and serialization utilities
    serialization: Full state serialization/deserialization
    nodes: Error-handling node wrappers for graph execution
    middleware: Authentication middleware (API key, JWT)
    telemetry: Langfuse integration
    redis: Redis caching layer
"""

__version__ = "0.1.0"
