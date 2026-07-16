"""HeadlessRuntime — ServerRuntime adapter for headless mode."""

from __future__ import annotations


class HeadlessUser:
    """Minimal user for headless mode (no SSO)."""

    def __init__(self, identity: str = "headless-worker") -> None:
        """Initialize the headless user with the given identity."""
        self.identity = identity
        self.access_token: str | None = None
        self.refresh_token: str | None = None


class HeadlessRuntime:
    """Adapter that provides a ServerRuntime-compatible interface for headless mode.

    The graph factory (graph.py:agent) reads runtime.user.access_token,
    runtime.user.refresh_token, and runtime.user.identity. This class
    provides those attributes without SSO.
    """

    def __init__(self, identity: str = "headless-worker") -> None:
        """Initialize the headless runtime with the given identity."""
        self.user = HeadlessUser(identity)
