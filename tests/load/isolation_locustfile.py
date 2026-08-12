"""Locust entry point for user isolation load tests.

Verifies that user data isolation holds under concurrent multi-user
load. Each simulated user authenticates with a unique self-signed JWT
and asserts it can only see its own data.

The JWT provider starts a JWKS server automatically. The app under
test must be configured to validate tokens against this server:

    export ENABLE_AUTH=true
    export SSO_JWKS_URI=http://127.0.0.1:{JWKS_PORT}/jwks
    export SSO_ISSUER_URL=locust-isolation-test

Usage:

    # Smoke test (5 users, 2 min)
    LOAD_PROFILE=smoke locust -f tests/load/isolation_locustfile.py \
        --headless --host http://localhost:8123

    # Thorough test (10 users, 5 min)
    LOAD_PROFILE=thorough locust -f tests/load/isolation_locustfile.py \
        --headless --host http://localhost:8123

    # With Locust UI
    locust -f tests/load/isolation_locustfile.py \
        --host http://localhost:8123
"""

import logging
import os

from locust import events

from tests.load.jwt_provider import ISSUER, JWKS_URL
from tests.load.scenarios.user_isolation import IsolatedUser, canary_registry

logger = logging.getLogger(__name__)

LOAD_PROFILES = {
    "smoke": {"users": 5, "spawn_rate": 1, "run_time": "2m"},
    "thorough": {"users": 10, "spawn_rate": 1, "run_time": "5m"},
}

_isolation_violations: list[str] = []


@events.init.add_listener
def on_init(environment, **kwargs):  # type: ignore[no-untyped-def]
    """Print required env vars and active profile on startup."""
    profile_name = os.environ.get("LOAD_PROFILE", "smoke")
    profile = LOAD_PROFILES.get(profile_name)

    if profile is None:
        logger.warning(
            "Unknown LOAD_PROFILE '%s', falling back to 'smoke'. Valid profiles: %s",
            profile_name,
            ", ".join(sorted(LOAD_PROFILES)),
        )
        profile_name = "smoke"
        profile = LOAD_PROFILES[profile_name]

    logger.info(
        "=== User Isolation Load Test ===\n"
        "  Profile: %s (users=%d, spawn_rate=%d, run_time=%s)\n"
        "  JWKS server: %s\n"
        "  Required app env vars:\n"
        "    ENABLE_AUTH=true\n"
        "    SSO_JWKS_URI=%s\n"
        "    SSO_ISSUER_URL=%s\n"
        "    SSO_JWT_AUDIENCE=",
        profile_name,
        profile["users"],
        profile["spawn_rate"],
        profile["run_time"],
        JWKS_URL,
        JWKS_URL,
        ISSUER,
    )


@events.request.add_listener
def on_request(request_type, name, exception, **kwargs):  # type: ignore[no-untyped-def]
    """Track isolation violations across the run."""
    if request_type == "ISOLATION" and exception is not None:
        _isolation_violations.append(f"{name}: {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):  # type: ignore[no-untyped-def]
    """Log test start."""
    logger.info(
        "Isolation load test started. Target: %s",
        environment.host or "not set",
    )


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):  # type: ignore[no-untyped-def]
    """Log summary with isolation violation count."""
    stats = environment.stats
    remaining_canaries = len(canary_registry)

    if _isolation_violations:
        logger.critical(
            "!!! ISOLATION VIOLATIONS DETECTED: %d !!!\n%s",
            len(_isolation_violations),
            "\n".join(f"  - {v}" for v in _isolation_violations[:20]),
        )
    else:
        logger.info("No isolation violations detected.")

    logger.info(
        "Test complete. Requests: %d, Failures: %d, "
        "Isolation violations: %d, Remaining canaries: %d",
        stats.num_requests,
        stats.num_failures,
        len(_isolation_violations),
        remaining_canaries,
    )


__all__ = ["IsolatedUser"]
