"""The unit suite must not reach a real datastore.

A suite that passes because Postgres happens to be running on the
developer's machine is a suite that tells you nothing. This repo has
already paid for that once: a unit test quietly depended on a live
database, passed locally, and hung for 29 minutes in CI.
"""

from __future__ import annotations

import pytest


def test_redis_url_points_at_a_dead_port():
    import deep_agent.aegra.redis as redis_mod

    assert ":1/" in redis_mod.REDIS_URL


def test_the_sync_redis_client_cannot_connect():
    """None, not a client — so cache helpers take their disabled path."""
    import deep_agent.aegra.redis as redis_mod

    assert redis_mod.get_redis_client() is None


async def test_the_async_redis_client_cannot_connect():
    import deep_agent.aegra.redis as redis_mod

    assert await redis_mod.get_async_redis_client() is None


def test_postgres_points_at_a_dead_port():
    from deep_agent.src.settings import Settings

    assert Settings().POSTGRES_PORT == 1


@pytest.mark.real_datastores
def test_the_marker_opts_out():
    """Escape hatch for a test that brings its own datastore."""
    import os

    assert os.environ.get("POSTGRES_PORT") != "1"
