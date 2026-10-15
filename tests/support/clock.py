"""A clock tests can control.

Backoff, lease expiry, heartbeat, token-expiry margins, cache TTLs and
memory decay are all time-dependent. Against a wall clock they are
either untestable or tested with `sleep`, which makes the suite slow and
flaky in exactly the places correctness matters.

`FakeClock` replaces the time sources those paths read and advances only
when a test says so.
"""

from __future__ import annotations

import datetime as _datetime
import time as _time
from typing import Any


class FakeClock:
    """A monotonic, manually advanced clock.

    `time.monotonic` and `time.time` move together: code that mixes them
    (ours does — token expiry uses `time.time`, backoff uses
    `monotonic`) must still see a consistent ordering.
    """

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._now = start
        self.sleeps: list[float] = []

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time does not go backwards")
        self._now += seconds

    def time(self) -> float:
        return self._now

    def monotonic(self) -> float:
        return self._now

    def now(self, tz: Any = None) -> _datetime.datetime:
        return _datetime.datetime.fromtimestamp(self._now, tz or _datetime.timezone.utc)

    async def asleep(self, seconds: float, *_args: Any, **_kwargs: Any) -> None:
        """Stand-in for `asyncio.sleep`: records and advances, never waits.

        Recording the durations is what lets a test assert *backoff
        shape* — that retries grow — rather than just that a retry
        happened.
        """
        self.sleeps.append(seconds)
        self.advance(seconds)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


def install(monkeypatch: Any, clock: FakeClock, *modules: str) -> None:
    """Point time sources at `clock`.

    Patches the `time` and `asyncio` modules globally, plus any named
    module that imported `time` directly, since `from time import time`
    binds a reference patching `time.time` will not reach.
    """
    import asyncio
    import importlib

    monkeypatch.setattr(_time, "time", clock.time)
    monkeypatch.setattr(_time, "monotonic", clock.monotonic)
    monkeypatch.setattr(_time, "sleep", clock.sleep)
    monkeypatch.setattr(asyncio, "sleep", clock.asleep)

    for name in modules:
        module = importlib.import_module(name)
        for attr, value in (
            ("time", clock.time),
            ("monotonic", clock.monotonic),
        ):
            if hasattr(module, attr):
                monkeypatch.setattr(module, attr, value, raising=False)
