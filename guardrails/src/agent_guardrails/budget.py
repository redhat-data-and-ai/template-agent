"""Run a scorer inside the input budget.

Builtin scorers run inline. Model scorers run on a daemon thread and the
turn is denied when they are still running at the deadline.
"""

from __future__ import annotations

import threading
import time
from typing import Any


def score_text(
    scorer: Any, text: str, timeout_s: float
) -> tuple[float, bool, bool, float]:
    """Return ``(score, timed_out, failed, latency_ms)``.

    ``failed`` is true when the scorer raised. Callers deny the turn for
    both a timeout and a failure.
    """
    started = time.perf_counter()
    if timeout_s <= 0:
        return 1.0, True, False, 0.0
    if not getattr(scorer, "uses_model", False):
        try:
            score = float(scorer.score(text))
        except Exception:
            latency = (time.perf_counter() - started) * 1000
            return 1.0, False, True, latency
        latency = (time.perf_counter() - started) * 1000
        return score, False, False, latency

    box: dict[str, object] = {}

    def _run() -> None:
        try:
            box["score"] = float(scorer.score(text))
        except Exception as exc:
            box["error"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout_s)
    latency = (time.perf_counter() - started) * 1000
    if worker.is_alive():
        return 1.0, True, False, latency
    if "error" in box:
        return 1.0, False, True, latency
    found = box.get("score")
    if not isinstance(found, float):
        return 1.0, False, True, latency
    return found, False, False, latency
