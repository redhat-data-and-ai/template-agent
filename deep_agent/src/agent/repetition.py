"""Degenerate repetition-loop detection for LLM output.

Gemini (and other autoregressive models) can occasionally get stuck emitting
the same sentence/phrase dozens of times in a single completion — most often
when refusing a prompt. This wastes tokens/cost, degrades latency and UX,
and can restart the entire response block mid-generation.

``detect_repetition_loop`` is a pure, dependency-free utility that looks for a
short unit of text repeated many times *consecutively* at the end of a (possibly
still-growing) string and returns a truncated version with the repeats collapsed
to a single copy. It is intentionally conservative (exact-match, bounded window)
so it runs on every streamed token chunk without reasoning about semantics.

Used by :mod:`deep_agent.aegra.safety` (``SafetyAwareRunnable``) to break a
generation loop early when streaming, and to clean up the final message when
not streaming.
"""

from __future__ import annotations

from deep_agent.src.settings import settings

# Hard cap on repeated-unit length. Bounds each check to O(_MAX_UNIT_LEN ** 2)
# since only the tail of the text is ever inspected.
_MAX_UNIT_LEN = 400


def detect_repetition_loop(
    text: str,
    min_unit_len: int | None = None,
    min_repeats: int | None = None,
    max_unit_len: int = _MAX_UNIT_LEN,
) -> tuple[bool, str]:
    """Detect a consecutively repeated unit of text at the end of ``text``.

    Args:
        text: The (possibly partial/streaming) completion text to inspect.
        min_unit_len: Minimum character length of the repeated unit.
            Defaults to ``settings.REPETITION_LOOP_MIN_UNIT_LEN``.
        min_repeats: Minimum consecutive repeats to flag a loop.
            Defaults to ``settings.REPETITION_LOOP_MIN_REPEATS``.
        max_unit_len: Largest unit length to consider.

    Returns:
        ``(False, text)`` if no loop is detected.
        ``(True, truncated_text)`` with all but one copy removed if detected.
    """
    if min_unit_len is None:
        min_unit_len = settings.REPETITION_LOOP_MIN_UNIT_LEN
    if min_repeats is None:
        min_repeats = settings.REPETITION_LOOP_MIN_REPEATS

    if not text or min_unit_len <= 0 or min_repeats <= 1:
        return False, text

    n = len(text)
    largest_unit_len = min(max_unit_len, n // min_repeats)

    for unit_len in range(min_unit_len, largest_unit_len + 1):
        window_len = unit_len * min_repeats
        if window_len > n:
            break

        tail = text[-window_len:]
        unit = tail[:unit_len]
        if not unit.strip():
            # Skip whitespace-only "units" — not a meaningful loop.
            continue

        is_loop = all(
            tail[i * unit_len : (i + 1) * unit_len] == unit
            for i in range(1, min_repeats)
        )
        if not is_loop:
            continue

        # Extend backward to collapse all consecutive repeats (may exceed min_repeats).
        total_repeats = min_repeats
        while True:
            start = n - (total_repeats + 1) * unit_len
            if start < 0:
                break
            if text[start : start + unit_len] != unit:
                break
            total_repeats += 1

        keep_len = n - (total_repeats - 1) * unit_len
        return True, text[:keep_len]

    return False, text
