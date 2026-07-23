"""Shared configuration and helpers for load tests.

Provides:
- Environment-based endpoint configuration
- SSE stream parsing utilities
- TTFT extraction from SSE events
- Prompt corpus loading
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

# ── Endpoint configuration ──────────────────────────────────────

AGENT_BASE_URL = os.environ.get("AGENT_BASE_URL", "http://localhost:8123")

THREADS_ENDPOINT = "/threads"
THREADS_SEARCH_ENDPOINT = "/threads/search"
RUNS_STREAM_ENDPOINT = "/threads/{thread_id}/runs/stream"
MEMORIES_ENDPOINT = "/memories"
RULES_ENDPOINT = "/rules"
FEEDBACK_ENDPOINT = "/feedback"
FEEDBACK_GET_ENDPOINT = "/feedback/{thread_id}"

# ── SSE parsing ─────────────────────────────────────────────────


@dataclass
class SSEEvent:
    """Parsed Server-Sent Event."""

    event: str = ""
    data: str = ""
    id: str = ""
    retry: int | None = None


def parse_sse_stream(
    response_iter: Generator[bytes, None, None],
) -> Generator[SSEEvent, None, None]:
    """Parse an SSE byte stream into structured events.

    Handles multi-line ``data:`` fields, event type prefixes,
    and blank-line delimiters per the SSE specification.

    Args:
        response_iter: Iterator of raw bytes from an HTTP response
            (e.g. ``response.iter_lines()`` or ``response.iter_content()``).

    Yields:
        Parsed ``SSEEvent`` instances for each complete event block.
    """
    current = SSEEvent()
    data_lines: list[str] = []

    for raw_chunk in response_iter:
        if isinstance(raw_chunk, bytes):
            line = raw_chunk.decode("utf-8", errors="replace")
        else:
            line = raw_chunk

        # Blank line signals end of an event block
        if not line.strip():
            if data_lines or current.event:
                current.data = "\n".join(data_lines)
                yield current
                current = SSEEvent()
                data_lines = []
            continue

        if line.startswith("event:"):
            current.event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        elif line.startswith("id:"):
            current.id = line[len("id:") :].strip()
        elif line.startswith("retry:"):
            try:
                current.retry = int(line[len("retry:") :].strip())
            except ValueError:
                pass
        # Lines starting with ':' are comments — ignore silently

    # Yield any trailing event that was not terminated by a blank line
    if data_lines or current.event:
        current.data = "\n".join(data_lines)
        yield current


# ── TTFT measurement ────────────────────────────────────────────


@dataclass
class StreamMetrics:
    """Metrics collected from consuming an SSE stream."""

    ttft_ms: float = 0.0
    total_time_ms: float = 0.0
    token_count: int = 0
    error: Exception | None = None
    first_token_received: bool = False


def extract_content_from_event(sse_event: SSEEvent) -> str | None:
    """Extract text content from an SSE data event.

    Attempts to parse the data field as JSON and extract a ``content``
    key.  Falls back to returning the raw data if JSON parsing fails
    and the data is non-empty.

    Returns:
        The content string, or ``None`` if the event carries no content.
    """
    if not sse_event.data:
        return None

    try:
        parsed = json.loads(sse_event.data)
        if isinstance(parsed, dict):
            return parsed.get("content")
    except (json.JSONDecodeError, TypeError):
        pass

    # Some SSE implementations send raw text content
    stripped = sse_event.data.strip()
    return stripped if stripped else None


def measure_sse_stream(
    response_iter: Generator[bytes, None, None],
    start_time: float | None = None,
) -> StreamMetrics:
    """Consume an SSE stream and collect performance metrics.

    Measures TTFT (time to first content-bearing token), total stream
    duration, and token count.

    Args:
        response_iter: Raw byte iterator from the HTTP response.
        start_time: Epoch timestamp when the request was sent.
            Defaults to ``time.time()`` if not provided.

    Returns:
        Populated ``StreamMetrics`` with timing and count data.
    """
    if start_time is None:
        start_time = time.time()

    metrics = StreamMetrics()

    try:
        for event in parse_sse_stream(response_iter):
            content = extract_content_from_event(event)
            if content is not None:
                metrics.token_count += 1
                if not metrics.first_token_received:
                    metrics.ttft_ms = (time.time() - start_time) * 1000
                    metrics.first_token_received = True
    except Exception as exc:
        metrics.error = exc

    metrics.total_time_ms = (time.time() - start_time) * 1000
    return metrics


# ── Prompt corpus ───────────────────────────────────────────────

_PROMPTS_PATH = Path(__file__).parent / "payloads" / "prompts.json"
_prompt_cache: dict[str, list[str]] | None = None


def load_prompts() -> dict[str, list[str]]:
    """Load the prompt corpus from ``payloads/prompts.json``.

    Returns:
        Dictionary with keys ``short``, ``medium``, ``long``, each
        mapping to a list of prompt strings.

    Raises:
        FileNotFoundError: If the prompts file is missing.
        json.JSONDecodeError: If the prompts file contains invalid JSON.
    """
    global _prompt_cache  # noqa: PLW0603

    if _prompt_cache is not None:
        return _prompt_cache

    with open(_PROMPTS_PATH) as f:
        _prompt_cache = json.load(f)

    return _prompt_cache


def get_all_prompts() -> list[str]:
    """Return a flat list of all prompts across categories.

    Useful for random selection during load tests.
    """
    corpus = load_prompts()
    all_prompts: list[str] = []
    for category in ("short", "medium", "long"):
        all_prompts.extend(corpus.get(category, []))
    return all_prompts


# ── Run payload builder ─────────────────────────────────────────


def build_run_payload(message: str, assistant_id: str = "agent") -> dict:
    """Build the JSON payload for a streaming run request.

    Args:
        message: The human message content.
        assistant_id: The assistant identifier (defaults to ``agent``).

    Returns:
        Dictionary ready to be serialized as JSON in the request body.
    """
    return {
        "input": {
            "messages": [
                {
                    "role": "human",
                    "content": message,
                }
            ]
        },
        "assistant_id": assistant_id,
    }
