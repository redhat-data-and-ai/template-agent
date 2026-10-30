#!/usr/bin/env python3
"""Framework-path benchmark for LangGraph Platform (Aegra).

Measures pure framework overhead — NO LLM calls. Captures:
  - /health and /readyz endpoint latency (100 iterations)
  - Thread create / get / delete lifecycle timing (100 iterations)
  - Graph build cold vs cache-hit (first vs second /runs/stream on same thread)

Usage:
    python3 benchmark_framework.py                           # default 100 iterations
    python3 benchmark_framework.py --iterations 50 --output results.json
    python3 benchmark_framework.py --compare before.json after.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AGENT_URL = "http://localhost:5002"
ASSISTANT_ID = "agent"
DEFAULT_ITERATIONS = 100
STREAM_TIMEOUT = 120


def _post_json(url: str, body: dict, timeout: float = 30) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get_json(url: str, timeout: float = 10) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get_status(url: str, timeout: float = 10) -> tuple[int, float]:
    """Return (status_code, latency_seconds)."""
    req = urllib.request.Request(url)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            return resp.status, time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, time.perf_counter() - t0


def _delete(url: str, timeout: float = 10) -> int:
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def _stream_first_event_time(
    url: str, body: dict, timeout: float = STREAM_TIMEOUT
) -> float:
    """POST to streaming endpoint and return time-to-first-SSE-data-line (seconds)."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
            if line.startswith("data: "):
                return time.perf_counter() - t0
    return time.perf_counter() - t0


def compute_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "p50": 0,
            "p95": 0,
            "p99": 0,
            "stdev": 0,
            "count": 0,
        }
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    p50_idx = min(int(n * 0.50), n - 1)
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)
    return {
        "min": round(sorted_vals[0], 6),
        "max": round(sorted_vals[-1], 6),
        "mean": round(statistics.mean(sorted_vals), 6),
        "median": round(statistics.median(sorted_vals), 6),
        "p50": round(sorted_vals[p50_idx], 6),
        "p95": round(sorted_vals[p95_idx], 6),
        "p99": round(sorted_vals[p99_idx], 6),
        "stdev": round(statistics.stdev(sorted_vals), 6) if n > 1 else 0,
        "count": n,
    }


def benchmark_health(base_url: str, iterations: int) -> dict[str, Any]:
    """Measure /health endpoint latency."""
    print(f"  [1/5] /health latency ({iterations} iterations) ...", flush=True)
    latencies: list[float] = []
    errors = 0
    for i in range(iterations):
        try:
            status, lat = _get_status(f"{base_url}/health")
            if status == 200:
                latencies.append(lat)
            else:
                errors += 1
        except Exception:
            errors += 1
        if (i + 1) % 25 == 0:
            print(f"         {i + 1}/{iterations} done", flush=True)
    print(f"         OK ({len(latencies)} successful, {errors} errors)")
    return {"latency_stats": compute_stats(latencies), "errors": errors}


def benchmark_readyz(base_url: str, iterations: int) -> dict[str, Any]:
    """Measure /readyz (or /ok) endpoint latency."""
    print(f"  [2/5] /readyz latency ({iterations} iterations) ...", flush=True)

    endpoint = "/readyz"
    try:
        _get_status(f"{base_url}{endpoint}")
    except Exception:
        endpoint = "/ok"
        try:
            _get_status(f"{base_url}{endpoint}")
        except Exception:
            print("         SKIPPED — neither /readyz nor /ok available")
            return {"latency_stats": compute_stats([]), "errors": 0, "endpoint": "N/A"}

    latencies: list[float] = []
    errors = 0
    for i in range(iterations):
        try:
            status, lat = _get_status(f"{base_url}{endpoint}")
            if status == 200:
                latencies.append(lat)
            else:
                errors += 1
        except Exception:
            errors += 1
        if (i + 1) % 25 == 0:
            print(f"         {i + 1}/{iterations} done", flush=True)
    print(
        f"         OK ({len(latencies)} successful, {errors} errors) [endpoint: {endpoint}]"
    )
    return {
        "latency_stats": compute_stats(latencies),
        "errors": errors,
        "endpoint": endpoint,
    }


def benchmark_thread_lifecycle(base_url: str, iterations: int) -> dict[str, Any]:
    """Measure thread create / get / delete lifecycle timing."""
    print(f"  [3/5] Thread lifecycle ({iterations} iterations) ...", flush=True)
    create_times: list[float] = []
    get_times: list[float] = []
    delete_times: list[float] = []
    errors: list[str] = []

    for i in range(iterations):
        try:
            t0 = time.perf_counter()
            result = _post_json(f"{base_url}/threads", {})
            create_time = time.perf_counter() - t0
            tid = result["thread_id"]
            create_times.append(create_time)

            t0 = time.perf_counter()
            _get_json(f"{base_url}/threads/{tid}")
            get_time = time.perf_counter() - t0
            get_times.append(get_time)

            t0 = time.perf_counter()
            _delete(f"{base_url}/threads/{tid}")
            delete_time = time.perf_counter() - t0
            delete_times.append(delete_time)

        except Exception as exc:
            errors.append(f"iteration {i + 1}: {exc}")

        if (i + 1) % 25 == 0:
            print(f"         {i + 1}/{iterations} done", flush=True)

    print(f"         OK ({len(create_times)} successful, {len(errors)} errors)")
    return {
        "create": compute_stats(create_times),
        "get": compute_stats(get_times),
        "delete": compute_stats(delete_times),
        "errors": errors,
    }


def benchmark_graph_cold_vs_cached(base_url: str, attempts: int = 5) -> dict[str, Any]:
    """Measure graph build: cold (new thread, first call) vs cache-hit (same thread, second call).

    Uses a minimal prompt that triggers the graph but produces a fast response.
    We measure time-to-first-SSE-data-line as a proxy for graph build + scheduling time.
    Runs `attempts` pairs and reports stats for cold vs cached.
    """
    print(f"  [4/5] Graph cold vs cached ({attempts} pairs) ...", flush=True)
    cold_times: list[float] = []
    cached_times: list[float] = []
    errors: list[str] = []

    body_template = {
        "assistant_id": ASSISTANT_ID,
        "input": {"messages": [{"role": "human", "content": "Say OK"}]},
        "stream_mode": ["messages"],
    }

    for i in range(attempts):
        try:
            tid = _post_json(f"{base_url}/threads", {})["thread_id"]
            url = f"{base_url}/threads/{tid}/runs/stream"

            cold = _stream_first_event_time(url, body_template)
            cold_times.append(cold)

            cached = _stream_first_event_time(url, body_template)
            cached_times.append(cached)

            speedup = ((cold - cached) / cold * 100) if cold > 0 else 0
            print(
                f"         pair {i + 1}/{attempts}: cold={cold:.3f}s  cached={cached:.3f}s  "
                f"speedup={speedup:+.1f}%"
            )
        except Exception as exc:
            errors.append(f"pair {i + 1}: {exc}")
            print(f"         pair {i + 1}/{attempts}: FAILED — {exc}")

    return {
        "cold": compute_stats(cold_times),
        "cached": compute_stats(cached_times),
        "errors": errors,
    }


def benchmark_version_info(base_url: str) -> dict[str, Any]:
    """Capture version/info metadata and measure latency."""
    print("  [5/5] Version & info capture ...", end=" ", flush=True)
    version_data: dict = {}
    info_data: dict = {}
    version_latency = 0.0
    info_latency = 0.0

    try:
        t0 = time.perf_counter()
        version_data = _get_json(f"{base_url}/version")
        version_latency = time.perf_counter() - t0
    except Exception:
        pass

    try:
        t0 = time.perf_counter()
        info_data = _get_json(f"{base_url}/info")
        info_latency = time.perf_counter() - t0
    except Exception:
        pass

    print("OK")
    return {
        "version": version_data,
        "info": info_data,
        "version_latency": round(version_latency, 6),
        "info_latency": round(info_latency, 6),
    }


def run_benchmark(
    base_url: str, iterations: int, graph_pairs: int = 5
) -> dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"  Framework Benchmark — {base_url}")
    print("  (NO LLM calls — pure framework overhead)")
    print(f"{'=' * 60}\n")

    meta_result = benchmark_version_info(base_url)
    health_result = benchmark_health(base_url, iterations)
    readyz_result = benchmark_readyz(base_url, iterations)
    thread_result = benchmark_thread_lifecycle(base_url, iterations)
    graph_result = benchmark_graph_cold_vs_cached(base_url, graph_pairs)

    now_iso = datetime.now(timezone.utc).isoformat()
    results: dict[str, Any] = {
        "meta": {
            "agent_url": base_url,
            "service": meta_result["version"].get("service", "unknown"),
            "version": meta_result["version"].get("version", "unknown"),
            "agent_name": meta_result["info"].get("name", "unknown"),
            "date": now_iso,
            "iterations": iterations,
            "graph_pairs": graph_pairs,
            "benchmark_type": "framework",
        },
        "health": health_result,
        "readyz": readyz_result,
        "thread_lifecycle": thread_result,
        "graph_cold_vs_cached": graph_result,
        "endpoint_metadata": meta_result,
    }

    _print_report(results)
    return results


def _print_report(results: dict) -> None:
    meta = results["meta"]

    print(f"\n{'=' * 60}")
    print("  Framework Benchmark Results")
    print(f"{'=' * 60}")
    print(f"  Agent    : {meta['agent_url']}")
    print(f"  Service  : {meta['service']} {meta['version']}")
    print(f"  Date     : {meta['date']}")
    print(f"  Iters    : {meta['iterations']}")
    print()

    header = f"  {'Metric':<28} {'min':>9} {'mean':>9} {'p50':>9} {'p95':>9} {'p99':>9} {'max':>9}"
    print("  --- Latency (seconds) ---")
    print(header)
    print("  " + "-" * (len(header) - 2))

    def _row(label: str, stats: dict) -> None:
        if stats["count"] == 0:
            return
        print(
            f"  {label:<28} {stats['min']:>9.4f} {stats['mean']:>9.4f} "
            f"{stats['p50']:>9.4f} {stats['p95']:>9.4f} {stats['p99']:>9.4f} {stats['max']:>9.4f}"
        )

    _row("/health", results["health"]["latency_stats"])
    _row("/readyz", results["readyz"]["latency_stats"])
    _row("Thread create", results["thread_lifecycle"]["create"])
    _row("Thread get", results["thread_lifecycle"]["get"])
    _row("Thread delete", results["thread_lifecycle"]["delete"])
    _row("Graph cold (first SSE)", results["graph_cold_vs_cached"]["cold"])
    _row("Graph cached (second SSE)", results["graph_cold_vs_cached"]["cached"])

    cold = results["graph_cold_vs_cached"]["cold"]
    cached = results["graph_cold_vs_cached"]["cached"]
    if cold["count"] > 0 and cached["count"] > 0:
        speedup = (
            ((cold["mean"] - cached["mean"]) / cold["mean"] * 100)
            if cold["mean"] > 0
            else 0
        )
        print(f"\n  Graph cache speedup: {speedup:+.1f}% (mean cold→cached)")

    total_errors = (
        results["health"]["errors"]
        + results["readyz"]["errors"]
        + len(results["thread_lifecycle"]["errors"])
        + len(results["graph_cold_vs_cached"]["errors"])
    )
    if total_errors:
        print(f"\n  ⚠ Total errors: {total_errors}")

    print(f"\n{'=' * 60}\n")


def compare_results(file1: str, file2: str) -> None:
    with open(file1) as f:
        r1 = json.load(f)
    with open(file2) as f:
        r2 = json.load(f)

    m1 = r1["meta"]
    m2 = r2["meta"]

    print(f"\n{'=' * 72}")
    print("  Framework Benchmark Comparison")
    print(f"{'=' * 72}")
    print(
        f"  BEFORE: {m1.get('date', '?')}  |  {m1.get('service', '?')} {m1.get('version', '?')}"
    )
    print(
        f"  AFTER : {m2.get('date', '?')}  |  {m2.get('service', '?')} {m2.get('version', '?')}"
    )
    print()

    header = f"  {'Metric':<28} {'Before':>9} {'After':>9} {'Delta':>9} {'Change':>9}"
    print("  --- Latency Comparison (mean, seconds) ---")
    print(header)
    print("  " + "-" * (len(header) - 2))

    comparisons = [
        ("/health", r1["health"]["latency_stats"], r2["health"]["latency_stats"]),
        ("/readyz", r1["readyz"]["latency_stats"], r2["readyz"]["latency_stats"]),
        (
            "Thread create",
            r1["thread_lifecycle"]["create"],
            r2["thread_lifecycle"]["create"],
        ),
        ("Thread get", r1["thread_lifecycle"]["get"], r2["thread_lifecycle"]["get"]),
        (
            "Thread delete",
            r1["thread_lifecycle"]["delete"],
            r2["thread_lifecycle"]["delete"],
        ),
        (
            "Graph cold",
            r1["graph_cold_vs_cached"]["cold"],
            r2["graph_cold_vs_cached"]["cold"],
        ),
        (
            "Graph cached",
            r1["graph_cold_vs_cached"]["cached"],
            r2["graph_cold_vs_cached"]["cached"],
        ),
    ]

    for label, s1, s2 in comparisons:
        v1 = s1.get("mean", 0)
        v2 = s2.get("mean", 0)
        if v1 == 0 and v2 == 0:
            continue
        delta = v2 - v1
        pct = ((delta / v1) * 100) if v1 else 0
        direction = "↑ SLOWER" if delta > 0 else "↓ faster" if delta < 0 else "  same"
        print(
            f"  {label:<28} {v1:>9.4f} {v2:>9.4f} {delta:>+9.4f} {pct:>+7.1f}% {direction}"
        )

    print()
    print("  --- P95 Comparison (seconds) ---")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, s1, s2 in comparisons:
        v1 = s1.get("p95", 0)
        v2 = s2.get("p95", 0)
        if v1 == 0 and v2 == 0:
            continue
        delta = v2 - v1
        pct = ((delta / v1) * 100) if v1 else 0
        direction = "↑ SLOWER" if delta > 0 else "↓ faster" if delta < 0 else "  same"
        print(
            f"  {label:<28} {v1:>9.4f} {v2:>9.4f} {delta:>+9.4f} {pct:>+7.1f}% {direction}"
        )

    cold1 = r1["graph_cold_vs_cached"]["cold"].get("mean", 0)
    cached1 = r1["graph_cold_vs_cached"]["cached"].get("mean", 0)
    cold2 = r2["graph_cold_vs_cached"]["cold"].get("mean", 0)
    cached2 = r2["graph_cold_vs_cached"]["cached"].get("mean", 0)
    if cold1 > 0 and cold2 > 0:
        speedup1 = (cold1 - cached1) / cold1 * 100
        speedup2 = (cold2 - cached2) / cold2 * 100
        print(f"\n  Graph cache speedup: {speedup1:+.1f}% → {speedup2:+.1f}%")

    print(f"\n{'=' * 72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Framework-path benchmark for LangGraph agent (no LLM calls)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Iterations for health/thread tests",
    )
    parser.add_argument(
        "--graph-pairs", type=int, default=5, help="Cold/cached graph build pairs"
    )
    parser.add_argument("--agent-url", default=AGENT_URL, help="Agent base URL")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Compare two result files",
    )
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    results = run_benchmark(args.agent_url, args.iterations, args.graph_pairs)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results saved to {out}\n")


if __name__ == "__main__":
    main()
