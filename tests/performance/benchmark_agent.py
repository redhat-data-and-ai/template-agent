#!/usr/bin/env python3
"""Agent performance benchmark for LangGraph Platform (Aegra).

Measures TTFT, total response time, and health latency against a running
template-agent instance. Uses only stdlib so it runs without extra pip installs.

Usage:
    python3 benchmark_agent.py                         # 20 runs, default URL
    python3 benchmark_agent.py --runs 10 --output results.json
    python3 benchmark_agent.py --compare before.json after.json
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
WARMUP_RUNS = 3
DEFAULT_RUNS = 20
STREAM_TIMEOUT = 120  # seconds

SIMPLE_PROMPTS = [
    "What is 2+2?",
    "Hello, how are you?",
    "Summarize what you can do in one sentence.",
    "What is the capital of France?",
    "Explain the Pythagorean theorem briefly.",
]

SUBAGENT_PROMPTS = [
    "Can you help me draft a short email to my team about a meeting tomorrow?",
    "Search for recent news about AI agents and summarize the top result.",
    "Calculate the BMI for a person who is 180cm tall and weighs 75kg.",
]


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


def _stream_post(
    url: str, body: dict, timeout: float = STREAM_TIMEOUT
) -> tuple[float, float, str]:
    """POST to a streaming endpoint and return (ttft, total_time, full_response_text).

    TTFT = time from request start to first SSE data line with message content.
    """
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )

    t0 = time.perf_counter()
    ttft: float | None = None
    response_text = ""

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")

            if line.startswith("data: "):
                payload = line[6:]
                if ttft is None and payload and payload != "{}":
                    try:
                        obj = json.loads(payload)
                        if _has_content(obj):
                            ttft = time.perf_counter() - t0
                            response_text = _extract_text(obj)
                    except json.JSONDecodeError:
                        pass

                if ttft is not None:
                    try:
                        obj = json.loads(payload)
                        txt = _extract_text(obj)
                        if txt:
                            response_text = txt
                    except json.JSONDecodeError:
                        pass

    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    return ttft, total, response_text


def _has_content(obj: Any) -> bool:
    """Return True if the SSE payload contains actual message content."""
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    return True
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("text", "").strip():
                            return True
    if isinstance(obj, dict):
        content = obj.get("content")
        if isinstance(content, str) and content.strip():
            return True
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("text", "").strip():
                    return True
    return False


def _extract_text(obj: Any) -> str:
    """Extract text from an SSE messages/partial payload."""
    items = obj if isinstance(obj, list) else [obj]
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            joined = " ".join(parts).strip()
            if joined:
                return joined
    return ""


def create_thread(base_url: str) -> str:
    result = _post_json(f"{base_url}/threads", {})
    return result["thread_id"]


def stream_message(
    base_url: str, thread_id: str, message: str
) -> tuple[float, float, str]:
    url = f"{base_url}/threads/{thread_id}/runs/stream"
    body = {
        "assistant_id": ASSISTANT_ID,
        "input": {"messages": [{"role": "human", "content": message}]},
        "stream_mode": ["messages"],
    }
    return _stream_post(url, body)


def measure_health(base_url: str) -> tuple[float, dict]:
    t0 = time.perf_counter()
    data = _get_json(f"{base_url}/health")
    latency = time.perf_counter() - t0
    return latency, data


def get_version(base_url: str) -> dict:
    try:
        return _get_json(f"{base_url}/version")
    except Exception:
        return {}


def get_info(base_url: str) -> dict:
    try:
        return _get_json(f"{base_url}/info")
    except Exception:
        return {}


def get_metrics(base_url: str) -> dict | None:
    try:
        return _get_json(f"{base_url}/api/metrics")
    except urllib.error.HTTPError:
        return None
    except Exception:
        return None


def compute_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "p95": 0,
            "p99": 0,
            "count": 0,
        }
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)
    return {
        "min": round(sorted_vals[0], 4),
        "max": round(sorted_vals[-1], 4),
        "mean": round(statistics.mean(sorted_vals), 4),
        "median": round(statistics.median(sorted_vals), 4),
        "p95": round(sorted_vals[p95_idx], 4),
        "p99": round(sorted_vals[p99_idx], 4),
        "count": n,
    }


def run_benchmark(
    base_url: str, num_runs: int, warmup: int = WARMUP_RUNS
) -> dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"  Agent Benchmark — {base_url}")
    print(f"{'=' * 60}\n")

    # ── Version & info ─────────────────────────────────────────
    version_data = get_version(base_url)
    info_data = get_info(base_url)
    print(f"  Service : {version_data.get('service', 'unknown')}")
    print(f"  Version : {version_data.get('version', 'unknown')}")
    print(f"  Agent   : {info_data.get('name', 'unknown')}")
    print()

    # ── Health baseline ────────────────────────────────────────
    print("  [1/4] Health baseline ...", end=" ", flush=True)
    health_latencies: list[float] = []
    health_data: dict = {}
    for _ in range(3):
        lat, hd = measure_health(base_url)
        health_latencies.append(lat)
        health_data = hd
    print(f"OK  (avg {statistics.mean(health_latencies) * 1000:.1f}ms)")

    # ── OTEL metrics snapshot (before) ─────────────────────────
    print("  [2/4] OTEL metrics snapshot ...", end=" ", flush=True)
    metrics_before = get_metrics(base_url)
    if metrics_before:
        print("OK")
    else:
        print("NOT AVAILABLE (endpoint returned 404)")

    # ── Warmup runs ────────────────────────────────────────────
    warmup + num_runs
    all_prompts = SIMPLE_PROMPTS + SUBAGENT_PROMPTS

    print(f"  [3/4] Warmup ({warmup} runs) ...", flush=True)
    for i in range(warmup):
        prompt = all_prompts[i % len(all_prompts)]
        try:
            tid = create_thread(base_url)
            ttft, total, text = stream_message(base_url, tid, prompt)
            preview = text[:60].replace("\n", " ") if text else "(empty)"
            print(
                f"         warmup {i + 1}/{warmup}: TTFT={ttft:.2f}s total={total:.2f}s  → {preview}"
            )
        except Exception as exc:
            print(f"         warmup {i + 1}/{warmup}: FAILED — {exc}")

    # ── Measurement runs ───────────────────────────────────────
    print(f"  [4/4] Benchmark ({num_runs} runs) ...", flush=True)
    ttft_values: list[float] = []
    total_values: list[float] = []
    simple_ttft: list[float] = []
    simple_total: list[float] = []
    subagent_ttft: list[float] = []
    subagent_total: list[float] = []
    run_details: list[dict] = []
    errors: list[str] = []

    for i in range(num_runs):
        prompt_idx = i % len(all_prompts)
        prompt = all_prompts[prompt_idx]
        is_subagent = prompt in SUBAGENT_PROMPTS
        label = "subagent" if is_subagent else "simple"

        try:
            t_create_start = time.perf_counter()
            tid = create_thread(base_url)
            t_create = time.perf_counter() - t_create_start

            ttft, total, text = stream_message(base_url, tid, prompt)
            ttft_values.append(ttft)
            total_values.append(total)

            if is_subagent:
                subagent_ttft.append(ttft)
                subagent_total.append(total)
            else:
                simple_ttft.append(ttft)
                simple_total.append(total)

            detail = {
                "run": i + 1,
                "prompt": prompt,
                "type": label,
                "thread_id": tid,
                "thread_create_time": round(t_create, 4),
                "ttft": round(ttft, 4),
                "total_time": round(total, 4),
                "response_preview": (text[:120] if text else ""),
            }
            run_details.append(detail)

            preview = text[:50].replace("\n", " ") if text else "(empty)"
            print(
                f"         run {i + 1:>2}/{num_runs}: [{label:>8}] "
                f"TTFT={ttft:.2f}s total={total:.2f}s  → {preview}"
            )
        except Exception as exc:
            errors.append(f"run {i + 1}: {exc}")
            print(f"         run {i + 1:>2}/{num_runs}: [{label:>8}] FAILED — {exc}")

    # ── Health latency (measurement set) ───────────────────────
    for _ in range(5):
        lat, _ = measure_health(base_url)
        health_latencies.append(lat)

    # ── OTEL metrics snapshot (after) ──────────────────────────
    metrics_after = get_metrics(base_url)

    # ── Metadata from first streaming response ─────────────────
    # Attempt to capture lc_versions from a lightweight extra call
    lc_versions = _probe_lc_versions(base_url)

    # ── Build results ──────────────────────────────────────────
    now_iso = datetime.now(timezone.utc).isoformat()
    results: dict[str, Any] = {
        "meta": {
            "agent_url": base_url,
            "service": version_data.get("service", "unknown"),
            "version": version_data.get("version", "unknown"),
            "agent_name": info_data.get("name", "unknown"),
            "lc_versions": lc_versions,
            "date": now_iso,
            "warmup_runs": warmup,
            "measurement_runs": num_runs,
            "prompts_simple": SIMPLE_PROMPTS,
            "prompts_subagent": SUBAGENT_PROMPTS,
        },
        "health": {
            "status": health_data,
            "latency_stats": compute_stats(health_latencies),
        },
        "latency": {
            "ttft_all": compute_stats(ttft_values),
            "total_all": compute_stats(total_values),
            "ttft_simple": compute_stats(simple_ttft),
            "total_simple": compute_stats(simple_total),
            "ttft_subagent": compute_stats(subagent_ttft),
            "total_subagent": compute_stats(subagent_total),
        },
        "otel_metrics": {
            "before": metrics_before,
            "after": metrics_after,
            "available": metrics_before is not None,
        },
        "run_details": run_details,
        "errors": errors,
    }

    _print_report(results)
    return results


def _probe_lc_versions(base_url: str) -> dict:
    """Send a tiny message and parse lc_versions from the messages/metadata event."""
    try:
        tid = create_thread(base_url)
        url = f"{base_url}/threads/{tid}/runs/stream"
        body = {
            "assistant_id": ASSISTANT_ID,
            "input": {"messages": [{"role": "human", "content": "hi"}]},
            "stream_mode": ["messages"],
        }
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
                if line.startswith("data: ") and "lc_versions" in line:
                    payload = json.loads(line[6:])
                    for _key, meta_block in payload.items():
                        if isinstance(meta_block, dict):
                            md = meta_block.get("metadata", {})
                            versions = md.get("lc_versions", {})
                            if versions:
                                return versions
    except Exception:
        pass
    return {}


def _print_report(results: dict) -> None:
    meta = results["meta"]
    lc = meta.get("lc_versions", {})

    print(f"\n{'=' * 60}")
    print("  Agent Benchmark Results")
    print(f"{'=' * 60}")
    print(f"  Agent    : {meta['agent_url']}")
    print(f"  Service  : {meta['service']} {meta['version']}")
    if lc:
        aegra_v = lc.get("deepagents", "?")
        lg_v = lc.get("langchain-core", "?")
        lc_v = lc.get("langchain", "?")
        print(
            f"  Aegra    : {aegra_v}  |  LangChain-core: {lg_v}  |  LangChain: {lc_v}"
        )
    print(f"  Date     : {meta['date']}")
    print(f"  Runs     : {meta['measurement_runs']} (+ {meta['warmup_runs']} warmup)")
    print()

    # Latency table
    print("  --- Latency Statistics (seconds) ---")
    header = f"  {'Metric':<22} {'min':>7} {'max':>7} {'mean':>7} {'median':>7} {'p95':>7} {'p99':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    lat = results["latency"]
    for label, key in [
        ("TTFT (all)", "ttft_all"),
        ("Total Response (all)", "total_all"),
        ("TTFT (simple)", "ttft_simple"),
        ("Total (simple)", "total_simple"),
        ("TTFT (subagent)", "ttft_subagent"),
        ("Total (subagent)", "total_subagent"),
    ]:
        s = lat[key]
        if s["count"] == 0:
            continue
        print(
            f"  {label:<22} {s['min']:>7.3f} {s['max']:>7.3f} "
            f"{s['mean']:>7.3f} {s['median']:>7.3f} {s['p95']:>7.3f} {s['p99']:>7.3f}"
        )

    hs = results["health"]["latency_stats"]
    print(
        f"  {'Health endpoint':<22} {hs['min']:>7.3f} {hs['max']:>7.3f} "
        f"{hs['mean']:>7.3f} {hs['median']:>7.3f} {hs['p95']:>7.3f} {hs['p99']:>7.3f}"
    )

    # Health details
    print()
    print("  --- Infrastructure (from /health) ---")
    for k, v in results["health"]["status"].items():
        print(f"  {k}: {v}")

    # OTEL metrics
    print()
    print("  --- OTEL Metrics ---")
    if results["otel_metrics"]["available"]:
        print("  (delta between before/after snapshots)")
        print(
            f"  Before: {json.dumps(results['otel_metrics']['before'], indent=2)[:300]}"
        )
    else:
        print("  /api/metrics endpoint not available on this deployment.")

    # Errors
    if results["errors"]:
        print()
        print(f"  --- Errors ({len(results['errors'])}) ---")
        for e in results["errors"]:
            print(f"  ⚠  {e}")

    print()
    print(f"{'=' * 60}\n")


def compare_results(file1: str, file2: str) -> None:
    with open(file1) as f:
        r1 = json.load(f)
    with open(file2) as f:
        r2 = json.load(f)

    m1 = r1["meta"]
    m2 = r2["meta"]
    lc1 = m1.get("lc_versions", {})
    lc2 = m2.get("lc_versions", {})

    print(f"\n{'=' * 72}")
    print("  Benchmark Comparison")
    print(f"{'=' * 72}")
    print(
        f"  BEFORE: {m1.get('date', '?')}  |  {m1.get('service', '?')} {m1.get('version', '?')}"
    )
    if lc1:
        print(
            f"          Aegra={lc1.get('deepagents', '?')}  LG-core={lc1.get('langchain-core', '?')}"
        )
    print(
        f"  AFTER : {m2.get('date', '?')}  |  {m2.get('service', '?')} {m2.get('version', '?')}"
    )
    if lc2:
        print(
            f"          Aegra={lc2.get('deepagents', '?')}  LG-core={lc2.get('langchain-core', '?')}"
        )
    print()

    header = f"  {'Metric':<22} {'Before':>9} {'After':>9} {'Delta':>9} {'Change':>9}"
    print("  --- Latency Comparison (mean, seconds) ---")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, key in [
        ("TTFT (all)", "ttft_all"),
        ("Total Response", "total_all"),
        ("TTFT (simple)", "ttft_simple"),
        ("Total (simple)", "total_simple"),
        ("TTFT (subagent)", "ttft_subagent"),
        ("Total (subagent)", "total_subagent"),
    ]:
        s1 = r1["latency"].get(key, {})
        s2 = r2["latency"].get(key, {})
        v1 = s1.get("mean", 0)
        v2 = s2.get("mean", 0)
        if v1 == 0 and v2 == 0:
            continue
        delta = v2 - v1
        pct = ((delta / v1) * 100) if v1 else 0
        direction = "↑ SLOWER" if delta > 0 else "↓ faster" if delta < 0 else "  same"
        print(
            f"  {label:<22} {v1:>9.3f} {v2:>9.3f} {delta:>+9.3f} {pct:>+7.1f}% {direction}"
        )

    # P95 comparison
    print()
    print("  --- P95 Comparison (seconds) ---")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, key in [
        ("TTFT (all)", "ttft_all"),
        ("Total Response", "total_all"),
    ]:
        s1 = r1["latency"].get(key, {})
        s2 = r2["latency"].get(key, {})
        v1 = s1.get("p95", 0)
        v2 = s2.get("p95", 0)
        if v1 == 0 and v2 == 0:
            continue
        delta = v2 - v1
        pct = ((delta / v1) * 100) if v1 else 0
        direction = "↑ SLOWER" if delta > 0 else "↓ faster" if delta < 0 else "  same"
        print(
            f"  {label:<22} {v1:>9.3f} {v2:>9.3f} {delta:>+9.3f} {pct:>+7.1f}% {direction}"
        )

    # Health comparison
    hs1 = r1["health"]["latency_stats"]
    hs2 = r2["health"]["latency_stats"]
    print()
    print(
        f"  Health endpoint mean: {hs1.get('mean', 0) * 1000:.1f}ms → "
        f"{hs2.get('mean', 0) * 1000:.1f}ms"
    )

    print(f"\n{'=' * 72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a LangGraph agent")
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS, help="Number of measurement runs"
    )
    parser.add_argument("--agent-url", default=AGENT_URL, help="Agent base URL")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Compare two result files",
    )
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_RUNS, help="Number of warmup runs"
    )
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    results = run_benchmark(args.agent_url, args.runs, warmup=args.warmup)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results saved to {out}\n")


if __name__ == "__main__":
    main()
