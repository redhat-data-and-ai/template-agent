#!/usr/bin/env python
"""Answer "does this already exist?" before scoping work.

Six times in one session, work was scoped against a capability that
already existed — in Aegra, on `main`, or on an unmerged branch. Each
time the cost was a day of building something that had to be dropped.
Checking is seconds; this makes it one command.

    python scripts/whats_there.py queue
    make whats-there Q=approval

Reports, for a keyword:
  1. Aegra — the installed package's modules and routes
  2. Agent  — modules, routes and config on this branch
  3. Branches — unmerged work touching matching files
"""

from __future__ import annotations

import contextlib
import io as _io
import logging
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Importing the Aegra app emits a page of startup logs that buries the
# answer. The point of this tool is a short, readable report.
logging.disable(logging.CRITICAL)


def _grep(root: Path, term: str, suffixes: tuple[str, ...]) -> list[str]:
    """Files whose name or contents mention the term.

    Name-only matching misses the common case: Aegra's worker queue lives
    in `services/worker_executor.py`, which does not contain "queue".
    """
    hits: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in suffixes:
            continue
        rel = str(path.relative_to(root))
        if term in rel.lower():
            hits.add(rel)
            continue
        try:
            if term in path.read_text(errors="ignore").lower():
                hits.add(rel)
        except OSError:
            continue
    return sorted(hits)


def _print_section(title: str, hits: list[str]) -> None:
    print(f"\n{title}")
    if not hits:
        print("    (nothing)")
        return
    for hit in hits[:12]:
        print(f"    {hit}")
    if len(hits) > 12:
        print(f"    … and {len(hits) - 12} more")


def _aegra_hits(term: str) -> list[str]:
    """Modules and routes in the installed Aegra package."""
    try:
        import aegra_api
    except ImportError:
        return ["aegra_api not installed"]

    root = Path(aegra_api.__file__).parent
    hits = _grep(root, term, (".py",))

    try:
        from aegra_api.main import create_app
        from fastapi.routing import APIRoute

        paths: set[str] = set()

        def walk(routes: list) -> None:
            for route in routes:
                if isinstance(route, APIRoute):
                    if term in route.path.lower():
                        paths.add(f"route {sorted(route.methods)[0]} {route.path}")
                    continue
                nested = getattr(route, "original_router", None)
                if nested is not None:
                    walk(list(nested.routes))
                elif getattr(route, "routes", None):
                    walk(list(route.routes))

        # create_app() is chatty even with logging disabled.
        with (
            contextlib.redirect_stdout(_io.StringIO()),
            contextlib.redirect_stderr(_io.StringIO()),
        ):
            app = create_app()
        walk(list(app.routes))
        hits.extend(sorted(paths))
    except Exception as exc:
        hits.append(f"(could not introspect routes: {type(exc).__name__})")

    return hits


def _agent_hits(term: str) -> list[str]:
    """Modules and config on this branch."""
    hits: list[str] = []
    for base in ("deep_agent", "config"):
        root = REPO / base
        if not root.is_dir():
            continue
        hits += [
            f"{base}/{rel}"
            for rel in _grep(root, term, (".py", ".yaml", ".yml", ".json", ".md"))
        ]
    return sorted(set(hits))


def _branch_hits(term: str) -> list[str]:
    """Unmerged branches touching files whose names match."""
    try:
        branches = subprocess.run(
            ["git", "branch", "-r", "--format=%(refname:short)"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.split()
    except Exception:
        return ["(git unavailable)"]

    hits: list[str] = []
    for branch in branches:
        if branch.endswith("/HEAD") or branch.endswith("/main"):
            continue
        try:
            ahead = subprocess.run(
                ["git", "rev-list", "--count", f"origin/main..{branch}"],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
            if ahead in ("", "0"):
                continue
            files = subprocess.run(
                ["git", "diff", "--name-only", "origin/main", branch],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.splitlines()
        except Exception:
            continue
        matched = [f for f in files if term in f.lower()]
        if matched:
            hits.append(f"{branch} ({ahead} ahead) — {matched[0]}")
    return hits


def main() -> int:
    """Summarise what is configured. Returns a process exit code."""
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print(__doc__)
        return 2

    term = sys.argv[1].strip().lower()
    print(f'Searching for "{term}" — check all three before scoping work.')

    _print_section("AEGRA (a dependency — do not reimplement)", _aegra_hits(term))
    _print_section("AGENT (this branch)", _agent_hits(term))
    _print_section("UNMERGED BRANCHES (someone may be on it)", _branch_hits(term))

    print(
        "\nAlso check: Aegra's docs/feature-support.mdx for 'coming soon',\n"
        "and whether the design survives OpenShift (restricted SCC,\n"
        "arbitrary UID, internal Route shard, multi-pod)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
