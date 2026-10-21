"""Render kustomize overlays so tests can assert on the real output.

Deployment is part of the codebase: a manifest that violates an
OpenShift constraint fails at rollout, which is the most expensive
place to find out. The constraints in the PRD's OpenShift section are
prose; these make them executable.

Rendering (rather than reading the source YAML) is the point — patches,
components and generators all have to compose correctly for the
assertion to mean anything.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
OVERLAYS = REPO / "deployment" / "overlays"


def kustomize_available() -> bool:
    return shutil.which("kubectl") is not None or shutil.which("kustomize") is not None


@lru_cache(maxsize=8)
def render(overlay: str) -> tuple[dict[str, Any], ...]:
    """Render one overlay to a tuple of manifests.

    Cached and immutable because rendering shells out and every test
    would otherwise pay for it.
    """
    path = OVERLAYS / overlay
    if not path.is_dir():
        raise FileNotFoundError(f"No overlay at {path}")

    binary = (
        ["kubectl", "kustomize"] if shutil.which("kubectl") else ["kustomize", "build"]
    )
    result = subprocess.run(
        [*binary, str(path)], capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(f"kustomize failed for {overlay}:\n{result.stderr}")

    import yaml

    return tuple(d for d in yaml.safe_load_all(result.stdout) if d)


def of_kind(manifests: tuple[dict[str, Any], ...], kind: str) -> list[dict[str, Any]]:
    return [m for m in manifests if m.get("kind") == kind]


def pod_specs(manifests: tuple[dict[str, Any], ...]) -> list[tuple[str, dict]]:
    """Every pod template in the overlay, with the workload name.

    Covers Deployment, StatefulSet, Job and CronJob so a new workload
    cannot quietly skip the security assertions.
    """
    specs: list[tuple[str, dict]] = []
    for manifest in manifests:
        kind = manifest.get("kind")
        name = (manifest.get("metadata") or {}).get("name", "?")
        spec = manifest.get("spec") or {}
        if kind in {"Deployment", "StatefulSet", "DaemonSet"}:
            template = (spec.get("template") or {}).get("spec")
            if template:
                specs.append((f"{kind}/{name}", template))
        elif kind == "Job":
            template = (spec.get("template") or {}).get("spec")
            if template:
                specs.append((f"{kind}/{name}", template))
        elif kind == "CronJob":
            template = (
                ((spec.get("jobTemplate") or {}).get("spec") or {}).get("template")
                or {}
            ).get("spec")
            if template:
                specs.append((f"{kind}/{name}", template))
    return specs


def all_containers(pod_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Init containers count: they run with the same SCC."""
    return list(pod_spec.get("initContainers") or []) + list(
        pod_spec.get("containers") or []
    )


def dumps(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, sort_keys=True)
