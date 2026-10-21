"""Validate an agent's `config/` directory.

These templates are config-driven: building an agent means editing
`config/`, not Python. That only holds if a mistake in `config/` is
caught up front. Without this, a typo surfaces as a runtime error on the
first request, or worse, as a silent fallback to a default.

The shipped `config/` is a worked example (a health assistant), so this
also flags example content left in place — forking and forgetting to
change the prompt ships someone else's agent.

    python -m deep_agent.src.agent.config.validate
    make validate-config
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from deep_agent.src.settings import settings


@dataclass
class Finding:
    """One problem, located precisely enough to fix without searching."""

    file: str
    location: str
    message: str
    level: str = "error"

    def render(self) -> str:
        """One finding as an aligned line for the terminal."""
        where = f"{self.file}:{self.location}" if self.location else self.file
        return f"  {self.level.upper():7} {where}\n          {self.message}"


@dataclass
class Report:
    """Everything config validation found, errors and warnings together."""

    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        """Findings that should stop a deployment."""
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        """Findings worth reading that should not stop a deployment."""
        return [f for f in self.findings if f.level == "warning"]

    def add(self, file: str, location: str, message: str, level: str = "error") -> None:
        """Record a finding against a file and a location inside it."""
        self.findings.append(Finding(file, location, message, level))


def _pydantic_location(error: dict[str, Any]) -> str:
    """Render a pydantic error path as a config path a human can grep."""
    parts = [str(p) for p in error.get("loc", ())]
    return ".".join(parts) if parts else "(root)"


def _first_present(config_dir: Path, name: str) -> str | None:
    """The accepted relative path for *name* that exists, preferred first."""
    from deep_agent.src.agent.config.paths import candidates_for

    try:
        options = candidates_for(name)
    except KeyError:
        return None
    for relative in options:
        if (config_dir / relative).is_file():
            return relative
    return None


def _validate_model(
    report: Report, path: Path, rel: str, model: type[BaseModel], section: str | None
) -> BaseModel | None:
    """Parse one YAML file into a model, reporting field-level errors."""
    if not path.is_file():
        return None

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        report.add(rel, "", f"Not valid YAML: {exc}")
        return None

    if section is not None:
        raw = raw.get(section) or {}

    try:
        return model(**raw)
    except ValidationError as exc:
        for error in exc.errors():
            report.add(rel, _pydantic_location(error), error["msg"])
    except Exception as exc:
        report.add(rel, "", str(exc))
    return None


# Where the example-content markers live. Kept in config, not here:
# a brand name in a .py is a name somebody has to find in source when
# they fork, and the needles have to track whatever the shipped config
# carries anyway.
_MARKERS_FILE = settings.TEMPLATE_MARKERS_FILE


def _example_markers(config_dir: Path) -> list[tuple[str, str, str]]:
    """Load the marker list. A missing or unreadable file means none."""
    path = config_dir / _MARKERS_FILE
    if not path.is_file():
        return []
    try:
        raw = yaml.safe_load(path.read_text()) or {}
        entries = raw.get("markers") or []
    except Exception:
        # A malformed marker file must not fail validation of the
        # config it is describing.
        return []

    markers: list[tuple[str, str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        needle = str(entry.get("needle") or "")
        rel = str(entry.get("file") or "")
        if needle and rel:
            markers.append(
                (needle, rel, str(entry.get("message") or "is still the example's"))
            )
    return markers


def _check_example_leftovers(report: Report, config_dir: Path) -> None:
    """Warn about demo content a fork should have replaced."""
    for needle, rel, message in _example_markers(config_dir):
        path = config_dir / rel
        if not path.is_file():
            continue
        if needle in path.read_text():
            report.add(rel, "", f"{message} ({needle!r})", level="warning")

    # No demo-skill check. The shipped skills used to be a health
    # tutorial — bmi-report, client-intake, email-formatter — which no
    # fork would want and which therefore deserved a warning. They have
    # been replaced by skills that are genuinely reusable: a fork that
    # also uses GitLab or Atlan should keep data-discovery and
    # merge-request-review, so warning about them would be noise, and a
    # warning everyone learns to ignore weakens the ones above it.


def _check_placeholder_secrets(report: Report, config_dir: Path) -> None:
    """Catch unreplaced placeholders that would fail only at first use."""
    # Both layouts, because this check silently did nothing the moment
    # the file moved to its preferred path — a validator that passes
    # because it looked in the wrong place is worse than no validator.
    mcp = config_dir / "integrations" / "mcp.json"
    if not mcp.is_file():
        mcp = config_dir / "mcp.json"
    if not mcp.is_file():
        return

    text = mcp.read_text()
    if 'ssl_verify": false' in text or '"ssl_verify": false' in text:
        report.add(
            "mcp.json",
            "ssl_verify",
            "TLS verification disabled; forced on in production but unsafe elsewhere",
            level="warning",
        )


def validate_config(config_dir: Path) -> Report:
    """Validate every config file we have a model for."""
    from deep_agent.src.agent.config.cache import CacheFileConfig
    from deep_agent.src.agent.config.filesystem import FilesystemFileConfig
    from deep_agent.src.agent.config.middleware import (
        ExecutionConfig,
        MiddlewareDefaults,
    )
    from deep_agent.src.agent.config.opa import OpaFileConfig
    from deep_agent.src.agent.config.otel import OtelFileConfig
    from deep_agent.src.agent.config.providers import ProvidersFileConfig
    from deep_agent.src.channels.config import ChannelsFileConfig
    from deep_agent.src.routing.config import RouterConfig

    report = Report()

    if not config_dir.is_dir():
        report.add(str(config_dir), "", "Config directory does not exist")
        return report

    # (config name, model, section within the file)
    #
    # By name, not by path. The preferred layout was hardcoded here,
    # so a fork still on the flat layout — `agent.yaml` at the root,
    # which `paths.py` accepts and the loader reads — had every one
    # of these quietly skipped and got no field validation at all.
    targets: list[tuple[str, type[BaseModel], str | None]] = [
        # agent.yaml's `middleware:` block is the *defaults* shape;
        # MiddlewareFileConfig describes a separate defaults/profiles file.
        ("agent", MiddlewareDefaults, "middleware"),
        ("agent", ProvidersFileConfig, "providers"),
        ("agent", CacheFileConfig, "cache"),
        ("agent", FilesystemFileConfig, "filesystem"),
        ("agent", OpaFileConfig, "opa"),
        ("agent", ExecutionConfig, "execution"),
        ("observability", OtelFileConfig, None),
        ("router", RouterConfig, None),
        ("channels", ChannelsFileConfig, None),
    ]

    for name, model, section in targets:
        rel = _first_present(config_dir, name)
        if rel is None:
            continue
        _validate_model(report, config_dir / rel, rel, model, section)

    _check_required_files(report, config_dir)

    _check_example_leftovers(report, config_dir)
    _check_placeholder_secrets(report, config_dir)
    _check_legacy_paths(report, config_dir)
    return report


def _check_required_files(report: Report, config_dir: Path) -> None:
    """Assert the files the agent cannot start without are present.

    Resolved through `config.paths`, not by literal path: a config that
    adopted the new layout would otherwise be reported as missing the
    file it correctly renamed.
    """
    from deep_agent.src.agent.config.paths import _CANDIDATES, config_root

    required = ("prompt", "mcp", "agent")
    use_resolver = config_dir == config_root()

    for name in required:
        candidates = _CANDIDATES[name]
        if use_resolver:
            from deep_agent.src.agent.config.paths import config_file

            found = config_file(name) is not None
        else:
            found = any((config_dir / rel).is_file() for rel in candidates)

        if not found:
            report.add(
                candidates[0],
                "",
                f"Required config '{name}' is missing "
                f"(looked for: {', '.join(candidates)})",
            )


def _check_legacy_paths(report: Report, config_dir: Path) -> None:
    """Report config files still at a pre-rename path.

    Advisory: the resolver keeps reading them, but they will stop being
    read in a future version.
    """
    from deep_agent.src.agent.config.paths import config_root, describe_layout

    # describe_layout reads CONFIG_PATH; only meaningful for that root.
    if config_dir != config_root():
        return

    for name, info in sorted(describe_layout().items()):
        if info["legacy"]:
            report.add(
                str(info["path"]),
                "",
                f"legacy path for '{name}' — rename to {info['preferred']!r}",
                level="warning",
            )


def main() -> int:
    """Print the report. Non-zero exit on any error."""
    config_dir = Path(
        os.getenv(
            "CONFIG_PATH",
            str(Path(__file__).resolve().parents[4] / "config" / "agent"),
        )
    )

    report = validate_config(config_dir)

    print(f"Validating {config_dir}\n")
    for finding in report.findings:
        print(finding.render())

    if not report.findings:
        print("  OK — no problems found")

    print(f"\n{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    # Warnings are advisory: the example is expected to trip them.
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
