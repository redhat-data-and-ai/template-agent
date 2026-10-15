"""Resolve logical config names to files on disk.

The stated problem was that the config namespace is fragmented and that
"each rename is breaking for every forked agent". The second half is the
one worth fixing: a layout can always be argued about, but a layout that
cannot be changed without breaking every fork is a trap.

So renames go through here. Each logical name lists its preferred path
first and any legacy paths after; a fork that still uses an old path keeps
working and gets one warning telling it what to rename.

Everything resolves under a single root — `CONFIG_PATH`, which the
container mounts at `/app/config/agent`. Siblings of that directory
(`config/auth/`, `config/sre/`, as an earlier roadmap assumed) are not
mounted in the Kubernetes path and would silently not exist.
"""

from __future__ import annotations

import os
from pathlib import Path

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Logical name -> candidate relative paths, preferred first.
#
# Adding a rename means prepending the new path, never removing the old
# one in the same change.
_CANDIDATES: dict[str, tuple[str, ...]] = {
    "prompt": ("PROMPT.md",),
    "headless_prompt": ("HEADLESS_PROMPT.md",),
    "mcp": ("integrations/mcp.json", "mcp.json"),
    "channels": ("integrations/channels.yaml",),
    "agent": ("runtime/agent.yaml", "agent.yaml"),
    "observability": ("runtime/observability.yaml", "observability.yaml"),
    "pii": ("runtime/pii.yaml", "pii.yaml"),
    "router": ("runtime/router.yaml", "router.yaml"),
    "ui": ("runtime/ui.yaml", "ui.yaml"),
    "filesystem": ("runtime/filesystem.yaml", "filesystem.yaml"),
    "memory_instructions": ("prompts/memory_instructions.md", "memory_instructions.md"),
    "guardrails": ("guardrails/guardrails.yaml", "runtime/guardrails.yaml"),
}

# Directories, resolved the same way.
_DIR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "skills": ("skills",),
    "subagents": ("subagents",),
    "policies": ("compliance/policies",),
    "evals": ("evals",),
}

# One warning per logical name, not one per read.
_warned: set[str] = set()


def config_root() -> Path:
    """The single mounted config directory."""
    return Path(
        os.getenv(
            "CONFIG_PATH",
            str(Path(__file__).resolve().parents[4] / "config" / "agent"),
        )
    )


def _resolve(
    name: str, candidates: dict[str, tuple[str, ...]], is_dir: bool
) -> Path | None:
    """First candidate that exists, warning when it is not the preferred one."""
    options = candidates.get(name)
    if not options:
        raise KeyError(f"Unknown config name: {name!r}")

    root = config_root()
    exists = Path.is_dir if is_dir else Path.is_file

    for index, relative in enumerate(options):
        path = root / relative
        if exists(path):
            if index > 0 and name not in _warned:
                _warned.add(name)
                logger.warning(
                    "Config '%s' found at the legacy path %r — rename to %r; "
                    "the old path will stop being read in a future version.",
                    name,
                    relative,
                    options[0],
                )
            return path
    return None


def candidates_for(name: str) -> tuple[str, ...]:
    """The accepted relative paths for *name*, preferred first.

    For callers that have their own notion of the config root and must
    not pick up `config_root()`. The entrypoint is the case: it reads
    `CONFIG_PATH` once into a module constant, so resolving through
    `config_file()` would consult the environment again and silently
    answer about a different directory.
    """
    options = _CANDIDATES.get(name) or _DIR_CANDIDATES.get(name)
    if not options:
        raise KeyError(f"Unknown config name: {name!r}")
    return options


def config_file(name: str) -> Path | None:
    """Path to a config file, or None when the agent does not define it."""
    return _resolve(name, _CANDIDATES, is_dir=False)


def config_dir(name: str) -> Path | None:
    """Path to a config directory, or None when absent."""
    return _resolve(name, _DIR_CANDIDATES, is_dir=True)


def preferred_path(name: str) -> Path:
    """Where a config file *should* live, whether or not it exists yet.

    Used by scaffolding and by error messages, so both name the new
    layout rather than wherever this agent happens to keep it.
    """
    options = _CANDIDATES.get(name) or _DIR_CANDIDATES.get(name)
    if not options:
        raise KeyError(f"Unknown config name: {name!r}")
    return config_root() / options[0]


def describe_layout() -> dict[str, dict[str, object]]:
    """Every logical name, where it resolved, and whether that is legacy.

    Surfaced by `validate-config` so a fork can see what it still needs
    to rename.
    """
    report: dict[str, dict[str, object]] = {}
    for name, options in {**_CANDIDATES, **_DIR_CANDIDATES}.items():
        is_dir = name in _DIR_CANDIDATES
        resolved = _resolve(name, _DIR_CANDIDATES if is_dir else _CANDIDATES, is_dir)
        relative = (
            str(resolved.relative_to(config_root())) if resolved is not None else None
        )
        report[name] = {
            "path": relative,
            "preferred": options[0],
            "legacy": relative is not None and relative != options[0],
            "present": resolved is not None,
        }
    return report
