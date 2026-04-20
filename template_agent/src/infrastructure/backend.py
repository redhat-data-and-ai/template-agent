"""Agent backend for state management and skill execution.

This module provides the backend infrastructure for agents to execute skills
in isolated Python environments. It creates dedicated virtual environments for
skill execution, manages dependencies from agent_config/pyproject.toml, and
provides a safe execution sandbox.

Why this exists:
    Skills need to run Python code with specific dependencies without polluting
    the main application environment. This backend creates isolated venvs for
    safe execution of agent skills.

Functions:
    get_backend: Get or create the configured backend instance
    initialize_backend: One-time backend initialization at app startup
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from deepagents.backends import LocalShellBackend

from template_agent.src.agent.config import agent_config
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

_SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"
_PASSTHROUGH_VARS = ("HOME", "USER", "LANG", "LC_ALL", "TZ", "TERM")


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_backend: LocalShellBackend | None = None


def _base_python() -> str:
    """Resolve the base (non-venv) Python so the agent venv is independent.

    Prefers the versioned binary (e.g. python3.12) to avoid picking up the
    UBI9 system python3 → 3.9 symlink when the app runs inside a 3.12 venv.
    """
    if sys.prefix != sys.base_prefix:
        v = sys.version_info
        base_bin = Path(sys.base_prefix) / "bin"
        for name in (f"python{v.major}.{v.minor}", "python3"):
            candidate = base_bin / name
            if candidate.exists():
                return str(candidate)
    return sys.executable


def _ensure_venv(root_dir: Path, pyproject: Path) -> Path:
    """Create an isolated venv in user cache directory and install from *pyproject*.

    The venv directory is keyed by a hash of *root_dir* **and** the contents of
    *pyproject* so a changed ``pyproject.toml`` triggers a reinstall.

    Uses /app/.cache/template-agent/venvs/ (or ~/.cache/ outside containers) to
    avoid security risks with world-readable /tmp directories on shared hosts.
    """
    project_hash = hashlib.sha256(str(root_dir.resolve()).encode()).hexdigest()[:12]
    toml_hash = hashlib.sha256(pyproject.read_bytes()).hexdigest()[:8]

    # Prefer /app/.cache inside containers (always writable); fall back to
    # the user home cache dir for local / non-container runs.
    app_cache = Path("/app/.cache")
    base_cache = app_cache if app_cache.parent.is_dir() else Path.home() / ".cache"
    cache_dir = base_cache / "template-agent" / "venvs"
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)  # User-only permissions

    venv_dir = cache_dir / f"agent-venv-{project_hash}"
    stamp = venv_dir / ".toml_hash"

    needs_install = False

    if not (venv_dir / "bin" / "python").exists():
        base = _base_python()
        logger.info(f"Creating agent venv at {venv_dir} (python: {base})")
        subprocess.run(
            [base, "-m", "venv", "--clear", str(venv_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        needs_install = True

    if not needs_install and stamp.exists() and stamp.read_text() == toml_hash:
        logger.info(f"Agent venv up-to-date ({venv_dir})")
        return venv_dir

    # If pyproject.toml changed, clear the venv to remove stale dependencies
    if stamp.exists() and stamp.read_text() != toml_hash:
        base = _base_python()
        logger.info(f"pyproject.toml changed — clearing venv at {venv_dir}")
        subprocess.run(
            [base, "-m", "venv", "--clear", str(venv_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

    pkg_dir = venv_dir / "_pkg"
    pkg_dir.mkdir(exist_ok=True)
    shutil.copy2(pyproject, pkg_dir / "pyproject.toml")

    pip = str(venv_dir / "bin" / "pip")
    logger.info(f"Installing dependencies from {pyproject.name}")
    result = subprocess.run(
        [pip, "install", "--quiet", str(pkg_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pip install failed: {result.stderr.strip()}")

    stamp.write_text(toml_hash)
    return venv_dir


def _build_env(venv_dir: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Minimal env: allowlisted host vars + venv activation + optional overrides."""
    env = {k: os.environ[k] for k in _PASSTHROUGH_VARS if k in os.environ}
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = f"{venv_dir}/bin:{_SYSTEM_PATH}"
    if extra:
        env.update(extra)
    return env


def create_backend(
    root_dir: Path,
    pyproject: Path,
    *,
    timeout: int = 120,
    max_output_bytes: int = 100_000,
    extra_env: dict[str, str] | None = None,
) -> LocalShellBackend:
    """Create a :class:`LocalShellBackend` backed by an isolated agent venv.

    Args:
        root_dir: Shell working directory.
        pyproject: Path to a ``pyproject.toml`` whose dependencies are installed.
        timeout: Default per-command timeout in seconds.
        max_output_bytes: Max captured output before truncation.
        extra_env: Extra env vars (highest priority).
    """
    if not pyproject.is_file():
        raise FileNotFoundError(f"pyproject.toml not found: {pyproject}")

    venv_dir = _ensure_venv(root_dir, pyproject)
    env = _build_env(venv_dir, extra_env)

    logger.info(f"Backend ready — venv={venv_dir}, pyproject={pyproject}")
    return LocalShellBackend(
        root_dir=str(root_dir),
        virtual_mode=False,
        timeout=timeout,
        max_output_bytes=max_output_bytes,
        env=env,
    )


def get_backend(
    root_dir: Path | None = None,
    pyproject: Path | None = None,
    *,
    timeout: int = 120,
    max_output_bytes: int = 100_000,
    extra_env: dict[str, str] | None = None,
) -> LocalShellBackend:
    """Return the singleton backend, creating it on the first call.

    Subsequent calls return the same instance regardless of arguments.
    When *root_dir* or *pyproject* are ``None`` the module-level defaults
    (``_REPO_ROOT`` / ``agent_config.get_pyproject_path()``) are used.
    """
    global _backend  # noqa: PLW0603
    if _backend is None:
        _backend = create_backend(
            root_dir or _REPO_ROOT,
            pyproject or agent_config.get_pyproject_path(),
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            extra_env=extra_env,
        )
    return _backend


def initialize_backend() -> LocalShellBackend:
    """Pre-initialize the singleton backend at server startup.

    Calling this early avoids the venv-creation penalty on the first request.
    """
    logger.info("Pre-initializing backend (venv + dependency install)")
    backend = get_backend()
    logger.info("Backend initialization complete")
    return backend
