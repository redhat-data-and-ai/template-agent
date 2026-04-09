"""Locked-down LocalShellBackend with an isolated venv.

Creates a fresh virtual environment in the system temp directory (never reusing
the project's own ``.venv``).  Dependencies are driven by a caller-supplied
``pyproject.toml`` which is copied into the venv and installed via
``pip install .``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from deepagents.backends import LocalShellBackend

from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"
_PASSTHROUGH_VARS = ("HOME", "USER", "LANG", "LC_ALL", "TZ", "TERM")


def _base_python() -> str:
    """Resolve the base (non-venv) Python so the agent venv is independent."""
    if sys.prefix != sys.base_prefix:
        candidate = Path(sys.base_prefix) / "bin" / "python3"
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _ensure_venv(root_dir: Path, pyproject: Path) -> Path:
    """Create an isolated venv under ``$TMPDIR`` and install from *pyproject*.

    The venv directory is keyed by a hash of *root_dir* **and** the contents of
    *pyproject* so a changed ``pyproject.toml`` triggers a reinstall.
    """
    project_hash = hashlib.sha256(str(root_dir.resolve()).encode()).hexdigest()[:12]
    toml_hash = hashlib.sha256(pyproject.read_bytes()).hexdigest()[:8]
    venv_dir = Path(tempfile.gettempdir()) / f"agent-venv-{project_hash}"
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
    env["PATH"] = f"{venv_dir}/bin:{SYSTEM_PATH}"
    if extra:
        env.update(extra)
    return env


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "agent_config"

_backend: LocalShellBackend | None = None


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
    (``_REPO_ROOT`` / ``_CONFIG_DIR``) are used.
    """
    global _backend  # noqa: PLW0603
    if _backend is None:
        _backend = create_backend(
            root_dir or _REPO_ROOT,
            pyproject or (_CONFIG_DIR / "pyproject.toml"),
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
